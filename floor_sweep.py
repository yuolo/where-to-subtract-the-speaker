"""Factor model under anti-speaker pressure and the CRNN input-CMN arm"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from nc_bridge import nc, fresh_config
from mi_estimation import mi_lower_bound_bits

BASE_LAMBDA_AFF_SPK_ADV = 0.35
BASE_LAMBDA_ORTH = 0.05
OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "floor_sweep")

_W: Dict[str, object] = {}


def apply_input_cmn(cache: Dict, dfs) -> Dict:
    new_cache = dict(cache)
    for df in dfs:
        for spk, grp in df.groupby(df["speaker"].astype(str)):
            paths = grp["path"].astype(str).tolist()
            total = None
            n_frames = 0
            for p in paths:
                lm = cache[p]["logmel"]
                total = lm.sum(axis=1) if total is None else total + lm.sum(axis=1)
                n_frames += lm.shape[1]
            mean = (total / max(n_frames, 1)).astype(np.float32)[:, None]
            for p in paths:
                entry = dict(cache[p])
                entry["logmel"] = (cache[p]["logmel"] - mean).astype(np.float32)
                new_cache[p] = entry
    return new_cache


def apply_pressure(cfg: "nc.Config", kappa: float) -> None:
    cfg.USE_STYLE_EMOTION_ADVERSARY = False
    if kappa <= 0:
        cfg.USE_AFF_SPEAKER_ADVERSARY = False
        cfg.USE_ORTHOGONALITY = False
        cfg.LAMBDA_AFF_SPK_ADV = 0.0
        cfg.LAMBDA_ORTH = 0.0
    else:
        cfg.USE_AFF_SPEAKER_ADVERSARY = True
        cfg.USE_ORTHOGONALITY = True
        cfg.LAMBDA_AFF_SPK_ADV = BASE_LAMBDA_AFF_SPK_ADV * kappa
        cfg.LAMBDA_ORTH = BASE_LAMBDA_ORTH * kappa


@torch.no_grad()
def collect_representations(model: torch.nn.Module, loader, device: str) -> Dict[str, np.ndarray]:
    model.eval()
    acc: Dict[str, List[np.ndarray]] = {k: [] for k in ("h", "c_aff_rep", "c_style_rep", "emotion_logits", "y")}
    for batch in loader:
        out = model(batch["x"].to(device), grl_lambda=0.0)
        acc["y"].append(batch["y"].numpy())
        for key in ("h", "c_aff_rep", "c_style_rep", "emotion_logits"):
            acc[key].append(out[key].float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in acc.items()}


def train_fold(cfg, model, train_loader, val_loader, emotion_weights, tag: str = "") -> int:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS)
    scaler = nc.make_grad_scaler(cfg, torch.device(cfg.DEVICE))

    best_score, best_state, best_epoch, bad = -1e9, None, 0, 0
    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        grl_lambda = nc.grl_schedule(epoch - 1, cfg.NUM_EPOCHS, cfg.GRL_MAX_LAMBDA, cfg.ADV_WARMUP_EPOCHS)
        nc.train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer, device=cfg.DEVICE,
            cfg=cfg, emotion_weights=emotion_weights, grl_lambda=grl_lambda, scaler=scaler,
        )
        scheduler.step()
        val_metrics = nc.evaluate_model(model, val_loader, cfg.DEVICE, cfg=cfg)["metrics"]
        score = val_metrics["uar"] + 0.5 * val_metrics["macro_f1"]
        if tag and (epoch == 1 or epoch % 10 == 0):
            print(f"    hb [{tag}] epoch {epoch}/{cfg.NUM_EPOCHS} val_uar={val_metrics['uar']:.3f}", flush=True)
        if score > best_score:
            best_score, best_epoch, bad = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_epoch


def ensure_data_bundle(dataset: str, out_dir: str, feature_jobs: int, rebuild: bool) -> str:
    bundle_path = os.path.join(out_dir, f"data_bundle_{dataset}.pkl")
    if os.path.exists(bundle_path) and not rebuild:
        print(f"Reusing data bundle: {bundle_path}")
        return bundle_path
    cfg = fresh_config(dataset, OUT_DIR=out_dir, FEATURE_EXTRACTION_JOBS=feature_jobs)
    df = nc.discover_dataset(cfg)
    cache = nc.build_feature_cache(df, cfg)
    with open(bundle_path, "wb") as f:
        pickle.dump({"df": df, "cache": cache}, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_gb = os.path.getsize(bundle_path) / 1e9
    print(f"Wrote data bundle ({size_gb:.2f} GB): {bundle_path}")
    return bundle_path


def _init_worker(
    bundle_path: str,
    dataset: str,
    epochs: int,
    out_dir: str,
    loader_workers: int,
    num_threads: int,
) -> None:
    try:
        torch.set_num_threads(max(1, num_threads))
    except Exception:
        pass
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)

    cfg = fresh_config(dataset, OUT_DIR=out_dir, NUM_EPOCHS=epochs)
    device = nc.select_device(cfg.DEVICE)
    nc.configure_torch_runtime(device)
    cfg.DEVICE = str(device)
    cfg.NUM_WORKERS = loader_workers if device.type == "cuda" else nc.resolve_num_workers(cfg, device)

    _W["df"] = bundle["df"]
    _W["cache"] = bundle["cache"]
    _W["cfg"] = cfg
    _W["epochs"] = epochs
    _W["dataset"] = dataset
    _W["out_dir"] = out_dir
    _W["n_emotions"] = len(nc.emotion_names_for_dataset(cfg.DATASET))


def run_single(task: Dict) -> Dict:
    seed, fold_wanted, kappa = int(task["seed"]), int(task["fold"]), float(task["kappa"])
    base_cfg: "nc.Config" = _W["cfg"]
    df: pd.DataFrame = _W["df"]
    cache = _W["cache"]
    n_emotions: int = _W["n_emotions"]

    cfg = fresh_config(
        _W["dataset"], OUT_DIR=_W["out_dir"], NUM_EPOCHS=_W["epochs"],
        SEED=seed, DEVICE=base_cfg.DEVICE, NUM_WORKERS=base_cfg.NUM_WORKERS,
        USE_STYLE_BRANCH=not bool(task.get("no_style", False)),
    )
    apply_pressure(cfg, kappa)
    nc.set_seed(seed)

    groups = df["speaker"].astype(str).to_numpy(dtype=str)
    y_all = df["emotion"].to_numpy(dtype=np.int64)
    outer = GroupKFold(n_splits=cfg.N_SPLITS)
    trainval_idx, test_idx = list(outer.split(np.arange(len(df)), y_all, groups))[fold_wanted - 1]

    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.INNER_VAL_SIZE, random_state=seed + fold_wanted)
    tr_idx, va_idx = next(gss.split(
        np.arange(len(trainval_df)),
        trainval_df["emotion"].to_numpy(np.int64),
        trainval_df["speaker"].astype(str).to_numpy(dtype=str),
    ))
    train_df = trainval_df.iloc[tr_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[va_idx].reset_index(drop=True)

    if task.get("input_cmn"):
        cache = apply_input_cmn(cache, (train_df, val_df, test_df))

    train_loader, val_loader, test_loader, speaker_to_local, _ = nc.make_loaders_for_fold(
        train_df, val_df, test_df, cache, cfg
    )
    model = nc.DisentangledAffectiveStyleCBM(
        n_mels=cfg.N_MELS, h_dim=cfg.H_DIM, n_aff=cfg.N_AFF_CONCEPTS,
        n_style=cfg.N_STYLE_CONCEPTS, n_emotions=n_emotions,
        n_train_speakers=len(speaker_to_local), dropout=cfg.DROPOUT,
        emotion_head_input=cfg.EMOTION_HEAD_INPUT,
        use_aff_concept_branch=cfg.USE_AFF_CONCEPT_BRANCH,
        use_style_branch=cfg.USE_STYLE_BRANCH,
    ).to(cfg.DEVICE)
    emotion_weights = nc.class_weights_from_labels(train_df["emotion"].to_numpy(np.int64), n_emotions)

    t0 = time.time()
    tag = f"s{seed}f{fold_wanted}k{kappa:g}"
    best_epoch = train_fold(cfg, model, train_loader, val_loader, emotion_weights, tag=tag)
    reps = collect_representations(model, test_loader, cfg.DEVICE)
    task_metrics = nc.compute_metrics(reps["y"], reps["emotion_logits"])
    test_speakers = test_df["speaker"].astype(str).to_numpy(dtype=str)

    row = {
        "dataset": _W["dataset"], "seed": seed, "fold": fold_wanted, "kappa": kappa,
        "input_cmn": bool(task.get("input_cmn", False)),
        "no_style": bool(task.get("no_style", False)),
        "lambda_aff_spk_adv": cfg.LAMBDA_AFF_SPK_ADV, "lambda_orth": cfg.LAMBDA_ORTH,
        "best_epoch": best_epoch, "train_seconds": round(time.time() - t0, 1),
        "test_uar": task_metrics["uar"], "test_macro_f1": task_metrics["macro_f1"],
        "test_acc": task_metrics["acc"],
    }
    for rep_name in ("c_aff_rep", "c_style_rep", "h"):
        for probe in ("linear", "mlp"):
            mi = mi_lower_bound_bits(reps[rep_name], test_speakers, probe=probe, seed=seed)
            prefix = f"{rep_name}_{probe}"
            row[f"{prefix}_mi_lb_bits"] = mi["mi_lb_bits"]
            row[f"{prefix}_control_bits"] = mi["control_mi_lb_bits"]
            row[f"{prefix}_entropy_bits"] = mi["entropy_bits"]
    audit = nc.speaker_leakage_audit(reps["c_aff_rep"], test_speakers, seed=seed)
    row["c_aff_probe_acc"] = audit["probe_acc_mean"]
    row["c_aff_leakage_index"] = audit["probe_leakage_index"]

    emb_dir = os.path.join(str(_W["out_dir"]), "embeddings")
    prefix = ("cmn_" if task.get("input_cmn") else "") + ("nostyle_" if task.get("no_style") else "")
    np.savez_compressed(
        os.path.join(emb_dir, f"{prefix}seed{seed}_fold{fold_wanted}_kappa{kappa:g}.npz"),
        speakers=test_speakers, **reps,
    )
    return row


def _format_row(row: Dict) -> str:
    return (
        f"[seed={row['seed']} fold={row['fold']} kappa={row['kappa']:g}] "
        f"UAR={row['test_uar']:.4f}  "
        f"I_LB(c_aff;S)={row['c_aff_rep_linear_mi_lb_bits']:.3f}b "
        f"(H(S)={row['c_aff_rep_linear_entropy_bits']:.2f}b, "
        f"ctrl={row['c_aff_rep_linear_control_bits']:+.3f}b)  "
        f"epoch*={row['best_epoch']}  {row['train_seconds']}s"
    )


def run_sweep(args: argparse.Namespace) -> pd.DataFrame:
    out_dir = os.path.join(args.results_root, "floor_sweep", args.dataset)
    os.makedirs(os.path.join(out_dir, "embeddings"), exist_ok=True)
    base_name = "cmn_results.csv" if args.input_cmn else "floor_results.csv"
    if args.no_style:
        base_name = base_name.replace(".csv", "_nostyle.csv")
    results_csv = os.path.join(out_dir, base_name)

    bundle_path = ensure_data_bundle(args.dataset, out_dir, args.feature_jobs, args.rebuild_bundle)

    if args.input_cmn:
        args.pressures = [0.0]
    tasks = [
        {"seed": seed, "fold": fold, "kappa": kappa, "input_cmn": args.input_cmn,
         "no_style": args.no_style}
        for seed in args.seeds
        for fold in range(1, (args.max_folds or nc.Config.N_SPLITS) + 1)
        for kappa in args.pressures
    ]
    init_args = (bundle_path, args.dataset, args.epochs, out_dir, args.loader_workers, args.num_threads)
    print(f"{len(tasks)} runs, jobs={args.jobs}, loader_workers={args.loader_workers}/job")

    rows: List[Dict] = []
    t_start = time.time()
    if args.jobs <= 1:
        _init_worker(*init_args)
        for task in tasks:
            rows.append(run_single(task))
            print(f"({len(rows)}/{len(tasks)}) " + _format_row(rows[-1]))
            pd.DataFrame(rows).to_csv(results_csv, index=False)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.jobs, mp_context=ctx,
            initializer=_init_worker, initargs=init_args,
        ) as pool:
            futures = {pool.submit(run_single, task): task for task in tasks}
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    row = fut.result()
                except Exception:
                    print(f"FAILED {task}:\n{traceback.format_exc()}")
                    continue
                rows.append(row)
                print(f"({len(rows)}/{len(tasks)}) " + _format_row(row), flush=True)
                pd.DataFrame(rows).to_csv(results_csv, index=False)

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["seed", "fold", "kappa"]).reset_index(drop=True)
    out.to_csv(results_csv, index=False)
    if len(out) != len(tasks):
        raise RuntimeError(
            f"{len(tasks) - len(out)} of {len(tasks)} runs failed. "
            f"Partial results saved to {results_csv}"
        )
    print(f"\nSaved {len(out)}/{len(tasks)} rows -> {results_csv}  "
          f"(wall {time.time() - t_start:.0f}s)")
    return out


def parse_args() -> argparse.Namespace:
    cpu = os.cpu_count() or 8
    cuda = torch.cuda.is_available()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="cremad", choices=["cremad", "ravdess", "iemocap"])
    p.add_argument("--results-root", default=os.path.dirname(OUT_ROOT),
                   help="root containing experiment results and data caches (default: outputs)")
    p.add_argument("--pressures", default="0,0.5,1,2,4,8,16",
                   help="comma-separated kappa values scaling the anti-speaker terms")
    p.add_argument("--seeds", default="42", help="comma-separated seeds")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--max-folds", type=int, default=0, help="0 = all N_SPLITS folds")
    p.add_argument("--jobs", type=int, default=8 if cuda else 1,
                   help="concurrent trainings sharing the GPU (default: 8 on CUDA, else 1)")
    p.add_argument("--loader-workers", type=int, default=max(1, min(2, cpu // 8)),
                   help="DataLoader workers per training job")
    p.add_argument("--num-threads", type=int, default=2,
                   help="torch/BLAS threads per worker process")
    p.add_argument("--feature-jobs", type=int, default=-1,
                   help="processes for the one-off feature extraction (-1 = auto)")
    p.add_argument("--rebuild-bundle", action="store_true",
                   help="force rebuilding the on-disk data bundle")
    p.add_argument("--no-style-branch", dest="no_style", action="store_true",
                   help="drop the style branch to match the operator head")
    p.add_argument("--input-cmn", action="store_true",
                   help="CMN baseline: per-speaker log-mel channel-mean subtraction, kappa forced to 0")
    args = p.parse_args()
    args.pressures = [float(x) for x in args.pressures.split(",")]
    args.seeds = [int(x) for x in args.seeds.split(",")]
    return args


if __name__ == "__main__":
    run_sweep(parse_args())
