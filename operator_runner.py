"""Shift-operator bottleneck on the CRNN substrate"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from nc_bridge import nc, fresh_config
from mi_estimation import mi_lower_bound_bits
from operator_model import ShiftOperatorCBM
from ssl_sweep import _swap_metrics
import floor_sweep as fs

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "operator")

_W: Dict[str, object] = {}
LAMBDA_CONCEPT = 1.50


def _init_worker(bundle_path, dataset, epochs, out_dir, loader_workers, num_threads):
    fs._init_worker(bundle_path, dataset, epochs, out_dir, loader_workers, num_threads)
    _W.update(fs._W)


@torch.no_grad()
def _collect_h(model: ShiftOperatorCBM, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    hs, ys = [], []
    for batch in loader:
        hs.append(model.encoder(batch["x"].to(device)).float().cpu().numpy())
        ys.append(batch["y"].numpy())
    return np.concatenate(hs), np.concatenate(ys)


AFFINE_EPS = 0.10


def _speaker_means(h: np.ndarray, speakers: np.ndarray) -> np.ndarray:
    out = np.empty_like(h)
    for spk in np.unique(speakers):
        m = speakers == spk
        out[m] = h[m].mean(axis=0)
    return out


def _speaker_scales(h: np.ndarray, speakers: np.ndarray) -> np.ndarray:
    out = np.ones_like(h)
    for spk in np.unique(speakers):
        m = speakers == spk
        out[m] = 1.0 / (h[m].std(axis=0) + AFFINE_EPS)
    return out


@torch.no_grad()
def _enroll_k_outputs(model, h: np.ndarray, speakers: np.ndarray, device: str,
                      k: int, seed: int):
    rng = np.random.default_rng(seed)
    baseline = np.empty_like(h)
    for spk in np.unique(speakers):
        idxs = np.where(speakers == spk)[0]
        for i in idxs:
            others = idxs[idxs != i]
            pick = others if len(others) <= k else rng.choice(others, size=k, replace=False)
            baseline[i] = h[pick].mean(axis=0)
    delta = torch.tensor(h - baseline, dtype=torch.float32, device=device)
    concepts = model.concept_head(delta)
    return concepts.cpu().numpy(), model.emotion_head(concepts).cpu().numpy()


@torch.no_grad()
def _train_baselines(model: ShiftOperatorCBM, loader, device: str, n_speakers: int, h_dim: int,
                     affine: bool = False):
    model.eval()
    sums = torch.zeros(n_speakers, h_dim, device=device)
    sq = torch.zeros(n_speakers, h_dim, device=device)
    counts = torch.zeros(n_speakers, device=device)
    for batch in loader:
        h = model.encoder(batch["x"].to(device)).float()
        spk = batch["speaker_local"].to(device)
        sums.index_add_(0, spk, h)
        sq.index_add_(0, spk, h ** 2)
        counts.index_add_(0, spk, torch.ones_like(spk, dtype=sums.dtype))
    seen = counts > 0
    denom = counts.clamp(min=1).unsqueeze(1)
    means = sums / denom
    if (~seen).any():
        means[~seen] = means[seen].mean(dim=0, keepdim=True)
    scales = None
    if affine:
        var = (sq / denom - means ** 2).clamp(min=0)
        scales = 1.0 / (var.sqrt() + AFFINE_EPS)
        if (~seen).any():
            scales[~seen] = 1.0
    return means, scales


@torch.no_grad()
def _evaluate(model: ShiftOperatorCBM, loader, speakers: np.ndarray, device: str,
              affine: bool = False) -> Dict[str, np.ndarray]:
    h, y = _collect_h(model, loader, device)
    delta = h - _speaker_means(h, speakers)
    if affine:
        delta = delta * _speaker_scales(h, speakers)
    delta_t = torch.tensor(delta, dtype=torch.float32, device=device)
    concepts = model.concept_head(delta_t)
    logits = model.emotion_head(concepts)
    return {
        "h": h, "delta": delta_t.cpu().numpy(), "y": y,
        "concepts": concepts.cpu().numpy(), "emotion_logits": logits.cpu().numpy(),
    }


def train_operator(cfg, model: ShiftOperatorCBM, train_loader, val_loader,
                   val_speakers: np.ndarray, emotion_weights, n_speakers: int, tag: str,
                   affine: bool = False) -> int:
    device = cfg.DEVICE
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS)
    emotion_weights = emotion_weights.to(device)

    best_score, best_state, best_epoch, bad = -1e9, None, 0, 0
    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        baselines, scales = _train_baselines(model, train_loader, device, n_speakers, cfg.H_DIM, affine=affine)
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            spk = batch["speaker_local"].to(device)
            aff_targets = batch["aff_targets"].to(device)

            bscale = scales[spk].detach() if scales is not None else None
            out = model(x, baseline=baselines[spk].detach(), scale=bscale)
            loss = F.cross_entropy(out["emotion_logits"], y, weight=emotion_weights)
            loss = loss + LAMBDA_CONCEPT * F.smooth_l1_loss(out["concepts"], aff_targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        scheduler.step()

        val_out = _evaluate(model, val_loader, val_speakers, device, affine=affine)
        val_metrics = nc.compute_metrics(val_out["y"], val_out["emotion_logits"])
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


def run_single(task: Dict) -> Dict:
    seed, fold_wanted = int(task["seed"]), int(task["fold"])
    affine = bool(task.get("affine", False))
    base_cfg = _W["cfg"]
    df: pd.DataFrame = _W["df"]
    cache = _W["cache"]
    n_emotions: int = _W["n_emotions"]

    cfg = fresh_config(_W["dataset"], OUT_DIR=_W["out_dir"], NUM_EPOCHS=_W["epochs"],
                       SEED=seed, DEVICE=base_cfg.DEVICE, NUM_WORKERS=base_cfg.NUM_WORKERS)
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

    train_loader, val_loader, test_loader, speaker_to_local, _ = nc.make_loaders_for_fold(
        train_df, val_df, test_df, cache, cfg
    )
    val_speakers = val_df["speaker"].astype(str).to_numpy(dtype=str)
    test_speakers = test_df["speaker"].astype(str).to_numpy(dtype=str)

    encoder = nc.CRNNEncoder(n_mels=cfg.N_MELS, h_dim=cfg.H_DIM, dropout=cfg.DROPOUT)
    model = ShiftOperatorCBM(
        encoder, h_dim=cfg.H_DIM, n_concepts=cfg.N_AFF_CONCEPTS,
        n_emotions=n_emotions, dropout=cfg.DROPOUT,
    ).to(cfg.DEVICE)
    emotion_weights = nc.class_weights_from_labels(train_df["emotion"].to_numpy(np.int64), n_emotions)

    t0 = time.time()
    tag = f"{'aff' if affine else 'op'}-s{seed}f{fold_wanted}"
    best_epoch = train_operator(cfg, model, train_loader, val_loader, val_speakers,
                                emotion_weights, len(speaker_to_local), tag, affine=affine)
    reps = _evaluate(model, test_loader, test_speakers, cfg.DEVICE, affine=affine)
    task_metrics = nc.compute_metrics(reps["y"], reps["emotion_logits"])

    row = {
        "dataset": _W["dataset"], "model": "affine_operator_v1" if affine else "shift_operator_v0",
        "seed": seed, "fold": fold_wanted,
        "best_epoch": best_epoch, "train_seconds": round(time.time() - t0, 1),
        "test_uar": task_metrics["uar"], "test_macro_f1": task_metrics["macro_f1"],
        "test_acc": task_metrics["acc"],
    }
    for rep_name in ("concepts", "delta", "h"):
        for probe in ("linear", "mlp"):
            mi = mi_lower_bound_bits(reps[rep_name], test_speakers, probe=probe, seed=seed)
            row[f"{rep_name}_{probe}_mi_lb_bits"] = mi["mi_lb_bits"]
            row[f"{rep_name}_{probe}_control_bits"] = mi["control_mi_lb_bits"]
            row[f"{rep_name}_{probe}_entropy_bits"] = mi["entropy_bits"]
    audit = nc.speaker_leakage_audit(reps["concepts"], test_speakers, seed=seed)
    row["concepts_probe_acc"] = audit["probe_acc_mean"]
    row["concepts_leakage_index"] = audit["probe_leakage_index"]

    row.update(_swap_metrics(model, reps, test_speakers, cfg.DEVICE, seed=seed, affine=affine))
    if not affine:
        for k in (1, 2, 5, 10):
            k_concepts, k_logits = _enroll_k_outputs(model, reps["h"], test_speakers,
                                                     cfg.DEVICE, k=k, seed=seed)
            km = nc.compute_metrics(reps["y"], k_logits)
            kmi = mi_lower_bound_bits(k_concepts, test_speakers, probe="linear", seed=seed)
            row[f"enrollk{k}_test_uar"] = km["uar"]
            row[f"enrollk{k}_concepts_mi_lb_bits"] = kmi["mi_lb_bits"]

    emb_dir = os.path.join(str(_W["out_dir"]), "embeddings")
    os.makedirs(emb_dir, exist_ok=True)
    prefix = "affine_" if affine else ""
    np.savez_compressed(os.path.join(emb_dir, f"{prefix}seed{seed}_fold{fold_wanted}.npz"),
                        speakers=test_speakers, **reps)
    return row


def _format_row(row: Dict) -> str:
    return (
        f"[op seed={row['seed']} fold={row['fold']}] UAR={row['test_uar']:.4f}  "
        f"I_LB(concepts;S)={row['concepts_linear_mi_lb_bits']:.3f}b "
        f"(mlp {row['concepts_mlp_mi_lb_bits']:.3f}b, "
        f"delta {row['delta_linear_mi_lb_bits']:.3f}b, "
        f"ctrl={row['concepts_linear_control_bits']:+.3f}b)  "
        f"epoch*={row['best_epoch']}  {row['train_seconds']}s"
    )


def main() -> None:
    cpu = os.cpu_count() or 8
    cuda = torch.cuda.is_available()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="cremad", choices=["cremad", "ravdess", "iemocap"])
    p.add_argument("--affine", action="store_true",
                   help="affine variant, delta = (h - b_s) / (sigma_s + eps)")
    p.add_argument("--seeds", default="42", help="comma-separated seeds")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--max-folds", type=int, default=0, help="0 = all N_SPLITS folds")
    p.add_argument("--jobs", type=int, default=8 if cuda else 1)
    p.add_argument("--loader-workers", type=int, default=0)
    p.add_argument("--num-threads", type=int, default=max(2, min(4, cpu // 8)))
    p.add_argument("--feature-jobs", type=int, default=-1)
    p.add_argument("--results-root", default=os.path.dirname(OUT_ROOT),
                   help="root containing experiment results and data caches (default: outputs)")
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    out_dir = os.path.join(args.results_root, "operator", args.dataset)
    os.makedirs(os.path.join(out_dir, "embeddings"), exist_ok=True)
    results_csv = os.path.join(out_dir, "affine_results.csv" if args.affine else "operator_results.csv")

    fs_dir = os.path.join(args.results_root, "floor_sweep", args.dataset)
    bundle = os.path.join(fs_dir, f"data_bundle_{args.dataset}.pkl")
    if not os.path.exists(bundle):
        os.makedirs(fs_dir, exist_ok=True)
        bundle = fs.ensure_data_bundle(args.dataset, fs_dir, args.feature_jobs, rebuild=False)

    tasks = [{"seed": s, "fold": f, "affine": args.affine} for s in seeds
             for f in range(1, (args.max_folds or nc.Config.N_SPLITS) + 1)]
    init_args = (bundle, args.dataset, args.epochs, out_dir, args.loader_workers, args.num_threads)
    print(f"{len(tasks)} operator runs, jobs={args.jobs}")

    rows: List[Dict] = []
    if args.jobs <= 1:
        _init_worker(*init_args)
        for task in tasks:
            rows.append(run_single(task))
            print(f"({len(rows)}/{len(tasks)}) " + _format_row(rows[-1]), flush=True)
            pd.DataFrame(rows).to_csv(results_csv, index=False)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx,
                                 initializer=_init_worker, initargs=init_args) as pool:
            futures = {pool.submit(run_single, t): t for t in tasks}
            for fut in as_completed(futures):
                try:
                    row = fut.result()
                except Exception:
                    print(f"FAILED {futures[fut]}:\n{traceback.format_exc()}", flush=True)
                    continue
                rows.append(row)
                print(f"({len(rows)}/{len(tasks)}) " + _format_row(row), flush=True)
                pd.DataFrame(rows).to_csv(results_csv, index=False)

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["seed", "fold"]).reset_index(drop=True)
    out.to_csv(results_csv, index=False)
    if len(out) != len(tasks):
        raise RuntimeError(
            f"{len(tasks) - len(out)} of {len(tasks)} runs failed. "
            f"Partial results saved to {results_csv}"
        )
    if len(out):
        print(f"\nmean UAR = {out['test_uar'].mean():.4f} +/- {out['test_uar'].std():.4f}")
        print(f"mean I_LB(concepts;S) linear = {out['concepts_linear_mi_lb_bits'].mean():.3f} "
              f"+/- {out['concepts_linear_mi_lb_bits'].std():.3f} bits")
    print(f"Saved {len(out)}/{len(tasks)} rows -> {results_csv}")


if __name__ == "__main__":
    main()
