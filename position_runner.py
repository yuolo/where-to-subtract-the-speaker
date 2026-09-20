"""Subtraction at a fixed depth, with the enrollment grids"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import multiprocessing as mp
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from nc_bridge import nc, fresh_config
from mi_estimation import mi_lower_bound_bits
from position_gate import PositionGatedCBM, N_POSITIONS, fit_position_baselines
import enroll_grid as eg
import fixed_enroll as fe
import floor_sweep as fs

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "position")

_W: Dict[str, object] = {}
LAMBDA_CONCEPT = 1.50


def _init_worker(bundle_path, dataset, epochs, out_dir, loader_workers, num_threads):
    fs._init_worker(bundle_path, dataset, epochs, out_dir, loader_workers, num_threads)
    _W.update(fs._W)


@torch.no_grad()
def _split_baselines(model: PositionGatedCBM, loader, device: str,
                     spk_idx: np.ndarray, n_speakers: int) -> List[torch.Tensor]:
    model.eval()
    g = model.gate()
    idx_t = torch.as_tensor(spk_idx, device=device, dtype=torch.long)
    out: List[Optional[torch.Tensor]] = [None] * N_POSITIONS
    for l in sorted(model.active_positions()):
        sums, counts, off = None, torch.zeros(n_speakers, device=device), 0
        for batch in loader:
            x = batch["x"].to(device)
            n = x.shape[0]
            idx = idx_t[off:off + n]
            off += n
            b = [out[j][idx] if (j < l and out[j] is not None) else None
                 for j in range(N_POSITIONS)]
            t = model.encoder(x, baselines=b, gate=g, taps_for={l})["taps"][l].float()
            if sums is None:
                sums = torch.zeros((n_speakers,) + t.shape[1:], device=device)
            sums.index_add_(0, idx, t)
            counts.index_add_(0, idx, torch.ones_like(idx, dtype=counts.dtype))
        seen = counts > 0
        m = sums / counts.clamp(min=1).view((-1,) + (1,) * (sums.dim() - 1))
        if (~seen).any():
            m[~seen] = m[seen].mean(dim=0, keepdim=True)
        out[l] = m
    return out


@torch.no_grad()
def _evaluate(model: PositionGatedCBM, loader, speakers: np.ndarray,
              device: str) -> Dict[str, np.ndarray]:
    uniq, spk_idx = np.unique(speakers.astype(str), return_inverse=True)
    baselines = _split_baselines(model, loader, device, spk_idx, len(uniq))
    idx_t = torch.as_tensor(spk_idx, device=device, dtype=torch.long)
    hs, hr, cs, ls, ys, off = [], [], [], [], [], 0
    model.eval()
    for batch in loader:
        x = batch["x"].to(device)
        n = x.shape[0]
        idx = idx_t[off:off + n]
        off += n
        out = model(x, baselines=[None if b is None else b[idx] for b in baselines])
        hs.append(out["h"].cpu().numpy())
        hr.append(out["h_raw"].cpu().numpy())
        cs.append(out["concepts"].cpu().numpy())
        ls.append(out["emotion_logits"].cpu().numpy())
        ys.append(batch["y"].numpy())
    return {"h": np.concatenate(hs), "h_raw": np.concatenate(hr),
            "concepts": np.concatenate(cs),
            "emotion_logits": np.concatenate(ls), "y": np.concatenate(ys)}


@torch.no_grad()
def _train_encoder_states(model: PositionGatedCBM, loader, device: str):
    model.eval()
    hs, sp = [], []
    for batch in loader:
        out = model.encoder(batch["x"].to(device), baselines=None,
                            gate=model.gate(), taps_for={N_POSITIONS - 1})
        hs.append(out["taps"][N_POSITIONS - 1].cpu().numpy())
        sp.append(batch["speaker_local"].numpy())
    return np.concatenate(hs), np.concatenate(sp)


def train_position(cfg, model: PositionGatedCBM, train_loader, val_loader,
                   val_speakers: np.ndarray, emotion_weights, n_speakers: int,
                   tag: str) -> int:
    device = cfg.DEVICE
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR,
                                  weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS)
    emotion_weights = emotion_weights.to(device)

    best_score, best_state, best_epoch, bad = -1e9, None, 0, 0
    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        baselines = fit_position_baselines(model, train_loader, device, n_speakers)
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            spk = batch["speaker_local"].to(device)
            aff_targets = batch["aff_targets"].to(device)

            out = model(x, baselines=[None if b is None else b[spk].detach()
                                      for b in baselines])
            loss = F.cross_entropy(out["emotion_logits"], y, weight=emotion_weights)
            loss = loss + LAMBDA_CONCEPT * F.smooth_l1_loss(out["concepts"], aff_targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        scheduler.step()

        val_out = _evaluate(model, val_loader, val_speakers, device)
        val_metrics = nc.compute_metrics(val_out["y"], val_out["emotion_logits"])
        score = val_metrics["uar"] + 0.5 * val_metrics["macro_f1"]
        if tag and (epoch == 1 or epoch % 10 == 0):
            g = model.gate().detach().cpu().numpy()
            print(f"    hb [{tag}] epoch {epoch}/{cfg.NUM_EPOCHS} "
                  f"val_uar={val_metrics['uar']:.3f} gate={np.round(g, 3).tolist()}", flush=True)
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
    pos: Optional[int] = task.get("fixed_position")
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
    model = PositionGatedCBM(
        encoder, h_dim=cfg.H_DIM, n_concepts=cfg.N_AFF_CONCEPTS,
        n_emotions=n_emotions, dropout=cfg.DROPOUT,
        fixed_position=pos, tau=float(task.get("tau", 1.0)),
    ).to(cfg.DEVICE)
    emotion_weights = nc.class_weights_from_labels(train_df["emotion"].to_numpy(np.int64), n_emotions)

    t0 = time.time()
    tag = f"{'p' + str(pos) if pos is not None else 'gate'}-s{seed}f{fold_wanted}"
    best_epoch = train_position(cfg, model, train_loader, val_loader, val_speakers,
                                emotion_weights, len(speaker_to_local), tag)
    reps = _evaluate(model, test_loader, test_speakers, cfg.DEVICE)
    task_metrics = nc.compute_metrics(reps["y"], reps["emotion_logits"])
    gate = model.gate().detach().cpu().numpy()

    row = {
        "dataset": _W["dataset"],
        "model": f"fixed_position_{pos}" if pos is not None else "learned_gate",
        "fixed_position": -1 if pos is None else pos,
        "seed": seed, "fold": fold_wanted,
        "best_epoch": best_epoch, "train_seconds": round(time.time() - t0, 1),
        "test_uar": task_metrics["uar"], "test_macro_f1": task_metrics["macro_f1"],
        "test_acc": task_metrics["acc"],
    }
    for l in range(N_POSITIONS):
        row[f"gate{l}"] = float(gate[l])
    row["gate_argmax"] = int(np.argmax(gate))
    row["gate_entropy_bits"] = float(-(gate * np.log(np.clip(gate, 1e-12, 1))).sum() / np.log(2))

    for rep_name in ("concepts", "h", "h_raw"):
        for probe in ("linear", "mlp"):
            mi = mi_lower_bound_bits(reps[rep_name], test_speakers, probe=probe, seed=seed)
            row[f"{rep_name}_{probe}_mi_lb_bits"] = mi["mi_lb_bits"]
            row[f"{rep_name}_{probe}_control_bits"] = mi["control_mi_lb_bits"]
            row[f"{rep_name}_{probe}_entropy_bits"] = mi["entropy_bits"]

    if task.get("enroll_grid"):
        h_tr, sp_tr = _train_encoder_states(model, train_loader, cfg.DEVICE)
        tau2, within = eg.variance_decomposition(h_tr, sp_tr)
        B_tr, W_tr = eg.speaker_covariances(h_tr, sp_tr)
        ks = [1, 2, 5, 10, 20]
        recs = []
        for g in eg.grid(model, reps["h_raw"], test_speakers, reps["y"], cfg.DEVICE,
                         tau2, within, seed, ks=ks,
                         lams=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
                         B=B_tr, W=W_tr):
            met = nc.compute_metrics(reps["y"], g["logits"])
            mi = mi_lower_bound_bits(g["concepts"], test_speakers, probe="linear", seed=seed)
            recs.append({"dataset": _W["dataset"], "seed": seed, "fold": fold_wanted,
                         "k": g["k"], "lam": g["lam"], "lam_star": g["lam_star"],
                         "is_lam_star": g["is_lam_star"], "is_matrix": g["is_matrix"],
                         "tau2": round(tau2, 4), "within": round(within, 4),
                         "test_uar": met["uar"], "test_acc": met["acc"],
                         "leak_bits": mi["mi_lb_bits"],
                         "control_bits": mi["control_mi_lb_bits"]})
        gdir = os.path.join(str(_W["out_dir"]), "enroll_grid")
        os.makedirs(gdir, exist_ok=True)
        pd.DataFrame(recs).to_csv(
            os.path.join(gdir, f"grid_seed{seed}_fold{fold_wanted}.csv"), index=False)

        prior = eg.population_prior(h_tr, sp_tr)
        crecs = []
        clams = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0] if task.get("centered_grid") else None
        for g in eg.grid_centered(model, reps["h_raw"], test_speakers, cfg.DEVICE,
                                  tau2, within, seed, ks=ks, prior=prior, lams=clams):
            met = nc.compute_metrics(reps["y"], g["logits"])
            mi = mi_lower_bound_bits(g["concepts"], test_speakers, probe="linear", seed=seed)
            crecs.append({"dataset": _W["dataset"], "seed": seed, "fold": fold_wanted,
                          "k": g["k"], "lam": g["lam"], "lam_star": g["lam_star"],
                          "is_lam_star": g["is_lam_star"], "centered": 1,
                          "tau2": round(tau2, 4), "within": round(within, 4),
                          "test_uar": met["uar"], "test_acc": met["acc"],
                          "leak_bits": mi["mi_lb_bits"],
                          "control_bits": mi["control_mi_lb_bits"]})
        pd.DataFrame(crecs).to_csv(
            os.path.join(gdir, f"grid_centered_seed{seed}_fold{fold_wanted}.csv"), index=False)

    if task.get("fixed_enroll") and pos is not None:
        kw = {}
        if pos == N_POSITIONS - 1:
            h_tr, sp_tr = _train_encoder_states(model, train_loader, cfg.DEVICE)
            tau2, within = eg.variance_decomposition(h_tr, sp_tr)
            kw = {"h_raw": reps["h_raw"], "tau2": tau2, "within": within,
                  "prior": eg.population_prior(h_tr, sp_tr)}
        frecs = fe.evaluate(model, test_loader, cfg.DEVICE, pos, test_speakers, reps["y"],
                            seed, fold_wanted, nc.compute_metrics, **kw)
        fdir = os.path.join(str(_W["out_dir"]), "fixed_enroll")
        os.makedirs(fdir, exist_ok=True)
        pd.DataFrame([dict(r, dataset=_W["dataset"]) for r in frecs]).to_csv(
            os.path.join(fdir, f"fixed_pos{pos}_seed{seed}_fold{fold_wanted}.csv"), index=False)

    emb_dir = os.path.join(str(_W["out_dir"]), "embeddings")
    os.makedirs(emb_dir, exist_ok=True)
    stem = f"pos{pos}" if pos is not None else "gate"
    np.savez_compressed(os.path.join(emb_dir, f"{stem}_seed{seed}_fold{fold_wanted}.npz"),
                        speakers=test_speakers, gate=gate, **reps)
    return row


def _format_row(row: Dict) -> str:
    g = [round(row[f"gate{l}"], 3) for l in range(N_POSITIONS)]
    return (
        f"[{row['model']} seed={row['seed']} fold={row['fold']}] UAR={row['test_uar']:.4f}  "
        f"I_LB(c;S)={row['concepts_linear_mi_lb_bits']:.3f}b "
        f"(ctrl={row['concepts_linear_control_bits']:+.3f}b)  "
        f"gate={g}  epoch*={row['best_epoch']}  {row['train_seconds']}s"
    )


def main() -> None:
    cpu = os.cpu_count() or 8
    cuda = torch.cuda.is_available()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="cremad", choices=["cremad", "ravdess", "iemocap"])
    p.add_argument("--fixed-position", type=int, default=None, choices=list(range(N_POSITIONS)),
                   help="pin the subtraction to one tap; omit to learn the gate")
    p.add_argument("--tau", type=float, default=1.0, help="gate softmax temperature")
    p.add_argument("--enroll-grid", action="store_true",
                   help="after training, evaluate k-shot enrollment over the shrinkage grid and the closed form")
    p.add_argument("--centered-grid", action="store_true",
                   help="with --enroll-grid, also evaluate the grid in the centered form")
    p.add_argument("--fixed-enroll", action="store_true",
                   help="after training, evaluate fixed enrollment on held-out utterances")
    p.add_argument("--seeds", default="42", help="comma-separated seeds")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--max-folds", type=int, default=0, help="0 = all N_SPLITS folds")
    p.add_argument("--jobs", type=int, default=8 if cuda else 1)
    p.add_argument("--loader-workers", type=int, default=0)
    p.add_argument("--num-threads", type=int, default=max(2, min(4, cpu // 8)))
    p.add_argument("--feature-jobs", type=int, default=-1)
    p.add_argument("--out-root", default=OUT_ROOT,
                   help="results root")
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    out_dir = os.path.join(args.out_root, args.dataset)
    os.makedirs(os.path.join(out_dir, "embeddings"), exist_ok=True)
    stem = f"pos{args.fixed_position}" if args.fixed_position is not None else "gate"
    results_csv = os.path.join(out_dir, f"{stem}_results.csv")

    fs_dir = os.path.join(fs.OUT_ROOT, args.dataset)
    bundle = os.path.join(fs_dir, f"data_bundle_{args.dataset}.pkl")
    if not os.path.exists(bundle):
        os.makedirs(fs_dir, exist_ok=True)
        bundle = fs.ensure_data_bundle(args.dataset, fs_dir, args.feature_jobs, rebuild=False)

    tasks = [{"seed": s, "fold": f, "fixed_position": args.fixed_position,
              "tau": args.tau, "enroll_grid": args.enroll_grid,
              "centered_grid": args.centered_grid, "fixed_enroll": args.fixed_enroll}
             for s in seeds for f in range(1, (args.max_folds or nc.Config.N_SPLITS) + 1)]
    init_args = (bundle, args.dataset, args.epochs, out_dir, args.loader_workers, args.num_threads)
    print(f"{len(tasks)} runs ({stem}), jobs={args.jobs}")

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
        print(f"mean I_LB(c;S) linear = {out['concepts_linear_mi_lb_bits'].mean():.3f} "
              f"+/- {out['concepts_linear_mi_lb_bits'].std():.3f} bits")
        print("mean gate = " + str([round(out[f'gate{l}'].mean(), 3) for l in range(N_POSITIONS)]))
    print(f"Saved {len(out)}/{len(tasks)} rows -> {results_csv}")


if __name__ == "__main__":
    main()
