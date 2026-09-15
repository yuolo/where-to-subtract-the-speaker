"""Post-hoc subtraction, LEACE and INLP on a trained wav2vec 2.0 encoder"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from nc_bridge import nc
from mi_estimation import mi_lower_bound_bits
import ssl_sweep
from ssl_sweep import (BATCH, H_DIM, LR, PATIENCE, WEIGHT_DECAY, load_data,
                       make_encoder, _head, _speaker_means_np)

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "outputs", "posthoc")

_W: Dict[str, object] = {}


class PlainW2V2(nn.Module):
    def __init__(self, in_dim: int, n_emotions: int):
        super().__init__()
        self.encoder = make_encoder(in_dim)
        self.emotion_head = _head(H_DIM, 64, n_emotions)

    def forward(self, x):
        h = self.encoder(x)
        return {"h": h, "emotion_logits": self.emotion_head(h)}


def leace_eraser(X: np.ndarray, labels: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Xs = (X - mu) / sd
    n = len(X)

    uniq, inv = np.unique(labels, return_inverse=True)
    Z = np.zeros((n, len(uniq)))
    Z[np.arange(n), inv] = 1.0
    Zc = Z - Z.mean(axis=0)

    sigma = Xs.T @ Xs / n
    evals, evecs = np.linalg.eigh(sigma)
    keep = evals > max(evals.max(), 1e-12) * 1e-12
    w_isqrt = (evecs[:, keep] / np.sqrt(evals[keep])) @ evecs[:, keep].T
    w_sqrt = (evecs[:, keep] * np.sqrt(evals[keep])) @ evecs[:, keep].T

    m = w_isqrt @ (Xs.T @ Zc / n)
    u, s, _ = np.linalg.svd(m, full_matrices=False)
    rank = int((s > s.max() * 1e-10).sum()) if s.size else 0
    u = u[:, :rank]
    proj = u @ u.T

    erase = w_sqrt @ proj @ w_isqrt

    def apply(Y: np.ndarray) -> np.ndarray:
        Ys = (np.asarray(Y, dtype=np.float64) - mu) / sd
        return (mu + (Ys - Ys @ erase.T) * sd).astype(np.float32)

    return apply


def inlp_eraser(X: np.ndarray, labels: np.ndarray, seed: int = 0,
                max_iter: int = 40) -> Callable[[np.ndarray], np.ndarray]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Xs = (X - mu) / sd
    _, y = np.unique(labels, return_inverse=True)
    chance = 1.0 / len(np.unique(y))
    rng = np.random.default_rng(seed)

    d = X.shape[1]
    P_total = np.eye(d)
    cur = Xs
    for _ in range(max_iter):
        acc = cross_val_score(
            LogisticRegression(max_iter=1000, random_state=seed),
            cur, y, cv=3).mean()
        if acc <= chance + 0.02:
            break
        clf = LogisticRegression(max_iter=1000, random_state=int(rng.integers(1e6)))
        clf.fit(cur, y)
        W = clf.coef_
        q, _ = np.linalg.qr(W.T)
        P_step = np.eye(d) - q @ q.T
        P_total = P_step @ P_total
        cur = Xs @ P_total.T

    def apply(Y: np.ndarray) -> np.ndarray:
        Ys = (np.asarray(Y, dtype=np.float64) - mu) / sd
        return (mu + (Ys @ P_total.T) * sd).astype(np.float32)

    return apply


def train_plain(X, y, speakers, tr, va, n_emotions, seed, epochs, device):
    nc.set_seed(seed)
    model = PlainW2V2(X.shape[1], n_emotions).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gen = torch.Generator().manual_seed(seed)
    Xtr = torch.tensor(X[tr], device=device)
    ytr = torch.tensor(y[tr], device=device)
    weights = nc.class_weights_from_labels(y[tr], n_emotions).to(device)

    best_score, best_state, best_epoch, bad = -1e9, None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for bidx in ssl_sweep._batches(len(tr), gen):
            b = bidx.to(device)
            out = model(Xtr[b])
            loss = F.cross_entropy(out["emotion_logits"], ytr[b], weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        with torch.no_grad():
            model.eval()
            vm = nc.compute_metrics(
                y[va], model(torch.tensor(X[va], device=device))["emotion_logits"].cpu().numpy())
        score = vm["uar"] + 0.5 * vm["macro_f1"]
        if score > best_score:
            best_score, best_epoch, bad = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def run_one(task: Dict) -> List[Dict]:
    dataset = str(task["dataset"])
    seed, fold = int(task["seed"]), int(task["fold"])
    device = str(_W["device"])
    data = _W["data"][dataset]
    X, y, speakers = data["X"], data["y"], data["speakers"]

    outer = GroupKFold(n_splits=5)
    trainval_idx, test_idx = list(outer.split(np.arange(len(y)), y, speakers))[fold - 1]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed + fold)
    tr_rel, va_rel = next(gss.split(trainval_idx, y[trainval_idx], speakers[trainval_idx]))
    tr, va, te = trainval_idx[tr_rel], trainval_idx[va_rel], test_idx

    t0 = time.time()
    model, best_epoch = train_plain(X, y, speakers, tr, va,
                                    int(data["n_emotions"]), seed,
                                    int(_W["epochs"]), device)
    with torch.no_grad():
        h_tr = model.encoder(torch.tensor(X[tr], device=device)).cpu().numpy()
        h_te = model.encoder(torch.tensor(X[te], device=device)).cpu().numpy()

    variants: Dict[str, np.ndarray] = {"plain": h_te}
    variants["subtract"] = h_te - _speaker_means_np(h_te, speakers[te])
    variants["leace_trans"] = leace_eraser(h_te, speakers[te])(h_te)
    variants["leace_trainfit"] = leace_eraser(h_tr, speakers[tr])(h_te)
    variants["inlp_trans"] = inlp_eraser(h_te, speakers[te], seed=seed)(h_te)

    rows = []
    with torch.no_grad():
        for name, feats in variants.items():
            logits = model.emotion_head(
                torch.tensor(feats, dtype=torch.float32, device=device)).cpu().numpy()
            m = nc.compute_metrics(y[te], logits)
            mi = mi_lower_bound_bits(feats, speakers[te], probe="linear", seed=seed)
            rows.append({
                "dataset": dataset, "variant": name, "seed": seed, "fold": fold,
                "best_epoch": best_epoch,
                "train_seconds": round(time.time() - t0, 1),
                "test_uar": m["uar"], "test_macro_f1": m["macro_f1"],
                "h_linear_mi_lb_bits": mi["mi_lb_bits"],
                "h_linear_control_bits": mi["control_mi_lb_bits"],
                "h_linear_entropy_bits": mi["entropy_bits"],
            })
    return rows


def _init_worker(datasets: List[str], epochs: int, device: str):
    torch.set_num_threads(2)
    data = {}
    for ds in datasets:
        csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "outputs", "floor_sweep", ds,
                           "concept_feature_cache_opensmile_eGeMAPSv02.csv")
        data[ds] = load_data(ds, csv)
    _W.update(epochs=epochs, device=device, data=data)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", default="cremad,iemocap,ravdess")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        datasets, args.epochs = ["ravdess"], 3
        tasks = [{"dataset": "ravdess", "seed": 42, "fold": 1}]
    else:
        datasets = args.datasets.split(",")
        tasks = [{"dataset": d, "seed": int(s), "fold": f}
                 for d in datasets for s in args.seeds.split(",")
                 for f in range(1, 6)]

    os.makedirs(OUT_ROOT, exist_ok=True)
    results_csv = os.path.join(
        OUT_ROOT, "posthoc_results_smoke.csv" if args.smoke else "posthoc_results.csv")
    print(f"{len(tasks)} trainings x 4 variants, jobs={args.jobs}")

    rows: List[Dict] = []

    def _consume(new_rows: List[Dict]) -> None:
        rows.extend(new_rows)
        r = {x["variant"]: x for x in new_rows}
        print(f"({len(rows)//4}/{len(tasks)}) "
              f"[{new_rows[0]['dataset']} seed={new_rows[0]['seed']} fold={new_rows[0]['fold']}] "
              + "  ".join(f"{v}: {r[v]['test_uar']:.3f}/{r[v]['h_linear_mi_lb_bits']:.3f}b"
                          for v in ("plain", "subtract", "leace_trans",
                                    "leace_trainfit", "inlp_trans")),
              flush=True)
        pd.DataFrame(rows).to_csv(results_csv, index=False)

    if args.jobs <= 1:
        _init_worker(datasets, args.epochs, args.device)
        for t in tasks:
            _consume(run_one(t))
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx,
                                 initializer=_init_worker,
                                 initargs=(datasets, args.epochs, args.device)) as pool:
            futures = {pool.submit(run_one, t): t for t in tasks}
            for fut in as_completed(futures):
                try:
                    _consume(fut.result())
                except Exception:
                    print(f"FAILED {futures[fut]}:\n{traceback.format_exc()}", flush=True)

    out = pd.DataFrame(rows).sort_values(["dataset", "variant", "seed", "fold"])
    out.to_csv(results_csv, index=False)
    print(f"\nSaved {len(out)} rows -> {results_csv}\n")
    if len(out):
        agg = out.groupby(["dataset", "variant"]).agg(
            uar=("test_uar", "mean"), mi=("h_linear_mi_lb_bits", "mean")).round(3)
        print(agg.to_string())


if __name__ == "__main__":
    main()
