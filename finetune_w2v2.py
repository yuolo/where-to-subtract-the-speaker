"""Fine-tuned wav2vec 2.0 with plain, factor, input-CMN and shift-operator heads"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from nc_bridge import nc
from mi_estimation import mi_lower_bound_bits
import ssl_sweep
from ssl_sweep import (H_DIM, PATIENCE as _UNUSED_PATIENCE,
                       W2V2DualCBM, W2V2OperatorCBM, _head, build_targets)

MODEL_NAME = "facebook/wav2vec2-base"
SR, MAX_SECONDS = 16000, 3.5
BATCH_DEFAULT, EPOCHS_DEFAULT, FT_PATIENCE = 16, 12, 4
ENCODER_LR, HEAD_LR, WEIGHT_DECAY = 1e-5, 1e-3, 1e-4

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "outputs", "finetune")

_W: Dict[str, object] = {}


def load_ft_data(dataset: str, concept_csv: str, limit: int = 0,
                 preload_workers: int = 8) -> Dict[str, object]:
    cfg = nc.Config()
    cfg.DATASET = nc.normalize_dataset_name(dataset)
    df = nc.discover_dataset(cfg).reset_index(drop=True)
    if limit:
        df = (df.sample(n=min(limit, len(df)), random_state=0)
                .reset_index(drop=True))

    feat = pd.read_csv(concept_csv)
    feat["path"] = feat["path"].astype(str)
    feat = feat.drop_duplicates(subset=["path"], keep="last").set_index("path")
    feature_names = [c for c in feat.columns if c not in {"path", "filename"}]
    by_base = {os.path.basename(p): i for i, p in enumerate(feat.index)}
    idx = [by_base[os.path.basename(str(p))] for p in df["path"]]
    Z = np.nan_to_num(feat[feature_names].to_numpy(dtype=np.float32)[idx])

    n = len(df)
    X = np.empty((n, int(SR * MAX_SECONDS)), dtype=np.float32)

    def _load(i):
        y = nc.load_audio_fixed(df.path.iloc[i], sr=SR, max_seconds=MAX_SECONDS)
        X[i] = (y - y.mean()) / (y.std() + 1e-7)

    with ThreadPoolExecutor(max_workers=preload_workers) as pool:
        list(pool.map(_load, range(n)))

    return {
        "X": X, "Z": Z,
        "speakers": df["speaker"].astype(str).to_numpy(dtype=str),
        "y": df["emotion"].to_numpy(dtype=np.int64),
        "feature_names": feature_names,
        "n_emotions": int(df["emotion"].max()) + 1,
    }


def _load_w2v2():
    from transformers import Wav2Vec2Model
    try:
        m = Wav2Vec2Model.from_pretrained(MODEL_NAME, local_files_only=True)
    except OSError:
        m = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    return m


class FTModel(nn.Module):
    def __init__(self, mode: str, n_emotions: int, n_speakers: int,
                 use_style_branch: bool = True):
        super().__init__()
        self.mode = mode
        self.w2v2 = _load_w2v2()
        self.w2v2.freeze_feature_encoder()
        dim = self.w2v2.config.hidden_size
        if mode == "plain":
            self.head = _head(dim, 128, n_emotions)
        elif mode in ("factor", "cmn"):
            self.head = W2V2DualCBM(dim, n_emotions, n_speakers,
                                    use_style_branch=use_style_branch)
        elif mode == "operator":
            self.head = W2V2OperatorCBM(dim, n_emotions)
        else:
            raise ValueError(mode)

    def conv_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.w2v2.feature_extractor(x).transpose(1, 2)

    def pooled(self, x: torch.Tensor,
               conv_shift: Optional[torch.Tensor] = None) -> torch.Tensor:
        if conv_shift is None:
            h = self.w2v2(x).last_hidden_state
        else:
            feats = self.conv_features(x) - conv_shift[:, None, :]
            hidden, _ = self.w2v2.feature_projection(feats)
            h = self.w2v2.encoder(hidden).last_hidden_state
        return h.mean(dim=1)


@torch.no_grad()
def pooled_split(model: FTModel, X: np.ndarray, device: str, batch: int,
                 conv_shift: Optional[np.ndarray] = None,
                 autocast: bool = False) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(X), batch):
        xb = torch.tensor(X[i:i + batch], device=device)
        cs = (torch.tensor(conv_shift[i:i + batch], device=device)
              if conv_shift is not None else None)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=autocast):
            outs.append(model.pooled(xb, conv_shift=cs).float().cpu().numpy())
    return np.concatenate(outs)


@torch.no_grad()
def speaker_conv_means(model: FTModel, X: np.ndarray, speakers: np.ndarray,
                       device: str, batch: int) -> np.ndarray:
    model.eval()
    sums: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    for i in range(0, len(X), batch):
        feats = model.conv_features(torch.tensor(X[i:i + batch], device=device))
        m = feats.mean(dim=1).float().cpu().numpy()
        for j, spk in enumerate(speakers[i:i + batch]):
            sums[spk] = sums.get(spk, 0.0) + m[j]
            counts[spk] = counts.get(spk, 0) + 1
    per_spk = {s: (sums[s] / counts[s]).astype(np.float32) for s in sums}
    return np.stack([per_spk[s] for s in speakers])


def train_one(task: Dict) -> Dict:
    mode = str(_W["mode"])
    data = _W["data"]
    device = str(_W["device"])
    epochs, batch = int(_W["epochs"]), int(_W["batch"])
    kappa = float(task.get("kappa", 0.0))
    seed, fold = int(task["seed"]), int(task["fold"])
    use_amp = device.startswith("cuda")

    nc.set_seed(seed)
    X, Z = data["X"], data["Z"]
    speakers, y_all = data["speakers"], data["y"]
    n_emotions = int(data["n_emotions"])

    outer = GroupKFold(n_splits=5)
    trainval_idx, test_idx = list(
        outer.split(np.arange(len(y_all)), y_all, speakers))[fold - 1]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed + fold)
    tr_rel, va_rel = next(gss.split(trainval_idx, y_all[trainval_idx],
                                    speakers[trainval_idx]))
    tr, va, te = trainval_idx[tr_rel], trainval_idx[va_rel], test_idx

    train_speakers = sorted(set(speakers[tr]))
    spk_to_local = {s: i for i, s in enumerate(train_speakers)}
    spk_local_np = np.array([spk_to_local[s] for s in speakers[tr]])
    weights = nc.class_weights_from_labels(y_all[tr], n_emotions).to(device)

    if mode != "plain":
        targets = build_targets(Z, speakers, tr, va, te, data["feature_names"])
        aff_t = torch.tensor(targets["train"][0], device=device)
        style_t = torch.tensor(targets["train"][1], device=device)

    no_style = bool(_W.get("no_style", False))
    model = FTModel(mode, n_emotions, len(train_speakers),
                    use_style_branch=not no_style).to(device)

    shifts: Dict[str, Optional[np.ndarray]] = {"tr": None, "va": None, "te": None}
    if mode == "cmn":
        shifts["tr"] = speaker_conv_means(model, X[tr], speakers[tr], device, batch)
        shifts["va"] = speaker_conv_means(model, X[va], speakers[va], device, batch)
        shifts["te"] = speaker_conv_means(model, X[te], speakers[te], device, batch)

    enc_params = [p for p in model.w2v2.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": float(_W.get("encoder_lr", ENCODER_LR))},
         {"params": head_params, "lr": HEAD_LR}], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    gen = torch.Generator().manual_seed(seed)

    lam_adv = ssl_sweep.BASE_LAMBDA_AFF_SPK_ADV * kappa
    lam_orth = ssl_sweep.BASE_LAMBDA_ORTH * kappa
    is_factor = mode in ("factor", "cmn")

    def eval_split(rows_key: str, rows: np.ndarray) -> Dict[str, np.ndarray]:
        pooled = pooled_split(model, X[rows], device, batch,
                              conv_shift=shifts[rows_key], autocast=use_amp)
        if mode == "plain":
            with torch.no_grad():
                logits = model.head(torch.tensor(pooled, device=device)).cpu().numpy()
            return {"h": pooled, "emotion_logits": logits}
        if is_factor:
            out = ssl_sweep._eval_factor(model.head, pooled, device)
        else:
            out = ssl_sweep._eval_operator(model.head, pooled, speakers[rows], device)
        out["h"] = pooled
        return out

    best_score, best_state, best_epoch, bad = -1e9, None, 0, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        if mode == "operator":
            pooled_tr = pooled_split(model, X[tr], device, batch, autocast=use_amp)
            with torch.no_grad():
                h192 = model.head.encoder(torch.tensor(pooled_tr, device=device))
            baselines = torch.zeros(len(train_speakers), H_DIM, device=device)
            counts = torch.zeros(len(train_speakers), device=device)
            sl = torch.tensor(spk_local_np, device=device)
            counts.index_add_(0, sl, torch.ones(len(sl), device=device))
            baselines.index_add_(0, sl, h192)
            baselines /= counts.clamp(min=1).unsqueeze(1)
        grl = nc.grl_schedule(epoch - 1, epochs, 1.0, 0)
        model.train()
        perm = torch.randperm(len(tr), generator=gen).numpy()
        for i in range(0, len(tr), batch):
            bidx = perm[i:i + batch]
            rows = tr[bidx]
            xb = torch.tensor(X[rows], device=device)
            yb = torch.tensor(y_all[rows], device=device)
            cs = (torch.tensor(shifts["tr"][bidx], device=device)
                  if shifts["tr"] is not None else None)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=use_amp):
                pooled = model.pooled(xb, conv_shift=cs)
                if mode == "plain":
                    loss = F.cross_entropy(model.head(pooled), yb, weight=weights)
                elif is_factor:
                    out = model.head(pooled, grl_lambda=grl)
                    loss = F.cross_entropy(out["emotion_logits"], yb, weight=weights)
                    loss = loss + ssl_sweep.LAMBDA_AFF_CONCEPT * F.smooth_l1_loss(
                        out["c_aff"], aff_t[torch.tensor(bidx, device=device)])
                    loss = loss + ssl_sweep.LAMBDA_STYLE_CONCEPT * F.smooth_l1_loss(
                        out["c_style"], style_t[torch.tensor(bidx, device=device)])
                    sl_b = torch.tensor(spk_local_np[bidx], device=device)
                    loss = loss + ssl_sweep.LAMBDA_STYLE_SPEAKER * F.cross_entropy(
                        out["style_speaker_logits"], sl_b)
                    if kappa > 0:
                        loss = loss + lam_adv * F.cross_entropy(
                            out["aff_speaker_adv_logits"], sl_b)
                        loss = loss + lam_orth * nc.batch_correlation_penalty(
                            out["c_aff"], out["c_style"])
                else:
                    sl_b = torch.tensor(spk_local_np[bidx], device=device)
                    out = model.head(pooled, baseline=baselines[sl_b].detach())
                    loss = F.cross_entropy(out["emotion_logits"], yb, weight=weights)
                    loss = loss + ssl_sweep.LAMBDA_AFF_CONCEPT * F.smooth_l1_loss(
                        out["concepts"], aff_t[torch.tensor(bidx, device=device)])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        scheduler.step()

        vm = nc.compute_metrics(y_all[va], eval_split("va", va)["emotion_logits"])
        score = vm["uar"] + 0.5 * vm["macro_f1"]
        if score > best_score:
            best_score, best_epoch, bad = score, epoch, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= FT_PATIENCE:
                break
        print(f"    [hb {mode} s{seed} f{fold}] epoch {epoch}/{epochs} "
              f"val_uar={vm['uar']:.3f}", flush=True)
    model.load_state_dict(best_state)

    reps = eval_split("te", te)
    tm = nc.compute_metrics(y_all[te], reps["emotion_logits"])
    row = {
        "dataset": _W["dataset"], "mode": mode, "no_style": no_style,
        "encoder_lr": float(_W.get("encoder_lr", ENCODER_LR)),
        "seed": seed, "fold": fold,
        "kappa": kappa, "best_epoch": best_epoch,
        "train_seconds": round(time.time() - t0, 1),
        "test_uar": tm["uar"], "test_macro_f1": tm["macro_f1"], "test_acc": tm["acc"],
    }
    rep_names = {"plain": ("h",), "factor": ("c_aff", "h"),
                 "cmn": ("c_aff", "h"), "operator": ("concepts", "delta", "h")}[mode]
    for rep in rep_names:
        for probe in ("linear", "mlp"):
            mi = mi_lower_bound_bits(reps[rep], speakers[te], probe=probe, seed=seed)
            row[f"{rep}_{probe}_mi_lb_bits"] = mi["mi_lb_bits"]
            row[f"{rep}_{probe}_control_bits"] = mi["control_mi_lb_bits"]
            row[f"{rep}_{probe}_entropy_bits"] = mi["entropy_bits"]
    return row


def _init_worker(mode, dataset, concept_csv, epochs, batch, device_override,
                 limit, no_style=False, encoder_lr=ENCODER_LR):
    torch.set_num_threads(4)
    device = nc.select_device(device_override or "auto")
    nc.configure_torch_runtime(device)
    _W.update(mode=mode, dataset=dataset, epochs=epochs, batch=batch,
              device=str(device), no_style=bool(no_style),
              encoder_lr=float(encoder_lr),
              data=load_ft_data(dataset, concept_csv, limit=limit))


def _format_row(r: Dict) -> str:
    rep = {"plain": "h", "factor": "c_aff", "cmn": "c_aff",
           "operator": "concepts"}[r["mode"]]
    return (f"[{r['mode']} seed={r['seed']} fold={r['fold']} kappa={r['kappa']:g}] "
            f"UAR={r['test_uar']:.4f}  I_LB({rep};S)={r[f'{rep}_linear_mi_lb_bits']:.3f}b "
            f"epoch*={r['best_epoch']}  {r['train_seconds']}s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["plain", "factor", "cmn", "operator"],
                   default=None)
    p.add_argument("--dataset", default="cremad",
                   choices=["cremad", "iemocap", "ravdess"])
    p.add_argument("--concept-csv", default=None)
    p.add_argument("--results-root", default=os.path.dirname(OUT_ROOT),
                   help="root containing experiment results and data caches (default: outputs)")
    p.add_argument("--kappas", default="0",
                   help="factor mode: comma-separated pressure points")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--folds", default="1,3,5")
    p.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    p.add_argument("--batch", type=int, default=BATCH_DEFAULT)
    p.add_argument("--jobs", type=int, default=1,
                   help="concurrent fine-tunes on one GPU (2 fits in 32 GB)")
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=0,
                   help="subsample utterances (smoke/debug)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny local run through all four modes")
    p.add_argument("--encoder-lr", type=float, default=ENCODER_LR,
                   help="wav2vec 2.0 encoder learning rate, 0 freezes the encoder")
    p.add_argument("--tag", default="",
                   help="suffix for the results filename")
    p.add_argument("--no-style-branch", dest="no_style", action="store_true",
                   help="drop the style branch to match the operator head")
    args = p.parse_args()

    out_root = os.path.join(args.results_root, "finetune")

    def concept_csv_for(ds):
        return args.concept_csv or os.path.join(
            args.results_root, "floor_sweep", ds,
            "concept_feature_cache_opensmile_eGeMAPSv02.csv")

    if args.smoke:
        os.makedirs(os.path.join(out_root, "ravdess"), exist_ok=True)
        rows = []
        for mode in ("plain", "factor", "cmn", "operator"):
            _init_worker(mode, "ravdess", concept_csv_for("ravdess"),
                         epochs=2, batch=4, device_override=args.device,
                         limit=160)
            r = train_one({"seed": 42, "fold": 1, "kappa": 0.0})
            rows.append(r)
            print("SMOKE " + _format_row(r), flush=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(out_root, "ravdess", "ft_smoke.csv"), index=False)
        print("smoke ok")
        return

    if args.mode is None:
        p.error("--mode is required (or --smoke)")
    kappas = ([float(x) for x in args.kappas.split(",")]
              if args.mode == "factor" else [0.0])
    tasks = [{"seed": int(s), "fold": int(f), "kappa": k}
             for s in args.seeds.split(",")
             for f in args.folds.split(",")
             for k in kappas]

    out_dir = os.path.join(out_root, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    suffix = ("_nostyle" if args.no_style else "") + (f"_{args.tag}" if args.tag else "")
    results_csv = os.path.join(out_dir, f"ft_{args.mode}{suffix}_results.csv")
    init_args = (args.mode, args.dataset, concept_csv_for(args.dataset),
                 args.epochs, args.batch, args.device, args.limit, args.no_style,
                 args.encoder_lr)
    print(f"{len(tasks)} fine-tune runs (mode={args.mode}, "
          f"dataset={args.dataset}), jobs={args.jobs}")

    rows: List[Dict] = []
    if args.jobs <= 1:
        _init_worker(*init_args)
        for t in tasks:
            rows.append(train_one(t))
            print(f"({len(rows)}/{len(tasks)}) " + _format_row(rows[-1]), flush=True)
            pd.DataFrame(rows).to_csv(results_csv, index=False)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx,
                                 initializer=_init_worker,
                                 initargs=init_args) as pool:
            futures = {pool.submit(train_one, t): t for t in tasks}
            for fut in as_completed(futures):
                try:
                    row = fut.result()
                except Exception:
                    print(f"FAILED {futures[fut]}:\n{traceback.format_exc()}",
                          flush=True)
                    continue
                rows.append(row)
                print(f"({len(rows)}/{len(tasks)}) " + _format_row(row), flush=True)
                pd.DataFrame(rows).to_csv(results_csv, index=False)

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["seed", "fold", "kappa"])
    out.to_csv(results_csv, index=False)
    if len(out) != len(tasks):
        raise RuntimeError(
            f"{len(tasks) - len(out)} of {len(tasks)} runs failed. "
            f"Partial results saved to {results_csv}"
        )
    print(f"Saved {len(out)}/{len(tasks)} rows -> {results_csv}")


if __name__ == "__main__":
    main()
