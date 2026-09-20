"""Frozen wav2vec 2.0 with factor sweeps, input-CMN and the shift operator"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import RobustScaler

from nc_bridge import nc, NC_ROOT
from mi_estimation import mi_lower_bound_bits

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "ssl")

W2V2_DIRS = {
    "cremad": "wav2vec2_frozen_outputs",
    "iemocap": "iemocap_wav2vec2_frozen_outputs",
    "ravdess": "ravdess_wav2vec2_frozen_outputs",
}
BASE_LAMBDA_AFF_SPK_ADV = 0.35
BASE_LAMBDA_ORTH = 0.05
LAMBDA_AFF_CONCEPT = 1.50
LAMBDA_STYLE_CONCEPT = 0.50
LAMBDA_STYLE_SPEAKER = 0.50
H_DIM, ENC_HIDDEN, N_CONCEPTS = 192, 256, 6
LR, WEIGHT_DECAY, BATCH, EPOCHS_DEFAULT, PATIENCE = 1e-3, 1e-4, 32, 40, 10

_W: Dict[str, object] = {}


def load_data(dataset: str, concept_csv: str) -> Dict[str, object]:
    d = os.path.join(NC_ROOT, W2V2_DIRS[dataset])
    X = np.load(os.path.join(d, "wav2vec2_embeddings.npy")).astype(np.float32)
    md = pd.read_csv(os.path.join(d, "wav2vec2_metadata.csv"))
    assert len(md) == len(X)

    feat = pd.read_csv(concept_csv)
    feat["path"] = feat["path"].astype(str)
    feat = feat.drop_duplicates(subset=["path"], keep="last").set_index("path")
    feature_names = [c for c in feat.columns if c not in {"path", "filename"}]
    by_base = {os.path.basename(p): i for i, p in enumerate(feat.index)}
    idx = [by_base[os.path.basename(str(p))] for p in md["path"]]
    Z = feat[feature_names].to_numpy(dtype=np.float32)[idx]

    return {
        "X": X,
        "Z": np.nan_to_num(Z),
        "speakers": md["speaker"].astype(str).to_numpy(dtype=str),
        "y": md["emotion"].to_numpy(dtype=np.int64),
        "feature_names": feature_names,
        "n_emotions": int(md["emotion"].max()) + 1,
    }


def build_targets(Z, speakers, tr, va, te, feature_names):
    scaler = RobustScaler().fit(Z[tr])
    out = {}
    for name, rows in (("train", tr), ("val", va), ("test", te)):
        z = np.clip(scaler.transform(Z[rows]), -5.0, 5.0).astype(np.float32)
        spk = speakers[rows]
        baselines = nc.compute_speaker_baselines(z, spk)
        gb = np.mean(np.clip(scaler.transform(Z[tr]), -5, 5), axis=0).astype(np.float32)
        bmat = nc.baseline_matrix_for_rows(z, spk, baseline_by_speaker=baselines, fallback_global=gb)
        aff, style = nc.build_concepts_from_baseline_and_deviation(z, bmat, feature_names)
        out[name] = (aff.astype(np.float32), style.astype(np.float32))
    return out


def make_encoder(in_dim: int, dropout: float = 0.25) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, ENC_HIDDEN), nn.ReLU(inplace=True), nn.Dropout(dropout),
        nn.Linear(ENC_HIDDEN, H_DIM), nn.ReLU(inplace=True),
    )


def _head(in_dim, hidden, out_dim, dropout=0.25, sigmoid=False):
    layers = [nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
              nn.Linear(hidden, out_dim)]
    if sigmoid:
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


class W2V2DualCBM(nn.Module):
    def __init__(self, in_dim: int, n_emotions: int, n_speakers: int,
                 use_style_branch: bool = True):
        super().__init__()
        self.use_style_branch = use_style_branch
        self.encoder = make_encoder(in_dim)
        self.aff_head = _head(H_DIM, 128, N_CONCEPTS, sigmoid=True)
        self.style_head = _head(H_DIM, 128, N_CONCEPTS, sigmoid=True)
        self.emotion_head = _head(N_CONCEPTS, 64, n_emotions)
        self.style_speaker_head = _head(N_CONCEPTS, 96, n_speakers)
        self.aff_speaker_adv_head = _head(N_CONCEPTS, 96, n_speakers)

    def forward(self, x, grl_lambda: float = 1.0):
        h = self.encoder(x)
        c_aff = self.aff_head(h)
        if self.use_style_branch:
            c_style = self.style_head(h)
        else:
            c_style = torch.zeros(h.shape[0], N_CONCEPTS,
                                  dtype=h.dtype, device=h.device)
        return {
            "h": h, "c_aff": c_aff, "c_style": c_style,
            "emotion_logits": self.emotion_head(c_aff),
            "style_speaker_logits": self.style_speaker_head(c_style),
            "aff_speaker_adv_logits": self.aff_speaker_adv_head(nc.grad_reverse(c_aff, grl_lambda)),
        }


class W2V2CemCBM(nn.Module):
    EMB_DIM = 16

    def __init__(self, in_dim: int, n_emotions: int, n_speakers: int):
        super().__init__()
        self.encoder = make_encoder(in_dim)
        e = self.EMB_DIM
        self.aff_ctx = nn.ModuleList(
            [nn.Sequential(nn.Linear(H_DIM, 2 * e), nn.LeakyReLU(0.1)) for _ in range(N_CONCEPTS)])
        self.style_ctx = nn.ModuleList(
            [nn.Sequential(nn.Linear(H_DIM, 2 * e), nn.LeakyReLU(0.1)) for _ in range(N_CONCEPTS)])
        self.aff_score = nn.Linear(2 * e, 1)
        self.style_score = nn.Linear(2 * e, 1)
        d = N_CONCEPTS * e
        self.emotion_head = _head(d, 64, n_emotions)
        self.style_speaker_head = _head(d, 96, n_speakers)
        self.aff_speaker_adv_head = _head(d, 96, n_speakers)

    def _block(self, h, ctx_list, scorer):
        acts, embs = [], []
        for ctx in ctx_list:
            u = ctx(h)
            p = torch.sigmoid(scorer(u))
            pos, neg = u.chunk(2, dim=-1)
            embs.append(p * pos + (1 - p) * neg)
            acts.append(p)
        return torch.cat(acts, dim=-1), torch.cat(embs, dim=-1)

    def forward(self, x, grl_lambda: float = 1.0):
        h = self.encoder(x)
        c_aff, z_aff = self._block(h, self.aff_ctx, self.aff_score)
        c_style, z_style = self._block(h, self.style_ctx, self.style_score)
        return {
            "h": h, "c_aff": c_aff, "c_style": c_style,
            "z_aff": z_aff, "z_style": z_style,
            "emotion_logits": self.emotion_head(z_aff),
            "style_speaker_logits": self.style_speaker_head(z_style),
            "aff_speaker_adv_logits": self.aff_speaker_adv_head(nc.grad_reverse(z_aff, grl_lambda)),
        }


class W2V2OperatorCBM(nn.Module):
    def __init__(self, in_dim: int, n_emotions: int):
        super().__init__()
        self.encoder = make_encoder(in_dim)
        self.concept_head = _head(H_DIM, 128, N_CONCEPTS, sigmoid=True)
        self.emotion_head = _head(N_CONCEPTS, 64, n_emotions)

    def forward(self, x, baseline, scale: Optional[torch.Tensor] = None):
        h = self.encoder(x)
        delta = h - baseline
        if scale is not None:
            delta = delta * scale
        concepts = self.concept_head(delta)
        return {"h": h, "concepts": concepts, "emotion_logits": self.emotion_head(concepts)}


def _batches(n: int, gen: torch.Generator):
    perm = torch.randperm(n, generator=gen)
    for i in range(0, n, BATCH):
        yield perm[i:i + BATCH]


def _speaker_means_np(h: np.ndarray, speakers: np.ndarray) -> np.ndarray:
    out = np.empty_like(h)
    for spk in np.unique(speakers):
        m = speakers == spk
        out[m] = h[m].mean(axis=0)
    return out


AFFINE_EPS = 0.10


def _speaker_stats_np(h: np.ndarray, speakers: np.ndarray):
    mean = np.empty_like(h)
    scale = np.ones_like(h)
    for spk in np.unique(speakers):
        m = speakers == spk
        mean[m] = h[m].mean(axis=0)
        scale[m] = 1.0 / (h[m].std(axis=0) + AFFINE_EPS)
    return mean, scale


@torch.no_grad()
def _eval_factor(model, X, device) -> Dict[str, np.ndarray]:
    model.eval()
    out = model(torch.tensor(X, device=device), grl_lambda=0.0)
    keys = ("h", "c_aff", "c_style", "z_aff", "z_style", "emotion_logits")
    return {k: v.cpu().numpy() for k, v in out.items() if k in keys}


@torch.no_grad()
def _eval_operator(model, X, speakers, device, affine: bool = False) -> Dict[str, np.ndarray]:
    model.eval()
    h = model.encoder(torch.tensor(X, device=device))
    h_np = h.cpu().numpy()
    if affine:
        mean, scale = _speaker_stats_np(h_np, speakers)
        delta = (h - torch.tensor(mean, device=device)) * torch.tensor(scale, device=device)
    else:
        delta = h - torch.tensor(_speaker_means_np(h_np, speakers), device=device)
    concepts = model.concept_head(delta)
    return {"h": h_np, "delta": delta.cpu().numpy(),
            "concepts": concepts.cpu().numpy(),
            "emotion_logits": model.emotion_head(concepts).cpu().numpy()}


def _nontrivial_perm(n: int, rng: np.random.Generator) -> np.ndarray:
    perm = rng.permutation(n)
    fixed = perm == np.arange(n)
    if fixed.any():
        perm[fixed] = np.roll(perm[fixed], 1)
    return perm


@torch.no_grad()
def _swap_metrics(model, reps: Dict[str, np.ndarray], speakers: np.ndarray,
                  device, seed: int, affine: bool) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    pred0 = reps["emotion_logits"].argmax(1)

    perm = _nontrivial_perm(len(pred0), rng)
    swapped = torch.tensor(reps["concepts"][perm], device=device)
    pred_swap = model.emotion_head(swapped).cpu().numpy().argmax(1)
    concept_swap_sensitivity = float((pred_swap != pred0).mean())

    h = reps["h"]
    halves_pred = []
    half_assign = {spk: rng.permutation(np.where(speakers == spk)[0])
                   for spk in np.unique(speakers)}
    for half in (0, 1):
        baseline = np.empty_like(h)
        scale = np.ones_like(h)
        for spk, shuffled in half_assign.items():
            mid = max(len(shuffled) // 2, 1)
            part = shuffled[:mid] if half == 0 else shuffled[mid:]
            if len(part) == 0:
                part = shuffled
            rows = np.where(speakers == spk)[0]
            baseline[rows] = h[part].mean(axis=0)
            if affine:
                scale[rows] = 1.0 / (h[part].std(axis=0) + AFFINE_EPS)
        delta = torch.tensor((h - baseline) * (scale if affine else 1.0),
                             dtype=torch.float32, device=device)
        concepts = model.concept_head(delta)
        halves_pred.append(model.emotion_head(concepts).cpu().numpy().argmax(1))
    enroll_resample_consistency = float((halves_pred[0] == halves_pred[1]).mean())

    return {"concept_swap_sensitivity": concept_swap_sensitivity,
            "enroll_resample_consistency": enroll_resample_consistency}


@torch.no_grad()
def _eval_operator_enroll_k(model, X, speakers, device, k: int, seed: int) -> Dict[str, np.ndarray]:
    model.eval()
    h = model.encoder(torch.tensor(X, device=device)).cpu().numpy()
    rng = np.random.default_rng(seed)
    baseline = np.empty_like(h)
    for spk in np.unique(speakers):
        idxs = np.where(speakers == spk)[0]
        for i in idxs:
            others = idxs[idxs != i]
            pick = others if len(others) <= k else rng.choice(others, size=k, replace=False)
            baseline[i] = h[pick].mean(axis=0)
    delta = torch.tensor(h - baseline, device=device)
    concepts = model.concept_head(delta)
    return {"concepts": concepts.cpu().numpy(),
            "emotion_logits": model.emotion_head(concepts).cpu().numpy()}


def train_one(task: Dict) -> Dict:
    mode = str(_W["mode"])
    data = _W["data"]
    device = str(_W["device"])
    epochs = int(_W["epochs"])
    seed, fold_wanted = int(task["seed"]), int(task["fold"])
    kappa = float(task.get("kappa", 0.0))

    nc.set_seed(seed)
    X, Z = data["X"], data["Z"]
    speakers, y_all = data["speakers"], data["y"]
    n_emotions = data["n_emotions"]

    outer = GroupKFold(n_splits=5)
    trainval_idx, test_idx = list(outer.split(np.arange(len(y_all)), y_all, speakers))[fold_wanted - 1]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed + fold_wanted)
    tr_rel, va_rel = next(gss.split(trainval_idx, y_all[trainval_idx], speakers[trainval_idx]))
    tr, va, te = trainval_idx[tr_rel], trainval_idx[va_rel], test_idx

    if mode == "cmn":
        X = X.copy()
        for rows in (tr, va, te):
            X[rows] = X[rows] - _speaker_means_np(X[rows], speakers[rows])
        mode = "floor"

    targets = build_targets(Z, speakers, tr, va, te, data["feature_names"])
    train_speakers = sorted(set(speakers[tr]))
    spk_to_local = {s: i for i, s in enumerate(train_speakers)}
    spk_local = torch.tensor([spk_to_local[s] for s in speakers[tr]], device=device)

    Xtr = torch.tensor(X[tr], device=device)
    ytr = torch.tensor(y_all[tr], device=device)
    aff_t = torch.tensor(targets["train"][0], device=device)
    style_t = torch.tensor(targets["train"][1], device=device)
    weights = nc.class_weights_from_labels(y_all[tr], n_emotions).to(device)

    is_operator = mode in ("operator", "affine")
    is_factor = mode in ("floor", "cem")
    no_style = bool(_W.get("no_style", False))
    if mode == "floor":
        model = W2V2DualCBM(X.shape[1], n_emotions, len(train_speakers),
                            use_style_branch=not no_style).to(device)
    elif mode == "cem":
        model = W2V2CemCBM(X.shape[1], n_emotions, len(train_speakers)).to(device)
    else:
        model = W2V2OperatorCBM(X.shape[1], n_emotions).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    gen = torch.Generator().manual_seed(seed)

    arm = str(_W.get("arm", "both"))
    lam_adv = BASE_LAMBDA_AFF_SPK_ADV * kappa if arm in ("both", "adv") else 0.0
    lam_orth = BASE_LAMBDA_ORTH * kappa if arm in ("both", "orth") else 0.0

    best_score, best_state, best_epoch, bad = -1e9, None, 0, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        if is_operator:
            with torch.no_grad():
                model.eval()
                h_tr = model.encoder(Xtr)
                counts = torch.zeros(len(train_speakers), device=device)
                counts.index_add_(0, spk_local, torch.ones_like(spk_local, dtype=torch.float32))
                counts = counts.clamp(min=1).unsqueeze(1)
                baselines = torch.zeros(len(train_speakers), H_DIM, device=device)
                baselines.index_add_(0, spk_local, h_tr)
                baselines /= counts
                scales = None
                if mode == "affine":
                    sq = torch.zeros(len(train_speakers), H_DIM, device=device)
                    sq.index_add_(0, spk_local, h_tr ** 2)
                    var = (sq / counts - baselines ** 2).clamp(min=0)
                    scales = 1.0 / (var.sqrt() + AFFINE_EPS)
        grl = nc.grl_schedule(epoch - 1, epochs, 1.0, 0)
        model.train()
        for bidx in _batches(len(tr), gen):
            bidx_d = bidx.to(device)
            xb, yb = Xtr[bidx_d], ytr[bidx_d]
            if is_factor:
                out = model(xb, grl_lambda=grl)
                loss = F.cross_entropy(out["emotion_logits"], yb, weight=weights)
                loss = loss + LAMBDA_AFF_CONCEPT * F.smooth_l1_loss(out["c_aff"], aff_t[bidx_d])
                loss = loss + LAMBDA_STYLE_CONCEPT * F.smooth_l1_loss(out["c_style"], style_t[bidx_d])
                loss = loss + LAMBDA_STYLE_SPEAKER * F.cross_entropy(out["style_speaker_logits"], spk_local[bidx_d])
                if kappa > 0:
                    orth_a = out["z_aff"] if mode == "cem" else out["c_aff"]
                    orth_s = out["z_style"] if mode == "cem" else out["c_style"]
                    if lam_adv > 0:
                        loss = loss + lam_adv * F.cross_entropy(out["aff_speaker_adv_logits"], spk_local[bidx_d])
                    if lam_orth > 0:
                        loss = loss + lam_orth * nc.batch_correlation_penalty(orth_a, orth_s)
            else:
                bscale = scales[spk_local[bidx_d]].detach() if scales is not None else None
                out = model(xb, baseline=baselines[spk_local[bidx_d]].detach(), scale=bscale)
                loss = F.cross_entropy(out["emotion_logits"], yb, weight=weights)
                loss = loss + LAMBDA_AFF_CONCEPT * F.smooth_l1_loss(out["concepts"], aff_t[bidx_d])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        scheduler.step()

        if is_factor:
            val = _eval_factor(model, X[va], device)
        else:
            val = _eval_operator(model, X[va], speakers[va], device, affine=(mode == "affine"))
        vm = nc.compute_metrics(y_all[va], val["emotion_logits"])
        score = vm["uar"] + 0.5 * vm["macro_f1"]
        if score > best_score:
            best_score, best_epoch, bad = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)

    if mode == "cem":
        reps = _eval_factor(model, X[te], device)
        rep_names = ("z_aff", "c_aff", "h")
    elif mode == "floor":
        reps = _eval_factor(model, X[te], device)
        rep_names = ("c_aff", "c_style", "h")
    else:
        reps = _eval_operator(model, X[te], speakers[te], device, affine=(mode == "affine"))
        rep_names = ("concepts", "delta", "h")
    tm = nc.compute_metrics(y_all[te], reps["emotion_logits"])

    row = {
        "dataset": _W["dataset"], "mode": str(_W["mode"]), "arm": arm,
        "no_style": no_style,
        "seed": seed, "fold": fold_wanted,
        "kappa": kappa, "best_epoch": best_epoch,
        "train_seconds": round(time.time() - t0, 1),
        "test_uar": tm["uar"], "test_macro_f1": tm["macro_f1"], "test_acc": tm["acc"],
    }
    for rep in rep_names:
        for probe in ("linear", "mlp"):
            mi = mi_lower_bound_bits(reps[rep], speakers[te], probe=probe, seed=seed)
            row[f"{rep}_{probe}_mi_lb_bits"] = mi["mi_lb_bits"]
            row[f"{rep}_{probe}_control_bits"] = mi["control_mi_lb_bits"]
            row[f"{rep}_{probe}_entropy_bits"] = mi["entropy_bits"]

    if is_operator:
        row.update(_swap_metrics(model, reps, speakers[te], device, seed=seed,
                                 affine=(mode == "affine")))
        if mode == "operator":
            for k in (1, 2, 5, 10):
                ek = _eval_operator_enroll_k(model, X[te], speakers[te], device, k=k, seed=seed)
                km = nc.compute_metrics(y_all[te], ek["emotion_logits"])
                kmi = mi_lower_bound_bits(ek["concepts"], speakers[te], probe="linear", seed=seed)
                row[f"enrollk{k}_test_uar"] = km["uar"]
                row[f"enrollk{k}_concepts_mi_lb_bits"] = kmi["mi_lb_bits"]
    return row


def _init_worker(mode, dataset, concept_csv, epochs, arm="both", device_override=None,
                 no_style=False):
    torch.set_num_threads(2)
    device = nc.select_device(device_override or "auto")
    nc.configure_torch_runtime(device)
    _W.update(mode=mode, dataset=dataset, epochs=epochs, arm=arm, device=str(device),
              no_style=bool(no_style), data=load_data(dataset, concept_csv))


def _rep_for_mode(mode: str) -> str:
    if mode in ("operator", "affine"):
        return "concepts"
    return "z_aff" if mode == "cem" else "c_aff"


def _format_row(r: Dict) -> str:
    rep = _rep_for_mode(r["mode"])
    return (f"[{r['mode']} seed={r['seed']} fold={r['fold']} kappa={r['kappa']:g}] "
            f"UAR={r['test_uar']:.4f}  I_LB({rep};S)={r[f'{rep}_linear_mi_lb_bits']:.3f}b "
            f"(mlp {r[f'{rep}_mlp_mi_lb_bits']:.3f}b, ctrl={r[f'{rep}_linear_control_bits']:+.3f}b)  "
            f"epoch*={r['best_epoch']}  {r['train_seconds']}s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True,
                   choices=["floor", "operator", "affine", "cmn", "cem"])
    p.add_argument("--dataset", default="cremad", choices=["cremad", "iemocap", "ravdess"])
    p.add_argument("--results-root", default=os.path.dirname(OUT_ROOT),
                   help="root containing experiment results and data caches (default: outputs)")
    p.add_argument("--tag", default="",
                   help="optional filename tag after ssl_, e.g. local")
    p.add_argument("--concept-csv", default=None,
                   help="eGeMAPS concept CSV (default: outputs/floor_sweep/<ds>/concept_feature_cache_*.csv)")
    p.add_argument("--pressures", default="0",
                   help="floor/cem modes: comma-separated kappas")
    p.add_argument("--arm", default="both", choices=["both", "adv", "orth"],
                   help="floor mode: scale both anti-speaker terms or only one")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    p.add_argument("--max-folds", type=int, default=0)
    p.add_argument("--jobs", type=int, default=12 if torch.cuda.is_available() else 1)
    p.add_argument("--device", default=None,
                   help="override device (e.g. cpu for multi-process runs on a Mac)")
    p.add_argument("--no-style-branch", dest="no_style", action="store_true",
                   help="drop the style branch to match the operator head")
    args = p.parse_args()

    concept_csv = args.concept_csv or os.path.join(
        args.results_root, "floor_sweep", args.dataset, "concept_feature_cache_opensmile_eGeMAPSv02.csv")
    seeds = [int(x) for x in args.seeds.split(",")]
    kappas = ([float(x) for x in args.pressures.split(",")]
              if args.mode in ("floor", "cem") else [0.0])
    folds = range(1, (args.max_folds or 5) + 1)
    tasks = [{"seed": s, "fold": f, "kappa": k} for s in seeds for f in folds for k in kappas]

    out_dir = os.path.join(args.results_root, "ssl", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_arm{args.arm}" if args.arm != "both" else ""
    if args.no_style:
        suffix += "_nostyle"
    tag = f"{args.tag}_" if args.tag else ""
    results_csv = os.path.join(out_dir, f"ssl_{tag}{args.mode}{suffix}_results.csv")
    init_args = (args.mode, args.dataset, concept_csv, args.epochs, args.arm, args.device,
                 args.no_style)
    print(f"{len(tasks)} runs (mode={args.mode}, dataset={args.dataset}), jobs={args.jobs}")

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
                                 initializer=_init_worker, initargs=init_args) as pool:
            futures = {pool.submit(train_one, t): t for t in tasks}
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
        out = out.sort_values(["seed", "fold", "kappa"]).reset_index(drop=True)
    out.to_csv(results_csv, index=False)
    if len(out) != len(tasks):
        raise RuntimeError(
            f"{len(tasks) - len(out)} of {len(tasks)} runs failed. "
            f"Partial results saved to {results_csv}"
        )
    if len(out):
        rep = _rep_for_mode(args.mode)
        for k in sorted(out.kappa.unique()):
            sub = out[out.kappa == k]
            print(f"kappa={k:g}: UAR {sub.test_uar.mean():.4f}+/-{sub.test_uar.std():.4f}  "
                  f"I_LB({rep};S) {sub[f'{rep}_linear_mi_lb_bits'].mean():.3f}"
                  f"+/-{sub[f'{rep}_linear_mi_lb_bits'].std():.3f}b")
    print(f"Saved {len(out)}/{len(tasks)} rows -> {results_csv}")


if __name__ == "__main__":
    main()
