"""Once-only enrollment audited on held-out utterances"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

import enroll_grid as eg
from mi_estimation import mi_lower_bound_bits

KS = [1, 2, 5, 10]
HALF = -1
GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
DRAWS = 2


def draw_split(speakers: np.ndarray, seed: int, fold: int, draw: int) -> Dict[str, object]:
    rng = np.random.default_rng(100_000 * seed + 100 * fold + draw)
    perm: Dict[str, np.ndarray] = {}
    query = np.zeros(len(speakers), dtype=bool)
    for s in np.unique(speakers):
        idx = rng.permutation(np.where(speakers == s)[0])
        perm[s] = idx
        query[idx[len(idx) // 2:]] = True
    return {"perm": perm, "query": query}


def enroll_rows(taps: np.ndarray, speakers: np.ndarray, perm: Dict[str, np.ndarray],
                k: int) -> np.ndarray:
    b = np.empty_like(taps)
    for s, idx in perm.items():
        n = len(idx) // 2 if k == HALF else k
        b[speakers == s] = taps[idx[:n]].mean(axis=0)
    return b


def resampled_rows(taps: np.ndarray, speakers: np.ndarray, perm: Dict[str, np.ndarray],
                   k: int, rng: np.random.Generator) -> np.ndarray:
    b = np.empty_like(taps)
    for s, idx in perm.items():
        pool = idx[:len(idx) // 2]
        for i in np.where(speakers == s)[0]:
            b[i] = taps[rng.choice(pool, size=min(k, len(pool)), replace=False)].mean(axis=0)
    return b


def half_sizes(speakers: np.ndarray, perm: Dict[str, np.ndarray]) -> np.ndarray:
    out = np.empty(len(speakers), dtype=np.float64)
    for s, idx in perm.items():
        out[speakers == s] = len(idx) // 2
    return out


@torch.no_grad()
def tap_profiles(model, loader, device: str, pos: int) -> np.ndarray:
    model.eval()
    out = []
    for batch in loader:
        t = model.encoder(batch["x"].to(device), baselines=None, gate=model.gate(),
                          taps_for={pos})["taps"][pos]
        out.append(t.float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def forward_rows(model, loader, device: str, pos: int, rows: np.ndarray):
    model.eval()
    rows_t = torch.as_tensor(rows, dtype=torch.float32)
    cs, ls, off = [], [], 0
    for batch in loader:
        x = batch["x"].to(device)
        n = x.shape[0]
        b: List[Optional[torch.Tensor]] = [None] * 5
        b[pos] = rows_t[off:off + n].to(device)
        off += n
        out = model(x, baselines=b)
        cs.append(out["concepts"].cpu().numpy())
        ls.append(out["emotion_logits"].cpu().numpy())
    return np.concatenate(cs), np.concatenate(ls)


def _score(concepts, logits, y, speakers, mask, seed, metrics_fn, mlp: bool) -> Dict[str, float]:
    met = metrics_fn(y[mask], logits[mask])
    lin = mi_lower_bound_bits(concepts[mask], speakers[mask], probe="linear", seed=seed)
    rec = {"test_uar": met["uar"], "test_acc": met["acc"], "n_rows": int(mask.sum()),
           "leak_bits": lin["mi_lb_bits"], "control_bits": lin["control_mi_lb_bits"],
           "leak_mlp_bits": float("nan"), "control_mlp_bits": float("nan")}
    if mlp:
        m = mi_lower_bound_bits(concepts[mask], speakers[mask], probe="mlp", seed=seed)
        rec["leak_mlp_bits"], rec["control_mlp_bits"] = m["mi_lb_bits"], m["control_mi_lb_bits"]
    return rec


def evaluate(model, loader, device: str, pos: int, speakers: np.ndarray, y: np.ndarray,
             seed: int, fold: int, metrics_fn, h_raw: Optional[np.ndarray] = None,
             tau2: float = float("nan"), within: float = float("nan"),
             prior: Optional[np.ndarray] = None, draws: int = DRAWS) -> List[Dict]:
    speakers = np.asarray(speakers).astype(str)
    shrink = pos == 4
    if shrink:
        assert h_raw is not None and prior is not None
        taps = np.asarray(h_raw, dtype=np.float64)
    else:
        taps = tap_profiles(model, loader, device, pos).astype(np.float64)
    base = {"pos": pos, "seed": seed, "fold": fold, "tau2": round(tau2, 4), "within": round(within, 4)}
    recs: List[Dict] = []

    def codes(b: np.ndarray, lam) -> tuple:
        if shrink:
            lam_col = np.asarray(lam, dtype=np.float64).reshape(-1, 1) if np.ndim(lam) else lam
            return eg.heads_on(model, taps - prior - lam_col * (b - prior), device)
        return forward_rows(model, loader, device, pos, b)

    def add(protocol, draw, k, lam, lam_star, is_star, concepts, logits, mask, labels, mlp=False):
        rec = dict(base, protocol=protocol, draw=draw, k=k, lam=lam, lam_star=lam_star,
                   is_lam_star=is_star)
        rec.update(_score(concepts, logits, y, labels, mask, seed, metrics_fn, mlp))
        recs.append(rec)

    for d in range(draws):
        split = draw_split(speakers, seed, fold, d)
        perm, q = split["perm"], split["query"]
        rng = np.random.default_rng(7_919 * seed + 31 * fold + d)
        for k in KS + [HALF]:
            b = enroll_rows(taps, speakers, perm, k)
            if not shrink:
                c, l = codes(b, 1.0)
                add("fixed", d, k, 1.0, float("nan"), 0, c, l, q, speakers, mlp=(k == HALF))
                continue
            if k == HALF:
                ls_row = np.array([eg.lambda_k(tau2, within, int(n)) for n in half_sizes(speakers, perm)])
                c, l = codes(b, ls_row)
                add("fixed", d, k, round(float(ls_row[q].mean()), 4), round(float(ls_row[q].mean()), 4),
                    1, c, l, q, speakers, mlp=True)
                c, l = codes(b, 1.0)
                add("fixed", d, k, 1.0, round(float(ls_row[q].mean()), 4), 0, c, l, q, speakers, mlp=True)
                continue
            lstar = round(eg.lambda_k(tau2, within, k), 4)
            for lam, is_star in [(lstar, 1)] + [(x, 0) for x in GRID]:
                c, l = codes(b, lam)
                add("fixed", d, k, lam, lstar, is_star, c, l, q, speakers)
        for k in KS:
            b = resampled_rows(taps, speakers, perm, k, rng)
            lams = [(1.0, 0)] if not shrink else [(round(eg.lambda_k(tau2, within, k), 4), 1), (1.0, 0)]
            for lam, is_star in lams:
                c, l = codes(b, lam)
                add("resampled", d, k, lam, lams[0][0] if shrink else float("nan"), is_star,
                    c, l, q, speakers)

    rng = np.random.default_rng(104_729 * seed + fold)
    pseudo = speakers[rng.permutation(len(speakers))]
    b = np.empty_like(taps)
    for s in np.unique(pseudo):
        b[pseudo == s] = taps[pseudo == s].mean(axis=0)
    c, l = eg.heads_on(model, taps - b, device) if shrink else forward_rows(model, loader, device, pos, b)
    add("pseudo_split", 0, 0, 1.0, float("nan"), 0, c, l, np.ones(len(speakers), bool), pseudo, mlp=True)
    return recs
