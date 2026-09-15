"""k-shot enrollment grid with shrinkage of the enrollment mean"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def variance_decomposition(h: np.ndarray, speakers: np.ndarray) -> Tuple[float, float]:
    h = np.asarray(h, dtype=np.float64)
    means, wvar, counts = [], [], []
    for s in np.unique(speakers):
        m = speakers == s
        if m.sum() < 2:
            continue
        means.append(h[m].mean(0))
        wvar.append(h[m].var(0, ddof=1))
        counts.append(int(m.sum()))
    if len(means) < 2:
        return 0.0, 1.0
    W = np.asarray(wvar).mean(0)
    B = np.maximum(np.asarray(means).var(0, ddof=1) - W / float(np.mean(counts)), 0.0)
    return float(B.sum()), float(W.sum())


def speaker_covariances(h: np.ndarray, speakers: np.ndarray):
    h = np.asarray(h, dtype=np.float64)
    means, Ws, counts = [], [], []
    for s_id in np.unique(speakers):
        m = speakers == s_id
        if m.sum() < 2:
            continue
        means.append(h[m].mean(0))
        Ws.append(np.cov(h[m].T, ddof=1))
        counts.append(int(m.sum()))
    d = h.shape[1]
    if len(means) < 2:
        return np.zeros((d, d)), np.eye(d)
    W = np.mean(Ws, axis=0)
    B = np.cov(np.asarray(means).T, ddof=1) - W / float(np.mean(counts))
    B = (B + B.T) / 2.0
    ev, V = np.linalg.eigh(B)
    B = (V * np.maximum(ev, 0.0)) @ V.T
    return B, (W + W.T) / 2.0


def lambda_matrix(B: np.ndarray, W: np.ndarray, k: int, tol: float = 1e-8) -> np.ndarray:
    d = B.shape[0]
    ev, V = np.linalg.eigh((B + B.T) / 2.0)
    keep = ev > max(float(ev.max()), 0.0) * tol
    if not keep.any():
        return np.zeros((d, d))
    U = V[:, keep]
    Br = U.T @ B @ U
    Wr = U.T @ W @ U
    M = Br + Wr / float(k)
    M = M + 1e-10 * float(np.trace(M)) / M.shape[0] * np.eye(M.shape[0])
    return U @ (Br @ np.linalg.inv(M)) @ U.T


def lambda_k(tau2: float, within: float, k: int) -> float:
    denom = k * tau2 + within
    return float(k * tau2 / denom) if denom > 0 else 1.0


def population_prior(h: np.ndarray, speakers: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    return np.stack([h[speakers == s].mean(0) for s in np.unique(speakers)]).mean(0)


def enrollment_mean(h: np.ndarray, speakers: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    b = np.empty_like(h)
    for s in np.unique(speakers):
        idx = np.where(speakers == s)[0]
        for i in idx:
            others = idx[idx != i]
            if len(others) == 0:
                b[i] = h[i]
            else:
                pick = others if len(others) <= k else rng.choice(others, size=k, replace=False)
                b[i] = h[pick].mean(axis=0)
    return b


@torch.no_grad()
def heads_on(model, delta: np.ndarray, device: str):
    t = torch.tensor(delta, dtype=torch.float32, device=device)
    concepts = model.concept_head(t)
    return concepts.cpu().numpy(), model.emotion_head(concepts).cpu().numpy()


def grid(model, h_test: np.ndarray, speakers: np.ndarray, y: np.ndarray,
         device: str, tau2: float, within: float, seed: int,
         ks: List[int], lams: List[float],
         B: Optional[np.ndarray] = None, W: Optional[np.ndarray] = None) -> List[Dict]:
    out: List[Dict] = []
    for k in ks:
        lstar = lambda_k(tau2, within, k)
        b = enrollment_mean(h_test, speakers, k, seed)
        for lam in sorted(set([round(x, 4) for x in lams] + [round(lstar, 4)])):
            concepts, logits = heads_on(model, h_test - lam * b, device)
            out.append({"k": k, "lam": lam, "lam_star": round(lstar, 4),
                        "is_lam_star": int(abs(lam - round(lstar, 4)) < 1e-9),
                        "is_matrix": 0, "concepts": concepts, "logits": logits})
        if B is not None and W is not None:
            Lam = lambda_matrix(B, W, k)
            concepts, logits = heads_on(model, h_test - b @ Lam.T, device)
            out.append({"k": k, "lam": float(np.trace(Lam) / Lam.shape[0]),
                        "lam_star": round(lstar, 4), "is_lam_star": 0,
                        "is_matrix": 1, "concepts": concepts, "logits": logits})
    return out


def grid_centered(model, h_test: np.ndarray, speakers: np.ndarray, device: str,
                  tau2: float, within: float, seed: int, ks: List[int],
                  prior: np.ndarray, lams: Optional[List[float]] = None) -> List[Dict]:
    extra = [0.0] if lams is None else sorted(set(round(x, 4) for x in lams))
    out: List[Dict] = []
    for k in ks:
        lstar = round(lambda_k(tau2, within, k), 4)
        b = enrollment_mean(h_test, speakers, k, seed)
        for lam, is_star in [(lstar, 1)] + [(x, 0) for x in extra]:
            concepts, logits = heads_on(model, h_test - prior - lam * (b - prior), device)
            out.append({"k": k, "lam": lam, "lam_star": lstar, "is_lam_star": is_star,
                        "concepts": concepts, "logits": logits})
    return out
