"""Cross-fitted probe lower bounds on I(Z; S) in bits"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LN2 = float(np.log(2.0))
_PROB_EPS = 1e-12


def empirical_entropy_bits(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / LN2)


def _make_probe(kind: str, seed: int):
    if kind == "linear":
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    elif kind == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(128,),
            max_iter=800,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown probe kind {kind!r}")
    return make_pipeline(StandardScaler(), clf)


def _cross_fitted_ce_bits(
    X: np.ndarray,
    y: np.ndarray,
    probe_kind: str,
    n_splits: int,
    seed: int,
) -> float:
    classes = np.unique(y)
    class_to_col = {c: i for i, c in enumerate(classes)}
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    total_nll = 0.0
    n_scored = 0
    for split_idx, (tr, te) in enumerate(splitter.split(X, y)):
        probe = _make_probe(probe_kind, seed=seed + split_idx)
        probe.fit(X[tr], y[tr])
        proba = probe.predict_proba(X[te])
        full = np.full((len(te), len(classes)), _PROB_EPS, dtype=np.float64)
        probe_classes = probe.classes_ if hasattr(probe, "classes_") else probe[-1].classes_
        for j, c in enumerate(probe_classes):
            full[:, class_to_col[c]] = proba[:, j]
        p_true = full[np.arange(len(te)), [class_to_col[c] for c in y[te]]]
        total_nll += float(-np.log(np.clip(p_true, _PROB_EPS, 1.0)).sum())
        n_scored += len(te)

    return total_nll / max(n_scored, 1) / LN2


def mi_lower_bound_bits(
    representations: np.ndarray,
    labels: np.ndarray,
    probe: str = "linear",
    n_splits: int = 5,
    seed: int = 0,
    min_class_count: int = 5,
    permutation_control: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    X = np.asarray(representations, dtype=np.float64)
    y_str = np.asarray(labels).astype(str)

    unique, counts = np.unique(y_str, return_counts=True)
    keep = np.isin(y_str, unique[counts >= max(min_class_count, n_splits)])
    X = X[keep]
    _, y = np.unique(y_str[keep], return_inverse=True)

    n_classes = len(np.unique(y))
    result = {
        "mi_lb_bits": float("nan"),
        "mi_lb_bits_clamped": float("nan"),
        "entropy_bits": float("nan"),
        "ce_bits": float("nan"),
        "control_mi_lb_bits": float("nan"),
        "n_samples": float(len(y)),
        "n_classes": float(n_classes),
    }
    if n_classes < 2 or len(y) < 20:
        return result

    h_bits = empirical_entropy_bits(y)
    ce_bits = _cross_fitted_ce_bits(X, y, probe_kind=probe, n_splits=n_splits, seed=seed)
    bound = h_bits - ce_bits

    result.update(
        mi_lb_bits=float(bound),
        mi_lb_bits_clamped=float(max(bound, 0.0)),
        entropy_bits=h_bits,
        ce_bits=float(ce_bits),
    )

    if permutation_control:
        rng = rng or np.random.default_rng(seed)
        y_perm = rng.permutation(y)
        ce_perm = _cross_fitted_ce_bits(X, y_perm, probe_kind=probe, n_splits=n_splits, seed=seed)
        result["control_mi_lb_bits"] = float(empirical_entropy_bits(y_perm) - ce_perm)

    return result


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_spk, per_spk, dim = 12, 80, 16
    means = rng.normal(0.0, 2.0, size=(n_spk, dim))
    X = np.concatenate([means[s] + rng.normal(0, 1, size=(per_spk, dim)) for s in range(n_spk)])
    y = np.repeat([f"spk{s:02d}" for s in range(n_spk)], per_spk)

    for probe in ("linear", "mlp"):
        r = mi_lower_bound_bits(X, y, probe=probe, seed=0)
        print(
            f"[signal/{probe}] H(S)={r['entropy_bits']:.3f}b  "
            f"I_LB={r['mi_lb_bits']:.3f}b  control={r['control_mi_lb_bits']:+.3f}b"
        )
        assert r["mi_lb_bits"] > 1.0, "expected a clearly positive bound on signal data"
        assert r["mi_lb_bits"] <= r["entropy_bits"] + 1e-9
        assert abs(r["control_mi_lb_bits"]) < 0.35, "permutation control should be near zero"

    X_noise = rng.normal(size=X.shape)
    r = mi_lower_bound_bits(X_noise, y, probe="linear", seed=0)
    print(f"[noise/linear] I_LB={r['mi_lb_bits']:+.3f}b (should be ~<= 0)")
    assert r["mi_lb_bits"] < 0.15
    print("mi_estimation self-test OK")
