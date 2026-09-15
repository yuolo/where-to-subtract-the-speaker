"""Loaders and estimators shared by the enrollment-grid analyses"""

from __future__ import annotations

import argparse
import glob
import itertools
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CORPORA = ["cremad", "ravdess", "iemocap"]
LOCO_K1 = {"cremad": 0.7, "ravdess": 0.5, "iemocap": 0.7}
GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
KS = [1, 2, 5, 10, 20]
TOL = 0.005


def sign_flip_p(d: np.ndarray) -> float:
    d = np.asarray(d, dtype=float)
    obs = abs(d.sum())
    signs = np.array(list(itertools.product([1.0, -1.0], repeat=len(d))))
    return float(np.mean(np.abs(signs @ d) >= obs - 1e-12))


def load(root: str, corpus: str) -> Dict[str, Optional[pd.DataFrame]]:
    gdir = os.path.join(root, corpus, "enroll_grid")
    plain = sorted(glob.glob(os.path.join(gdir, "grid_seed*_fold*.csv")))
    cent = sorted(glob.glob(os.path.join(gdir, "grid_centered_seed*_fold*.csv")))
    return {
        "grid": pd.concat([pd.read_csv(f) for f in plain], ignore_index=True) if plain else None,
        "centered": pd.concat([pd.read_csv(f) for f in cent], ignore_index=True) if cent else None,
    }


def estimators(data: Dict[str, Optional[pd.DataFrame]], corpus: str, k: int) -> Dict[str, pd.DataFrame]:
    g = data["grid"]
    g = g[(g["k"] == k) & (g["is_matrix"] == 0)]
    key = ["seed", "fold"]
    out = {
        "unshrunk": g[np.isclose(g["lam"], 1.0)],
        "uncentered lambda*": g[g["is_lam_star"] == 1],
    }
    if k == 1:
        out["LOCO lambda"] = g[np.isclose(g["lam"], LOCO_K1[corpus])]
    c = data["centered"]
    if c is not None:
        c = c[c["k"] == k]
        out["centered lambda*"] = c[c["is_lam_star"] == 1]
        out["centered prior"] = c[np.isclose(c["lam"], 0.0)]
    return {name: df.drop_duplicates(key).set_index(key)[["test_uar", "leak_bits", "lam"]]
            for name, df in out.items()}


def loco_from_run(root: str) -> Dict[int, Dict[str, float]]:
    means = {}
    for corpus in CORPORA:
        g = load(root, corpus)["grid"]
        if g is None:
            continue
        g = g[(g["is_matrix"] == 0) & g["lam"].round(4).isin(GRID)]
        means[corpus] = g.groupby(["k", "lam"])["test_uar"].mean()
    picks: Dict[int, Dict[str, float]] = {}
    for k in KS:
        picks[k] = {}
        for held in means:
            others = [means[c].loc[k] for c in means if c != held]
            if others:
                picks[k][held] = float(pd.concat(others, axis=1).mean(axis=1).idxmax())
    return picks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(HERE, "outputs", "position_centered"))
    ap.add_argument("--reference", default=os.path.join(HERE, "outputs", "position"),
                    help="original runs, for the reproducibility check")
    args = ap.parse_args()

    lines: List[str] = [f"# Centered shrinkage test\n\nroot: `{os.path.relpath(args.root, HERE)}`\n"]
    verdict: Dict[str, Dict[str, bool]] = {}
    for corpus in CORPORA:
        data = load(args.root, corpus)
        if data["grid"] is None:
            lines.append(f"## {corpus}\n\nno grid files\n")
            continue
        lines += [f"## {corpus}", "",
                  "| k | estimator | n | UAR | diff vs unshrunk | p | leak bits |", "|---|---|---|---|---|---|---|"]
        verdict[corpus] = {}
        for k in KS:
            est = estimators(data, corpus, k)
            base = est["unshrunk"]
            for name, df in est.items():
                j = base.join(df, lsuffix="_u", how="inner")
                d = (j["test_uar"] - j["test_uar_u"]).to_numpy()
                p = sign_flip_p(d) if name != "unshrunk" and len(d) <= 16 else float("nan")
                lines.append(f"| {k} | {name} | {len(j)} | {df['test_uar'].mean():.4f} | "
                             f"{d.mean():+.4f} | {p:.2g} | {df['leak_bits'].mean():.3f} |")
            if "centered lambda*" in est:
                c = est["centered lambda*"]
                d_u = (c["test_uar"] - base.loc[c.index, "test_uar"]).to_numpy()
                if k == 1:
                    loco = est["LOCO lambda"]
                    d_l = (c["test_uar"] - loco.loc[c.index, "test_uar"]).to_numpy()
                    verdict[corpus]["S1"] = bool(d_u.mean() > 0 and sign_flip_p(d_u) <= 0.05)
                    verdict[corpus]["S2"] = bool(d_l.mean() >= -TOL)
                    lines.append(f"\nk=1 centered lambda* - LOCO lambda: {d_l.mean():+.4f} "
                                 f"(p {sign_flip_p(d_l):.2g})\n")
                    if k == 1:
                        lines += ["| k | estimator | n | UAR | diff vs unshrunk | p | leak bits |",
                                  "|---|---|---|---|---|---|---|"]
                if k in (5, 10, 20):
                    verdict[corpus].setdefault("S3", True)
                    verdict[corpus]["S3"] &= bool(d_u.mean() >= -TOL)
        lines.append("")

    if verdict and all("S1" in v for v in verdict.values()):
        lines += ["## Decision criterion", "", "| corpus | S1 | S2 | S3 |", "|---|---|---|---|"]
        for corpus, v in verdict.items():
            lines.append(f"| {corpus} | {v.get('S1')} | {v.get('S2')} | {v.get('S3')} |")
        ok = len(verdict) == 3 and all(all(v.get(s, False) for s in ("S1", "S2", "S3")) for v in verdict.values())
        lines += ["", f"**Criterion met on all three corpora: {ok}**", ""]

    picks = loco_from_run(args.root)
    if picks:
        lines += ["## LOCO lambda recomputed on this root", "",
                  "| k | " + " | ".join(CORPORA) + " |", "|---" * (len(CORPORA) + 1) + "|"]
        for k, v in picks.items():
            lines.append(f"| {k} | " + " | ".join(f"{v.get(c, float('nan')):.1f}" for c in CORPORA) + " |")
        lines.append("")

    if os.path.abspath(args.root) != os.path.abspath(args.reference):
        lines += ["## Reproducibility against the original runs", "",
                  "| corpus | k | unshrunk UAR new | original | mean |diff| per run |", "|---|---|---|---|---|"]
        for corpus in CORPORA:
            new, ref = load(args.root, corpus)["grid"], load(args.reference, corpus)["grid"]
            if new is None or ref is None:
                continue
            for k in (1, 20):
                a = estimators({"grid": new, "centered": None}, corpus, k)["unshrunk"]
                b = estimators({"grid": ref, "centered": None}, corpus, k)["unshrunk"]
                j = a.join(b, lsuffix="_new", how="inner")
                lines.append(f"| {corpus} | {k} | {a['test_uar'].mean():.4f} | {b['test_uar'].mean():.4f} | "
                             f"{(j['test_uar_new'] - j['test_uar']).abs().mean():.4f} |")
        lines.append("")

    md = "\n".join(lines)
    print(md)
    if os.path.isdir(args.root) and glob.glob(os.path.join(args.root, "*", "enroll_grid", "grid_centered_*.csv")):
        with open(os.path.join(args.root, "centered_shrinkage_results.md"), "w") as f:
            f.write(md + "\n")


if __name__ == "__main__":
    main()
