"""Closed-form shrinkage on the first retraining, seeds 45-47"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from centered_shrinkage_analysis import CORPORA, GRID, HERE, estimators, load, sign_flip_p

KS = [1, 2, 5, 10]
TOL_UAR = 0.005
R2_MIN = 0.90
LEAK_TOL = 0.02


def leakage_cells(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    g = data["grid"][data["grid"]["is_matrix"] == 0]
    c = data["centered"]
    rows: List[Tuple[str, int, pd.DataFrame]] = [
        ("h - p", 0, c[np.isclose(c["lam"], 0.0)]),
        ("h", 0, g[np.isclose(g["lam"], 0.0)]),
    ]
    for k in KS:
        gk, ck = g[g["k"] == k], c[c["k"] == k]
        star = gk[gk["is_lam_star"] == 1]
        rows += [("centered lambda*", k, ck[ck["is_lam_star"] == 1]), ("uncentered lambda*", k, star)]
        lstar = float(star["lam"].iloc[0])
        for lam in GRID:
            if lam > 0 and not np.isclose(lam, lstar):
                rows.append((f"grid {lam:g}", k, gk[np.isclose(gk["lam"], lam)]))
    return pd.DataFrame([{"form": f, "k": k, "lam": df["lam"].mean(), "leak": df["leak_bits"].mean(),
                          "n": len(df)} for f, k, df in rows])


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(HERE, "outputs", "position_centered_confirm"))
    ap.add_argument("--reference", default=os.path.join(HERE, "outputs", "position"))
    args = ap.parse_args()

    lines: List[str] = [f"# Closed-form shrinkage\n\nroot: `{os.path.relpath(args.root, HERE)}`\n"]
    verdict: Dict[str, Dict[str, bool]] = {}
    for corpus in CORPORA:
        data = load(args.root, corpus)
        if data["grid"] is None or data["centered"] is None:
            lines.append(f"## {corpus}\n\nmissing grid or centered files\n")
            continue
        seeds = sorted(int(s) for s in data["grid"]["seed"].unique())
        v: Dict[str, bool] = {"C2": True, "C3b": True}
        lines += [f"## {corpus}", "", f"seeds {seeds}, runs {data['centered'].groupby(['seed', 'fold']).ngroups}", "",
                  "| k | lambda* | closed form | unshrunk | diff | p | LOCO lambda | diff | p | h - p | leak closed | leak uncentered lambda* | leak unshrunk |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k in KS:
            e = estimators(data, corpus, k)
            cf, un, pr, uc = e["centered lambda*"], e["unshrunk"], e["centered prior"], e["uncentered lambda*"]
            d_u = (cf["test_uar"] - un.loc[cf.index, "test_uar"]).to_numpy()
            d_leak = cf["leak_bits"].mean() - uc["leak_bits"].mean()
            loco = ["", "", ""]
            if k == 1:
                d_l = (cf["test_uar"] - e["LOCO lambda"].loc[cf.index, "test_uar"]).to_numpy()
                p_l = sign_flip_p(d_l)
                v["C1"] = bool(d_l.mean() > 0 and p_l <= 0.05)
                loco = [f"{e['LOCO lambda']['test_uar'].mean():.4f}", f"{d_l.mean():+.4f}", f"{p_l:.2g}"]
            v["C2"] &= bool(d_u.mean() >= -TOL_UAR)
            v["C3b"] &= bool(d_leak <= LEAK_TOL)
            lines.append(f"| {k} | {cf['lam'].mean():.3f} | {cf['test_uar'].mean():.4f} | {un['test_uar'].mean():.4f} | "
                         f"{d_u.mean():+.4f} | {sign_flip_p(d_u):.2g} | {' | '.join(loco)} | {pr['test_uar'].mean():.4f} | "
                         f"{cf['leak_bits'].mean():.3f} | {uc['leak_bits'].mean():.3f} | {un['leak_bits'].mean():.3f} |")

        cells = leakage_cells(data)
        x = (1 - cells["lam"].to_numpy()) ** 2
        y = cells["leak"].to_numpy()
        A = np.c_[np.ones_like(x), x]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        fit_r2 = r2(y, A @ coef)
        v["C3a"] = bool(fit_r2 >= R2_MIN)
        lines += ["", f"C3a leakage fit over {len(cells)} cells: leak = {coef[0]:+.3f} + {coef[1]:.3f}*(1-lambda)^2, "
                      f"R^2 {fit_r2:.3f}"]

        L0 = cells.loc[cells["form"] == "h - p", "leak"].iloc[0]
        g = data["grid"][data["grid"]["is_matrix"] == 0]
        inner = cells[(cells["k"] > 0) & ~np.isclose(cells["lam"], 1.0)]
        L1 = inner["k"].map(lambda k: g[(g["k"] == k) & np.isclose(g["lam"], 1.0)]["leak_bits"].mean())
        pred = L1 + (L0 - L1) * (1 - inner["lam"]) ** 2
        res = inner["leak"] - pred
        lines += [f"Endpoint prediction over {len(inner)} cells: R^2 {r2(inner['leak'].to_numpy(), pred.to_numpy()):.3f}, "
                  f"mean abs error {res.abs().mean():.3f}, bias {res.mean():+.3f}", ""]
        verdict[corpus] = v

    if len(verdict) == len(CORPORA):
        lines += ["## Decision criteria", "", "| corpus | C1 | C2 | C3a | C3b |", "|---|---|---|---|---|"]
        for corpus, v in verdict.items():
            lines.append(f"| {corpus} | {v['C1']} | {v['C2']} | {v['C3a']} | {v['C3b']} |")
        lines.append("")

    lines += ["## Unshrunk UAR against the original runs", "",
              "| corpus | k | this root | original |", "|---|---|---|---|"]
    for corpus in CORPORA:
        means = []
        for root in (args.root, args.reference):
            gr = load(root, corpus)["grid"]
            means.append({k: estimators({"grid": gr, "centered": None}, corpus, k)["unshrunk"]["test_uar"].mean()
                          for k in (1, 10)} if gr is not None else {})
        for k in (1, 10):
            lines.append(f"| {corpus} | {k} | " + " | ".join(f"{m.get(k, float('nan')):.4f}" for m in means) + " |")
    lines.append("")

    md = "\n".join(lines)
    print(md)
    if glob.glob(os.path.join(args.root, "*", "enroll_grid", "grid_centered_*.csv")):
        with open(os.path.join(args.root, "confirm_results.md"), "w") as f:
            f.write(md + "\n")


if __name__ == "__main__":
    main()
