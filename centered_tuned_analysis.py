"""Closed form against a tuned centered lambda on the second retraining, seeds 48-50"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

from centered_shrinkage_analysis import CORPORA, GRID, HERE, load, sign_flip_p

KS = [1, 2, 5, 10]
D1_TOL = 0.005
D2_TOL = 0.010
KEY = ["seed", "fold"]


def pick(curve: pd.Series) -> float:
    best = curve.max()
    return float(min(lam for lam, v in curve.items() if v >= best - 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(HERE, "outputs", "position_centered_tuned"))
    args = ap.parse_args()

    centered: Dict[str, pd.DataFrame] = {}
    plain: Dict[str, pd.DataFrame] = {}
    for corpus in CORPORA:
        data = load(args.root, corpus)
        if data["centered"] is not None and data["grid"] is not None:
            centered[corpus], plain[corpus] = data["centered"], data["grid"]

    lines: List[str] = [f"# Closed form against a tuned centered lambda\n\nroot: `{os.path.relpath(args.root, HERE)}`\n"]
    curves: Dict[str, Dict[int, pd.Series]] = {}
    for corpus, c in centered.items():
        g = c[(c["is_lam_star"] == 0) & c["lam"].round(4).isin(GRID)]
        n_cells = g.groupby(KEY + ["k", "lam"]).size()
        if n_cells.max() > 1:
            raise SystemExit(f"{corpus}: duplicate centered grid rows")
        curves[corpus] = {k: g[g["k"] == k].groupby("lam")["test_uar"].mean() for k in KS}

        u = plain[corpus]
        u1 = u[(u["is_matrix"] == 0) & np.isclose(u["lam"], 1.0)].set_index(KEY + ["k"])["test_uar"]
        c1 = g[np.isclose(g["lam"], 1.0)].set_index(KEY + ["k"])["test_uar"]
        both = c1.index.intersection(u1.index)
        lines.append(f"{corpus}: centered lambda=1 against unshrunk, max |dUAR| over {len(both)} rows "
                     f"{(c1.loc[both] - u1.loc[both]).abs().max():.2e}")
    lines.append("")

    verdict: Dict[str, Dict[str, bool]] = {}
    for corpus, c in centered.items():
        runs = c.groupby(KEY).ngroups
        lines += [f"## {corpus}", "", f"runs {runs}", "",
                  "| k | lambda* | closed form | LOCO-c lambda | UAR | diff | p | n+ | oracle lambda | UAR | diff |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        v = {"D1": True, "D2": True}
        for k in KS:
            others = [curves[o][k] for o in curves if o != corpus]
            loco_lam = pick(pd.concat(others, axis=1).mean(axis=1)) if others else float("nan")
            oracle_lam = pick(curves[corpus][k])
            ck = c[c["k"] == k]
            cf = ck[ck["is_lam_star"] == 1].set_index(KEY)
            row = f"| {k} | {cf['lam'].mean():.3f} | {cf['test_uar'].mean():.4f} |"
            for lam, tol, name in ((loco_lam, D1_TOL, "D1"), (oracle_lam, D2_TOL, "D2")):
                if np.isnan(lam):
                    row += " n/a | | | |" if name == "D1" else " n/a | | |"
                    v[name] = False
                    continue
                ref = ck[(ck["is_lam_star"] == 0) & np.isclose(ck["lam"], lam)].set_index(KEY)
                d = (cf["test_uar"] - ref.loc[cf.index, "test_uar"]).to_numpy()
                v[name] &= bool(d.mean() >= -tol)
                if name == "D1":
                    row += f" {lam:g} | {ref['test_uar'].mean():.4f} | {d.mean():+.4f} | {sign_flip_p(d):.2g} | {(d > 0).sum()}/{len(d)} |"
                else:
                    row += f" {lam:g} | {ref['test_uar'].mean():.4f} | {d.mean():+.4f} |"
            lines.append(row)
        verdict[corpus] = v

        lines += ["", "Centered UAR over lambda (mean of runs):", "",
                  "| k | " + " | ".join(f"{lam:g}" for lam in GRID) + " |", "|---" * (len(GRID) + 1) + "|"]
        for k in KS:
            cur = curves[corpus][k]
            lines.append(f"| {k} | " + " | ".join(f"{cur.get(lam, float('nan')):.4f}" for lam in GRID) + " |")
        lines.append("")

    if len(verdict) == len(CORPORA):
        lines += ["## Decision criteria", "", "| corpus | D1 | D2 |", "|---|---|---|"]
        for corpus, v in verdict.items():
            lines.append(f"| {corpus} | {v['D1']} | {v['D2']} |")
        d1 = all(v["D1"] for v in verdict.values())
        d2 = all(v["D2"] for v in verdict.values())
        lines += ["", f"**D1 met on all corpora: {d1}. D2 met on all corpora: {d2}.**", ""]

    lines += ["## C1, C2 and C3a on this root", "", "```"]
    confirm = subprocess.run([sys.executable, os.path.join(HERE, "centered_confirm_analysis.py"), "--root", args.root],
                             capture_output=True, text=True)
    lines += [confirm.stdout.strip() or confirm.stderr.strip(), "```", ""]

    md = "\n".join(lines)
    print(md)
    if glob.glob(os.path.join(args.root, "*", "enroll_grid", "grid_centered_*.csv")):
        with open(os.path.join(args.root, "tuned_results.md"), "w") as f:
            f.write(md + "\n")


if __name__ == "__main__":
    main()
