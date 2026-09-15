"""Fixed-enrollment analysis, seeds 51-53"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from centered_shrinkage_analysis import sign_flip_p

HERE = os.path.dirname(os.path.abspath(__file__))
CORPORA = ["cremad", "ravdess", "iemocap"]
KS = [1, 2, 5, 10]
HALF = -1
GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
KEY = ["seed", "fold"]
ALPHA = 0.05


def load(root: str, corpus: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(root, corpus, "fixed_enroll", "fixed_pos*_seed*_fold*.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["lam_key"] = np.where(df["is_lam_star"] == 1, "star", df["lam"].round(4).astype(str))
    cols = ["test_uar", "leak_bits", "control_bits", "leak_mlp_bits", "control_mlp_bits",
            "lam", "tau2", "within"]
    return df.groupby(["pos", "protocol", "k", "lam_key"] + KEY, as_index=False)[cols].mean()


def cell(df: pd.DataFrame, pos: int, protocol: str, k: int, lam_key: str) -> pd.DataFrame:
    out = df[(df["pos"] == pos) & (df["protocol"] == protocol) & (df["k"] == k)
             & (df["lam_key"] == lam_key)].set_index(KEY)
    if out.index.duplicated().any():
        raise SystemExit(f"duplicate rows for pos{pos} {protocol} k={k} {lam_key}")
    return out


def paired(a: pd.DataFrame, b: pd.DataFrame, col: str):
    idx = a.index.intersection(b.index)
    d = (a.loc[idx, col] - b.loc[idx, col]).to_numpy(dtype=float)
    if len(d) == 0:
        return float("nan"), float("nan"), 0, 0
    return float(d.mean()), float(sign_flip_p(d)), int((d > 0).sum()), len(d)


def fmt(m, p, n_pos, n) -> str:
    return f"{m:+.4f} (p={p:.2g}, {n_pos}/{n} > 0)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(HERE, "outputs", "position_fixed_enroll"))
    args = ap.parse_args()

    lines: List[str] = [f"# Once-only enrollment on held-out queries\n\nroot: `{os.path.relpath(args.root, HERE)}`\n"]
    verdict: Dict[str, Dict[str, bool]] = {}
    ONE = "1.0"
    for corpus in CORPORA:
        df = load(args.root, corpus)
        if df.empty:
            continue
        v: Dict[str, bool] = {}
        runs = {p: df[df["pos"] == p].groupby(KEY).ngroups for p in (0, 4)}
        lines += [f"## {corpus}", "", f"runs: position 0 {runs[0]}, position 4 {runs[4]}", ""]

        a4, a0 = cell(df, 4, "fixed", HALF, ONE), cell(df, 0, "fixed", HALF, ONE)
        e1 = paired(a4, a0, "leak_bits")
        e2 = paired(a4, a0, "test_uar")
        strict = corpus in ("cremad", "ravdess")
        v["E1"] = e1[0] < 0 and (e1[1] <= ALPHA if strict else True)
        v["E2"] = (e2[0] > 0 and e2[1] <= ALPHA) if strict else True
        lines += ["Placement at half, fixed enrollment (position 4 minus position 0):", "",
                  f"- UAR {a0['test_uar'].mean():.4f} -> {a4['test_uar'].mean():.4f}, diff {fmt(*e2)}",
                  f"- linear leak {a0['leak_bits'].mean():+.3f} -> {a4['leak_bits'].mean():+.3f}, diff {fmt(*e1)}",
                  f"- MLP leak {a0['leak_mlp_bits'].mean():+.3f} -> {a4['leak_mlp_bits'].mean():+.3f}, "
                  f"diff {fmt(*paired(a4, a0, 'leak_mlp_bits'))}",
                  f"- linear control {a0['control_bits'].mean():+.3f} / {a4['control_bits'].mean():+.3f}", ""]

        lines += ["| k | lambda* | UAR lambda* | UAR lambda=1 | dUAR | leak lambda* | leak lambda=1 | dleak | leak lambda=1 resampled | fixed - resampled |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        v["E3"], v["E4"] = True, True
        for k in KS + [HALF]:
            s, one = cell(df, 4, "fixed", k, "star"), cell(df, 4, "fixed", k, ONE)
            du, dl = paired(s, one, "test_uar"), paired(s, one, "leak_bits")
            row = (f"| {'half' if k == HALF else k} | {s['lam'].mean():.3f} | {s['test_uar'].mean():.4f} | "
                   f"{one['test_uar'].mean():.4f} | {fmt(*du)} | {s['leak_bits'].mean():+.3f} | "
                   f"{one['leak_bits'].mean():+.3f} | {fmt(*dl)} |")
            if k in (1, 2):
                v["E3"] &= du[0] > 0 and du[1] <= ALPHA and dl[0] < 0 and dl[1] <= ALPHA
            if k in KS:
                r = cell(df, 4, "resampled", k, ONE)
                dr = paired(one, r, "leak_bits")
                row += f" {r['leak_bits'].mean():+.3f} | {fmt(*dr)} |"
                if k == 1:
                    v["E4"] = dr[0] > 0 and dr[1] <= ALPHA
            else:
                row += " | |"
            lines.append(row)
        lines.append("")

        c4 = df[(df["pos"] == 4) & (df["protocol"] == "fixed")]
        ratio = float((c4["within"] / c4["tau2"]).mean())
        pts, mins = [], []
        for k in KS:
            curve = {}
            for lam in GRID:
                cc = cell(df, 4, "fixed", k, str(round(lam, 4)))
                curve[lam] = cc["leak_bits"].mean()
                pts.append(((1 - lam) ** 2 + lam ** 2 * ratio / k, curve[lam]))
            lstar = cell(df, 4, "fixed", k, "star")["lam"].mean()
            mins.append(f"k={k}: argmin lambda {min(curve, key=curve.get):g}, lambda* {lstar:.3f}, "
                        + " ".join(f"{lam:g}:{curve[lam]:+.3f}" for lam in GRID))
        x, y = np.array(pts).T
        rho = float(pd.Series(x).rank().corr(pd.Series(y).rank()))
        v["E5"] = rho >= 0.90
        lines += [f"E5: W/tau2 = {ratio:.2f}, Spearman rho over {len(pts)} cells = {rho:.3f}", "",
                  "Fixed-enrollment leakage over lambda (position 4):", ""] + [f"- {m}" for m in mins] + [""]

        lines += ["Position 0, lambda = 1: " + ", ".join(
            f"k={'half' if k == HALF else k} UAR {cell(df, 0, 'fixed', k, ONE)['test_uar'].mean():.4f} "
            f"leak {cell(df, 0, 'fixed', k, ONE)['leak_bits'].mean():+.3f}"
            + (f" (resampled {cell(df, 0, 'resampled', k, ONE)['leak_bits'].mean():+.3f})" if k in KS else "")
            for k in KS + [HALF]), ""]
        for p in (0, 4):
            ps = cell(df, p, "pseudo_split", 0, ONE)
            res = os.path.join(args.root, corpus, f"pos{p}_results.csv")
            split = ""
            if os.path.exists(res):
                r = pd.read_csv(res)
                split = (f", split-level on true speakers: UAR {r['test_uar'].mean():.4f}, "
                         f"linear {r['concepts_linear_mi_lb_bits'].mean():+.3f} "
                         f"(control {r['concepts_linear_control_bits'].mean():+.3f}), "
                         f"MLP {r['concepts_mlp_mi_lb_bits'].mean():+.3f}")
            lines.append(f"- position {p} pseudo_split: linear {ps['leak_bits'].mean():+.3f} "
                         f"(its permutation control {ps['control_bits'].mean():+.3f}), "
                         f"MLP {ps['leak_mlp_bits'].mean():+.3f}{split}")
        lines.append("")
        verdict[corpus] = v

    if verdict:
        names = ["E1", "E2", "E3", "E4", "E5"]
        lines += ["## Decision criteria", "", "| corpus | " + " | ".join(names) + " |",
                  "|---" * (len(names) + 1) + "|"]
        for corpus, v in verdict.items():
            lines.append(f"| {corpus} | " + " | ".join(str(v[n]) for n in names) + " |")
        if len(verdict) == len(CORPORA):
            lines += [""] + [f"**{n} met on all corpora: {all(v[n] for v in verdict.values())}.**"
                             for n in names]
        lines.append("")

    md = "\n".join(lines)
    print(md)
    if glob.glob(os.path.join(args.root, "*", "fixed_enroll", "fixed_pos*_seed*_fold*.csv")):
        with open(os.path.join(args.root, "fixed_results.md"), "w") as f:
            f.write(md + "\n")


if __name__ == "__main__":
    main()
