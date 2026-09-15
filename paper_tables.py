"""Recomputes the numbers reported in the paper from the CSV files in outputs/"""

from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
CORPORA = [("cremad", "CREMA-D"), ("ravdess", "RAVDESS"), ("iemocap", "IEMOCAP")]
CHANCE = {"cremad": 1 / 6, "ravdess": 1 / 8, "iemocap": 1 / 4}
KEY = ["seed", "fold"]

TABLE1 = [
    ("Trained from scratch (CRNN)", "floor_sweep/{c}/cmn_results_nostyle.csv", "c_aff_rep",
     "operator/{c}/operator_results.csv", "concepts"),
    ("Fine-tuned at 1e-5", "finetune/{c}/ft_cmn_nostyle_gpu2_results.csv", "c_aff",
     "finetune/{c}/ft_operator_gpu2_results.csv", "concepts"),
    ("Frozen features, 2-layer MLP", "ssl/{c}/ssl_local_cmn_nostyle_results.csv", "c_aff",
     "ssl/{c}/ssl_local_operator_results.csv", "concepts"),
]


def sign_flip_p(d) -> float:
    d = np.asarray(d, dtype=float)
    signs = np.array(list(itertools.product([1.0, -1.0], repeat=len(d))))
    return float(np.mean(np.abs((signs * d).mean(axis=1)) >= abs(d.mean()) - 1e-12))


def runs(rel: str, corpus: str, clean: bool = False) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(OUT, rel.format(c=corpus)))
    if clean:
        df = df[df["test_uar"] >= 1.5 * CHANCE[corpus]]
    return df


def indexed(rel: str, corpus: str) -> pd.DataFrame:
    df = runs(rel, corpus)
    assert not df.duplicated(KEY).any(), rel
    return df.set_index(KEY)


def fixed_runs(corpus: str) -> pd.DataFrame:
    root = os.path.join(OUT, "position_fixed_enroll", corpus, "fixed_enroll")
    return pd.concat([pd.read_csv(os.path.join(root, f)) for f in sorted(os.listdir(root))], ignore_index=True)


def num(x: float, digits: int = 3) -> str:
    return f"{x:+.{digits}f}".replace("0.", ".", 1)


def section(title: str) -> None:
    print(f"\n== {title}")


def probe_agreement() -> None:
    section("Sec. 2.1 linear against MLP probe over the sweep-and-grid runs")
    sets = [("floor_sweep/{c}/floor_results.csv", "c_aff_rep"), ("ssl/{c}/ssl_floor_results.csv", "c_aff"),
            ("operator/{c}/operator_results.csv", "concepts"), ("ssl/{c}/ssl_operator_results.csv", "concepts")]
    x, y = [], []
    for rel, pre in sets:
        for corpus, _ in CORPORA:
            df = runs(rel, corpus)
            x.append(df[f"{pre}_linear_mi_lb_bits"].to_numpy())
            y.append(df[f"{pre}_mlp_mi_lb_bits"].to_numpy())
    x, y = np.concatenate(x), np.concatenate(y)
    print(f"  Pearson r = {np.corrcoef(x, y)[0, 1]:.3f} over {len(x)} runs")


def table1() -> None:
    section("Table 1, UAR / linear I_LB(c;S), shift operator minus input-CMN, with the MLP-probe gap")
    for regime, cmn_rel, cp, op_rel, op in TABLE1:
        for corpus, name in CORPORA:
            a, b = indexed(cmn_rel, corpus), indexed(op_rel, corpus)
            idx = a.index.intersection(b.index)
            a, b = a.loc[idx], b.loc[idx]
            gap = b[f"{op}_linear_mi_lb_bits"] - a[f"{cp}_linear_mi_lb_bits"]
            mlp = b[f"{op}_mlp_mi_lb_bits"] - a[f"{cp}_mlp_mi_lb_bits"]
            print(f"  {regime:29s} {name:8s} {a['test_uar'].mean():.3f} / {num(a[f'{cp}_linear_mi_lb_bits'].mean())}"
                  f"  {b['test_uar'].mean():.3f} / {num(b[f'{op}_linear_mi_lb_bits'].mean())}"
                  f"  gap {num(gap.mean())} p={sign_flip_p(gap):.2g} n={len(idx)}"
                  f"  | MLP gap {num(mlp.mean())} p={sign_flip_p(mlp):.2g} lower in {(mlp < 0).sum()}/{len(mlp)}")
            if not regime.startswith("Frozen"):
                for label, d in (("leakage", gap), ("UAR", b["test_uar"] - a["test_uar"])):
                    folds = d.groupby(level="fold").mean()
                    t, p = ttest_rel(folds.to_numpy(), np.zeros(len(folds)))
                    print(f"  {'':38s} fold-level {label:7s} {num(folds.mean())} t={t:.1f} p={p:.2g}, "
                          f"lower on {(folds < 0).sum()}/{len(folds)}, higher on {(folds > 0).sum()}/{len(folds)} folds")


def crnn_details() -> None:
    section("Sec. 3.3 CRNN: affine variant, and operator runs outside the region the pressure sweep reaches")
    for corpus, name in CORPORA:
        cmn = runs("floor_sweep/{c}/cmn_results_nostyle.csv", corpus)
        aff = runs("operator/{c}/affine_results.csv", corpus)
        sweep = runs("floor_sweep/{c}/floor_results.csv", corpus)
        op = runs("operator/{c}/operator_results.csv", corpus)
        u, l = sweep["test_uar"].to_numpy(), sweep["c_aff_rep_linear_mi_lb_bits"].to_numpy()
        outside = sum(not np.any((u >= a) & (l <= b))
                      for a, b in zip(op["test_uar"], op["concepts_linear_mi_lb_bits"]))
        print(f"  {name:8s} affine {aff['test_uar'].mean():.3f} / {num(aff['concepts_linear_mi_lb_bits'].mean())}"
              f" against input-CMN {cmn['test_uar'].mean():.3f} / {num(cmn['c_aff_rep_linear_mi_lb_bits'].mean())},"
              f" operator runs not dominated by any sweep run {outside}/{len(op)}")


def fixed_placement() -> None:
    section("Sec. 3.3 fixed enrollment, half of each test speaker enrolls, shift operator minus input-CMN")
    for corpus, name in CORPORA:
        df = fixed_runs(corpus)
        df = df[(df["protocol"] == "fixed") & (df["k"] == -1) & (df["is_lam_star"] == 0) & (df["lam"] == 1.0)]
        arm = {pos: df[df["pos"] == pos].groupby(KEY)[["test_uar", "leak_bits", "leak_mlp_bits"]].mean()
               for pos in (0, 4)}
        idx = arm[0].index.intersection(arm[4].index)
        for col, label in (("leak_bits", "linear leakage"), ("test_uar", "UAR"), ("leak_mlp_bits", "MLP leakage")):
            d = arm[4].loc[idx, col] - arm[0].loc[idx, col]
            folds = d.groupby(level="fold").mean()
            better = (folds > 0).sum() if col == "test_uar" else (folds < 0).sum()
            print(f"  {name:8s} {label:15s} {num(d.mean())} p={sign_flip_p(d):.2g} n={len(d)}, "
                  f"operator better on {better}/{len(folds)} fold means")


def depth_sweep() -> None:
    section("Sec. 3.3 subtraction depth, positions 0-4, means over 15 runs")
    for corpus, name in CORPORA:
        pos = [runs(f"position/{{c}}/pos{L}_results.csv", corpus) for L in range(5)]
        print(f"  {name:8s} leakage " + " ".join(num(p["concepts_linear_mi_lb_bits"].mean()) for p in pos)
              + "   UAR " + " ".join(f"{p['test_uar'].mean():.3f}" for p in pos))


def lr_sweep() -> None:
    section("Sec. 3.3 encoder learning rate, CREMA-D, input-CMN under 12 transformer layers")
    arms = {lr: indexed(f"finetune/{{c}}/ft_cmn_nostyle_lr{lr}_results.csv", "cremad") for lr in ("0", "1e-6", "1e-5")}
    for lr, df in arms.items():
        print(f"  lr={lr:5s} UAR {df['test_uar'].mean():.3f}, leakage {num(df['c_aff_linear_mi_lb_bits'].mean())}")
    d = arms["1e-5"]["c_aff_linear_mi_lb_bits"] - arms["1e-6"]["c_aff_linear_mi_lb_bits"]
    print(f"  1e-6 -> 1e-5: {num(d.mean())}, falls in {(d < 0).sum()}/{len(d)} pairs, p={sign_flip_p(d):.2g}")


def finetune_details() -> None:
    section("Sec. 3.3 after fine-tuning")
    h = {arm: runs(f"finetune/{{c}}/ft_{arm}_results.csv", "cremad")
         for arm in ("plain", "factor", "cmn_nostyle_gpu2", "operator_gpu2")}
    print("  CREMA-D linear probe on h, mean per arm: "
          + ", ".join(f"{arm} {df['h_linear_mi_lb_bits'].mean():.2f}" for arm, df in h.items())
          + f", of H(S) = {h['plain']['h_linear_entropy_bits'].mean():.2f} bits")
    print(f"  CREMA-D MLP probe on delta, operator: {h['operator_gpu2']['delta_mlp_mi_lb_bits'].mean():.2f} bits")
    op = runs("finetune/{c}/ft_operator_gpu2_results.csv", "iemocap")
    print("  IEMOCAP operator leakage per fold: "
          + ", ".join(f"fold {f} {v:.3f}" for f, v in op.groupby("fold")["concepts_linear_mi_lb_bits"].mean().items()))
    for corpus, name in CORPORA:
        op = runs("finetune/{c}/ft_operator_gpu2_results.csv", corpus)
        print(f"  {name:8s} operator code above control, linear "
              f"{num((op['concepts_linear_mi_lb_bits'] - op['concepts_linear_control_bits']).mean())}, MLP "
              f"{num((op['concepts_mlp_mi_lb_bits'] - op['concepts_mlp_control_bits']).mean())}")


def pressure_and_erasure() -> None:
    section("Sec. 3.4 CREMA-D/CRNN pressure sweep, runs above 1.5 times chance")
    sweep = runs("floor_sweep/{c}/floor_results.csv", "cremad", clean=True)
    for kappa, g in sweep.groupby("kappa"):
        print(f"  kappa={kappa:<4g} n={len(g):2d} UAR {g['test_uar'].mean():.3f}, "
              f"leakage {g['c_aff_rep_linear_mi_lb_bits'].mean():.3f}")
    section("Sec. 3.4 wav2vec 2.0 sweeps, lowest mean leakage over kappa, and the permutation control")
    for corpus, name in CORPORA:
        cells = []
        for arm, label in (("floor", "joint"), ("floor_armadv", "adversary-only"), ("floor_armorth", "orthogonality-only")):
            g = runs(f"ssl/{{c}}/ssl_{arm}_results.csv", corpus, clean=True).groupby("kappa").mean(numeric_only=True)
            cells.append(f"{label} {g['c_aff_linear_mi_lb_bits'].min():.3f} (control {num(g['c_aff_linear_control_bits'].mean())})")
        print(f"  {name:8s} " + ", ".join(cells))
    section("Sec. 3.4 post-hoc routes on the trained plain wav2vec 2.0 encoder, UAR / linear I_LB(h;S)")
    post = pd.read_csv(os.path.join(OUT, "posthoc", "posthoc_results.csv"))
    means = post.groupby(["dataset", "variant"])[["test_uar", "h_linear_mi_lb_bits", "h_linear_control_bits"]].mean()
    for corpus, name in CORPORA:
        print(f"  {name:8s} " + ", ".join(
            f"{v} {means.loc[(corpus, v), 'test_uar']:.3f} / {num(means.loc[(corpus, v), 'h_linear_mi_lb_bits'], 2)}"
            for v in ("plain", "subtract", "leace_trans", "leace_trainfit", "inlp_trans"))
            + f" (control {num(means.loc[corpus, 'h_linear_control_bits'].mean(), 2)})")


def enrollment() -> None:
    section("Table 2, shift-operator UAR with k = 1, 2, 5, 10 redrawn enrollment utterances and at split level")
    for sub, rel in (("CRNN", "operator/{c}/operator_results.csv"), ("wav2vec 2.0", "ssl/{c}/ssl_operator_results.csv")):
        for corpus, name in CORPORA:
            df = runs(rel, corpus)
            ks = (1, 2, 5, 10)
            print(f"  {sub:11s} {name:8s} UAR " + " ".join(f"{df[f'enrollk{k}_test_uar'].mean():.3f}" for k in ks)
                  + f" {df['test_uar'].mean():.3f}   leakage "
                  + " ".join(num(df[f"enrollk{k}_concepts_mi_lb_bits"].mean()) for k in ks)
                  + f" (control {num(df['concepts_linear_control_bits'].mean())})")
    factor = runs("floor_sweep/{c}/floor_results.csv", "cremad", clean=True).groupby("kappa")["test_uar"].mean()
    print(f"  best factorized CREMA-D model: UAR {factor.max():.3f} at kappa={factor.idxmax():g}")


def table3() -> None:
    section("Table 3, fixed enrollment at k=1, CRNN operator, UAR / I_LB(c;S) on the held-out half")
    for corpus, name in CORPORA:
        df = fixed_runs(corpus)
        df = df[(df["pos"] == 4) & (df["protocol"] == "fixed") & (df["k"] == 1)]
        cells = {"h - b_1": df[(df["is_lam_star"] == 0) & (df["lam"] == 1.0)],
                 "h - mu_0": df[(df["is_lam_star"] == 0) & (df["lam"] == 0.0)],
                 "closed form": df[df["is_lam_star"] == 1]}
        out = []
        for label, cell in cells.items():
            per_run = cell.groupby(KEY)[["test_uar", "leak_bits"]].mean()
            assert len(per_run) == 15, (corpus, label)
            out.append(f"{label} {per_run['test_uar'].mean():.3f} / {per_run['leak_bits'].mean():.3f}")
        print(f"  {name:8s} " + ", ".join(out))


def concepts() -> None:
    section("Sec. 3.5 concept interventions, shift and affine operators on both substrates")
    rels = ["operator/{c}/operator_results.csv", "operator/{c}/affine_results.csv",
            "ssl/{c}/ssl_operator_results.csv", "ssl/{c}/ssl_affine_results.csv"]
    frames = [runs(r, c) for r in rels for c, _ in CORPORA]
    swap = [f["concept_swap_sensitivity"].mean() for f in frames]
    keep = [f["enroll_resample_consistency"].mean() for f in frames]
    print(f"  swapping concepts changes {min(swap):.1%}-{max(swap):.1%} of predictions, "
          f"re-estimating the baseline from disjoint halves keeps {min(keep):.1%}-{max(keep):.1%}")


if __name__ == "__main__":
    probe_agreement()
    table1()
    crnn_details()
    fixed_placement()
    depth_sweep()
    lr_sweep()
    finetune_details()
    pressure_and_erasure()
    enrollment()
    table3()
    concepts()
