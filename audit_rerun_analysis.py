"""Masking-matched fine-tuning and rank-aware INLP reruns against the released runs"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from centered_shrinkage_analysis import sign_flip_p

HERE = os.path.dirname(os.path.abspath(__file__))
CORPORA = [("cremad", "CREMA-D"), ("ravdess", "RAVDESS"), ("iemocap", "IEMOCAP")]
KEY = ["seed", "fold"]
CMN = "finetune/{c}/ft_cmn_nostyle_gpu2_results.csv"
OPERATOR = "finetune/{c}/ft_operator_gpu2_results.csv"
POSTHOC = "posthoc/posthoc_results.csv"
VARIANTS = ["plain", "subtract", "leace_trans", "leace_trainfit", "inlp_trans"]


def arm(root: str, rel: str, corpus: str) -> pd.DataFrame:
    path = os.path.join(root, rel.format(c=corpus))
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path).groupby(KEY).mean(numeric_only=True)


def finetune_gap(root: str, corpus: str) -> Optional[Dict]:
    a, b = arm(root, CMN, corpus), arm(root, OPERATOR, corpus)
    if a.empty or b.empty:
        return None
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    leak = (b["concepts_linear_mi_lb_bits"] - a["c_aff_linear_mi_lb_bits"]).to_numpy(dtype=float)
    uar = (b["test_uar"] - a["test_uar"]).to_numpy(dtype=float)
    with np.errstate(all="ignore"):  # some BLAS builds warn on the sign-flip matmul
        leak_p = sign_flip_p(leak)
    return {
        "n": len(idx),
        "cmn_uar": a["test_uar"].mean(),
        "operator_uar": b["test_uar"].mean(),
        "cmn_leak": a["c_aff_linear_mi_lb_bits"].mean(),
        "operator_leak": b["concepts_linear_mi_lb_bits"].mean(),
        "leak_gap": leak.mean(),
        "leak_p": leak_p,
        "leak_lower": int((leak < 0).sum()),
        "uar_gap": uar.mean(),
    }


def posthoc(root: str) -> Optional[pd.DataFrame]:
    path = os.path.join(root, POSTHOC)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df.groupby(["dataset", "variant"])[["test_uar", "h_linear_mi_lb_bits"]].mean()


def signed(x: float, digits: int = 3) -> str:
    return ("-" if x < 0 else "+") + f"{abs(x):.{digits}f}".lstrip("0")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--released", default=os.path.join(HERE, "outputs"))
    p.add_argument("--reruns", default=os.path.join(HERE, "audit_reruns"))
    args = p.parse_args()

    print("Fine-tuned rows of Table 1, input-CMN against the shift operator.")
    print("Released runs mask only the operator arm. The reruns mask both arms.\n")
    head = (f"{'Corpus':8s} {'Runs':9s} {'n':>3s} {'CMN UAR/leak':>19s} "
            f"{'Operator UAR/leak':>19s} {'Gap':>7s} {'p':>7s}")
    print(head)
    print("-" * len(head))
    for corpus, name in CORPORA:
        for label, root in (("released", args.released), ("rerun", args.reruns)):
            g = finetune_gap(root, corpus)
            if g is None:
                print(f"{name:8s} {label:9s}   not present")
                continue
            print(f"{name:8s} {label:9s} {g['n']:3d} "
                  f"{g['cmn_uar']:8.3f} / {signed(g['cmn_leak']):>8s} "
                  f"{g['operator_uar']:8.3f} / {signed(g['operator_leak']):>8s} "
                  f"{signed(g['leak_gap']):>7s} {g['leak_p']:7.2g}")
        print()

    print("Sign of the leakage gap once the masking is matched:")
    for corpus, name in CORPORA:
        old, new = finetune_gap(args.released, corpus), finetune_gap(args.reruns, corpus)
        if old is None or new is None:
            continue
        kept = np.sign(old["leak_gap"]) == np.sign(new["leak_gap"])
        print(f"  {name:8s} {signed(old['leak_gap'])} -> {signed(new['leak_gap'])}, "
              f"sign kept {str(kept).lower()}, operator lower in "
              f"{new['leak_lower']}/{new['n']} runs, UAR gap "
              f"{signed(old['uar_gap'])} -> {signed(new['uar_gap'])}")

    old, new = posthoc(args.released), posthoc(args.reruns)
    if old is None or new is None:
        return
    print("\nPost-hoc baselines, released QR against rank-truncated INLP.")
    print("The rerun retrains from scratch, so the other variants bound the run-to-run spread.\n")
    print(f"{'Corpus':8s} {'Variant':15s} {'UAR':>16s} {'Leakage (bits)':>22s}")
    print("-" * 64)
    for corpus, name in CORPORA:
        for variant in VARIANTS:
            if (corpus, variant) not in old.index or (corpus, variant) not in new.index:
                continue
            a, b = old.loc[(corpus, variant)], new.loc[(corpus, variant)]
            print(f"{name:8s} {variant:15s} "
                  f"{a['test_uar']:7.3f} -> {b['test_uar']:6.3f} "
                  f"{a['h_linear_mi_lb_bits']:11.3f} -> {b['h_linear_mi_lb_bits']:8.3f}")
        print()


if __name__ == "__main__":
    main()
