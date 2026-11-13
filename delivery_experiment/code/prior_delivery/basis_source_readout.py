#!/usr/bin/env python3
"""prior_delivery.basis_source_readout -- does the basis source change the readout?

The geometry experiment says content determines which subspace the Jacobian occupies. It
does not say that this changes what the decoder can predict. The basis audit predicts it
should not for the annotation basis (5.3% of block variance) and should, if anything, for
an expression-fitted basis (57.9%). This script scores the four arms on the same readout,
with the same folds, so the comparison is purely between bases.

    py -3.13 -u basis_source_readout.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cv_nonlinear_null import load, standardise, ridge_oof  # noqa: E402

ARMS = {
    "annot/real_1": r"F:\UNAGI_Project\prior_delivery_runs\b1\ganoff\s42\real_a",
    "annot/real_2": r"F:\UNAGI_Project\prior_delivery_runs\b1\ganoff\s42\real_b",
    "annot/perm": r"F:\UNAGI_Project\prior_delivery_runs\b1\ganoff\s42\perm",
    "expr/real": r"F:\UNAGI_Project\prior_delivery_runs\b1\expr2\s42\real",
    "expr/perm": r"F:\UNAGI_Project\prior_delivery_runs\b1\expr2\s42\perm",
}
COV = np.load(r"F:\UNAGI_Project\prior_delivery_runs\b1\modules_p32.npz",
              allow_pickle=True)
OUT = r"F:\UNAGI_Project\prior_delivery_runs\b1\basis_source_readout.json"


def main():
    cov = np.where(np.abs(COV["C"]).sum(1) > 0)[0]
    res = {}
    for name, d in ARMS.items():
        if not os.path.exists(os.path.join(d, "latent.npz")):
            print(f"{name:14s} pending")
            continue
        Z, X = load(d)
        Z, X = standardise(Z), standardise(X)
        r2 = ridge_oof(Z, X)
        res[name] = dict(mean_cov=float(np.nanmean(r2[cov])),
                         mean_all=float(np.nanmean(r2)),
                         sum=float(np.nansum(r2)))
        print(f"{name:14s} covered {res[name]['mean_cov']:.5f} "
              f"all {res[name]['mean_all']:.5f} sum {res[name]['sum']:.1f}", flush=True)

    if {"expr/real", "annot/real_1"} <= set(res):
        a = res["expr/real"]["mean_cov"]
        for other in ("annot/real_1", "annot/real_2"):
            if other in res:
                print(f"expr/real - {other}: {a - res[other]['mean_cov']:+.5f} "
                      f"on the covered block")
        if "expr/perm" in res:
            print(f"expr interaction (real-perm): "
                  f"{a - res['expr/perm']['mean_cov']:+.5f}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
