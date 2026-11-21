#!/usr/bin/env python3
"""prior_delivery.modules_expr -- the same operator, a basis taken from the data.

The annotation basis captures 5.3% of the covered block's variance; a same-rank basis
fitted to the same block captures 60.7% (B1_DESIGN §7.8). Every result so far used the
annotation basis. The open question is whether the constraint fails because of the
operator or because of what the operator was pointed at.

This builds a drop-in replacement basis: same rows (the 1,999 TRRUST-covered genes), same
rank (32), same file format, but the subspace is the top-32 principal directions of the
expression block rather than the TRRUST co-target modules. The control keeps the identical
values and reassigns them to random genes within the same covered set, so support and rank
are held fixed and only the identity of the directions changes.

To avoid fitting the basis on the cells that will be used to score it, the PCA is fitted on
a random half of the cells (seed 42) and evaluated on the other half.

    py -3.13 -u modules_expr.py
"""
from __future__ import annotations

import json
import os

import numpy as np

ANN = r"F:\UNAGI_Project\prior_delivery_runs\b1\modules_p32.npz"
LATENT = r"F:\UNAGI_Project\prior_delivery_runs\dose20_e45\s42\real\latent.npz"
OUT = r"F:\UNAGI_Project\prior_delivery_runs\b1\modules_expr_p32.npz"
OUT_JSON = r"F:\UNAGI_Project\prior_delivery_runs\b1\modules_expr_p32.json"


def share(J, Q):
    P = Q @ (Q.T @ J)
    return float((P ** 2).sum() / (J ** 2).sum())


def main():
    ann = np.load(ANN, allow_pickle=True)
    C_ann, P_ann = ann["C"], ann["P"]
    cov = np.where(np.abs(C_ann).sum(1) > 0)[0]
    print(f"covered genes {cov.size}, rank {C_ann.shape[1]}")

    z = np.load(LATENT, allow_pickle=True)
    X = np.asarray(z["X"], dtype=np.float64)[:, cov]
    n = X.shape[0]
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    fit, held = idx[: n // 2], idx[n // 2:]

    # No per-gene standardisation: the variance audit of §7.8 (60.7% for PCA-32) is
    # computed on the unstandardised block, and the decoder's Jacobian magnitude is also
    # per-gene scale dependent. Matching that convention is what makes the two bases
    # comparable at all.
    Xf = X[fit]
    mu = Xf.mean(0, keepdims=True)
    Xf = Xf - mu
    _, S, Vt = np.linalg.svd(Xf, full_matrices=False)
    V = Vt[:32].T                                  # (covered, 32), orthonormal columns
    print(f"top-32 singular values: {S[:4].round(1)} ... {S[31]:.1f}")

    # 231 of the 1,999 TRRUST-covered genes are all-zero in this dataset, so their PCA
    # loadings are exactly zero. If the control permuted rows over all 1,999, the two arms
    # would end up constrained on different gene sets -- a matched-support design would
    # silently stop being matched. Restrict both bases to the rows that actually vary.
    nz = np.abs(V).sum(1) > 0
    cov = cov[nz]
    V = V[nz]
    print(f"genes with non-zero loadings: {cov.size} "
          f"({int((~nz).sum())} of the covered genes are constant in this dataset)")

    C = np.zeros_like(C_ann)
    C[cov, :] = V
    # content control: identical values, reassigned to random genes in the same set
    P = np.zeros_like(C_ann)
    P[cov[rng.permutation(cov.size)], :] = V

    # re-slice to the reduced cover set and re-centre on the fit half only
    X = np.asarray(z["X"], dtype=np.float64)[:, cov]
    Xh = X[held] - X[fit].mean(0, keepdims=True)
    rep = dict(
        covered=int(cov.size), constant_in_dataset=int((~nz).sum()), rank=32,
        var_share_expr=share(Xh.T, np.linalg.qr(V)[0]),
        var_share_perm=share(Xh.T, np.linalg.qr(P[cov])[0]),
        var_share_annotation=share(Xh.T, np.linalg.qr(C_ann[cov])[0]),
        var_share_annotation_perm=share(Xh.T, np.linalg.qr(P_ann[cov])[0]),
        support_C=int((np.abs(C).sum(1) > 0).sum()),
        support_P=int((np.abs(P).sum(1) > 0).sum()),
        chance=32 / cov.size,
        fit_cells=int(fit.size), held_cells=int(held.size),
    )
    cos = np.linalg.svd(np.linalg.qr(V)[0].T @ np.linalg.qr(C_ann[cov])[0],
                        compute_uv=False)
    rep["cos_expr_vs_annotation_max"] = float(cos.max())
    rep["cos_expr_vs_annotation_mean"] = float(cos.mean())
    print(json.dumps(rep, indent=1))

    np.savez_compressed(OUT, C=C, P=P, cov=cov,
                        kind=np.array(["expr_pca32"]))
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
