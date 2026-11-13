#!/usr/bin/env python3
"""prior_delivery.basis_variance_audit -- how much of the data's variation does a
gene-module basis actually span?

This is the zero-noise measurement that explains the B1 null and, more usefully,
puts a number on a premise the structured-decoder literature never reports: when a
method wires its decoder to gene modules, it is confining the model to a subspace.
How much of the expression variance lives in that subspace?

Bases compared at equal rank (p = 32) and, where possible, on the same gene set:

    grn_spectral   Laplacian eigenvectors of the TRRUST co-target graph   (B1's basis)
    grn_indicator  k-means modules on that embedding                      (VEGA-style wiring)
    permuted       grn_spectral with gene labels shuffled                 (content control)
    pca32          top-32 expression PCs over the same 1,999 genes        (data-derived)
    nmf32          NMF components over the same 1,999 genes               (data-derived, nonneg)
    chance         p / n_covered

Reported two ways, because they answer different questions:

    block share   fraction of the variance *within the basis's covered genes*
    total share   fraction of the variance of the whole transcriptome

A basis with small support cannot have a large total share even if it is perfect
inside its block, so both are needed to avoid an unfair comparison.
"""
from __future__ import annotations

import os
import sys
import json
import argparse

import numpy as np


def orth(B):
    Q, _ = np.linalg.qr(B)
    return Q


def shares(B, Xall, tot_all, topk=(8, 16, 32, 64, 128)):
    """Variance captured inside the basis's own covered block, and over the whole
    transcriptome. Each basis is evaluated on *its own* support, because comparing
    a basis on another basis's genes would be measuring the wrong thing."""
    cov = np.where(np.abs(B).sum(1) > 0)[0]
    Xb = Xall[:, cov]
    tot_b = float((Xb ** 2).sum())
    Q = orth(B[cov])
    P = Q @ (Q.T @ Xb.T)                    # (n_cov, cells)
    captured = float((P ** 2).sum())
    block = captured / tot_b
    total = captured / tot_all

    # how much of the *leading* expression directions falls inside span(B)
    U, S, Vt = np.linalg.svd(Xb, full_matrices=False)
    proj = Q @ (Q.T @ Vt.T)                 # (n_cov, n_comp)
    energy = (proj ** 2).sum(0)             # captured per component
    top = {}
    for k in topk:
        denom = float((Vt[:k].T ** 2).sum())
        top[f"top{k}"] = float(energy[:k].sum() / denom) if denom else float("nan")
    return block, total, int(cov.size), top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", default=r"F:\UNAGI_Project\prior_delivery_runs\b1\modules_p32.npz")
    ap.add_argument("--latent", default=r"F:\UNAGI_Project\prior_delivery_runs\b1\probe\s42_real_l0\latent.npz")
    ap.add_argument("--p", type=int, default=32)
    ap.add_argument("--out", default=r"F:\UNAGI_Project\prior_delivery_runs\b1\basis_variance_audit.json")
    a = ap.parse_args()

    mods = np.load(a.modules, allow_pickle=True)
    C, Cp = mods["C"], mods["P"]
    cov = np.where(np.abs(C).sum(1) > 0)[0]

    z = np.load(a.latent, allow_pickle=True)
    Xall = np.asarray(z["X"], dtype=np.float64)
    Xall -= Xall.mean(0, keepdims=True)
    tot_all = float((Xall ** 2).sum())
    Xc = Xall[:, cov]
    tot_cov = float((Xc ** 2).sum())

    from sklearn.cluster import KMeans
    emb = C[cov]
    lab = KMeans(n_clusters=a.p, n_init=10, random_state=0).fit_predict(emb)
    Cind = np.zeros_like(C)
    for k in range(a.p):
        Cind[cov[lab == k], k] = 1.0

    # data-derived, same genes, same rank
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Pca = np.zeros_like(C)
    Pca[cov, :] = Vt[:a.p].T

    from sklearn.decomposition import NMF
    Xn = np.clip(Xc, 0, None)
    nmf = NMF(n_components=a.p, init="nndsvd", max_iter=300, random_state=0)
    W = nmf.fit_transform(Xn)
    Nmf = np.zeros_like(C)
    Nmf[cov, :] = nmf.components_.T

    rows = []
    for name, B in (("grn_spectral", C), ("grn_indicator", Cind), ("permuted", Cp),
                    ("pca32", Pca), ("nmf32", Nmf)):
        block, total, ncov, top = shares(np.asarray(B, dtype=np.float64), Xall, tot_all)
        rows.append(dict(basis=name, p=int(B.shape[1]), n_covered=ncov,
                         block_share=block, total_share=total, **top))

    rows.append(dict(basis="chance", p=a.p, n_covered=int(cov.size),
                     block_share=a.p / cov.size, total_share=a.p / cov.size))


    print(f"X: {Xall.shape[0]} cells x {Xall.shape[1]} genes; "
          f"covered block {cov.size} genes; rank p = {a.p}")
    print(f"\n{'basis':<16} {'n_cov':>6} {'block share':>12} {'total share':>12}")
    print("-" * 50)
    for r in sorted(rows, key=lambda r: -r["block_share"]):
        print(f"{r['basis']:<16} {r['n_covered']:6d} {r['block_share']:12.4f} "
              f"{r['total_share']:12.4f}")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(dict(rows=rows, n_cells=int(Xall.shape[0]),
                       n_genes=int(Xall.shape[1]), p=a.p), f, indent=2)
    print(f"\n[basis] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
