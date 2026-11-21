#!/usr/bin/env python3
"""prior_delivery.basis_audit_second -- the same basis-variance audit on a second,
independent dataset.

The haematopoiesis numbers in `basis_variance_audit.py` answer "how much of this
data's variation does a TRRUST module basis span?". One dataset is enough to raise
the question and not enough to generalise it. This runs the identical audit on
Perturb-CITE-seq in patient-derived melanoma cells under autologous TIL co-culture
(frangieh_2021; the deposited file's uns/doi is 10.1038/s41588-021-00779-1), a
different tissue, different protocol, different perturbation modality and different
gene space, so that the table can be reported as a property of annotation-derived
bases rather than of one dataset.
"""
from __future__ import annotations

import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from basis_variance_audit import shares           # noqa: E402
from modules import build_modules, permute_modules  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default=r"F:\UNAGI_Project\data\public\frangieh_2021.h5ad")
    ap.add_argument("--trrust", default=r"F:\UNAGI_Project\unagi_custom_data\trrust_raw_human.txt")
    ap.add_argument("--p", type=int, default=32)
    ap.add_argument("--max-cells", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=r"F:\UNAGI_Project\prior_delivery_runs\b1\basis_audit_frangieh.json")
    a = ap.parse_args()

    import anndata as ad
    import scipy.sparse as sp
    from sklearn.cluster import KMeans
    from sklearn.decomposition import NMF

    dat = ad.read_h5ad(a.h5ad)
    rng = np.random.default_rng(a.seed)
    if dat.shape[0] > a.max_cells:
        sel = np.sort(rng.choice(dat.shape[0], a.max_cells, replace=False))
        dat = dat[sel].copy()
    X = dat.X.toarray() if sp.issparse(dat.X) else np.asarray(dat.X)
    X = np.asarray(X, dtype=np.float64)
    X -= X.mean(0, keepdims=True)
    tot_all = float((X ** 2).sum())
    genes = list(dat.var_names)

    C, sizes, info = build_modules(genes, a.trrust, p=a.p, seed=a.seed)
    Cp = permute_modules(C, seed=a.seed + 1)
    cov = np.where(np.abs(C).sum(1) > 0)[0]
    print(f"[second] {X.shape[0]} cells x {X.shape[1]} genes; "
          f"TRRUST-covered {cov.size}; p={a.p}", flush=True)

    emb = C[cov]
    lab = KMeans(n_clusters=a.p, n_init=10, random_state=0).fit_predict(emb)
    Cind = np.zeros_like(C)
    for k in range(a.p):
        Cind[cov[lab == k], k] = 1.0

    U, S, Vt = np.linalg.svd(X[:, cov], full_matrices=False)
    Pca = np.zeros_like(C)
    Pca[cov, :] = Vt[:a.p].T

    Xn = np.clip(X[:, cov], 0, None)
    nmf = NMF(n_components=a.p, init="nndsvd", max_iter=300, random_state=0)
    nmf.fit_transform(Xn)
    Nmf = np.zeros_like(C)
    Nmf[cov, :] = nmf.components_.T

    rows = []
    for name, B in (("grn_spectral", C), ("grn_indicator", Cind), ("permuted", Cp),
                    ("pca32", Pca), ("nmf32", Nmf)):
        block, total, ncov, top = shares(np.asarray(B, dtype=np.float64), X, tot_all)
        rows.append(dict(basis=name, n_covered=ncov, block_share=block,
                         total_share=total, **top))
    rows.append(dict(basis="chance", n_covered=int(cov.size),
                     block_share=a.p / cov.size, total_share=float("nan")))

    print(f"\n{'basis':<16}{'n_cov':>7}{'block':>9}{'total':>9}{'top8':>8}{'top32':>8}")
    print("-" * 58)
    for r in sorted(rows, key=lambda r: -r["block_share"]):
        print(f"{r['basis']:<16}{r['n_covered']:7d}{r['block_share']:9.4f}"
              f"{r['total_share']:9.4f}{r.get('top8', float('nan')):8.4f}"
              f"{r.get('top32', float('nan')):8.4f}")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(dict(dataset=os.path.basename(a.h5ad), n_cells=int(X.shape[0]),
                       n_genes=int(X.shape[1]), p=a.p, rows=rows, build=info),
                  f, indent=2)
    print(f"\n[basis] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
