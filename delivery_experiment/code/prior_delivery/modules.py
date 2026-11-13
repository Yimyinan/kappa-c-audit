#!/usr/bin/env python3
"""prior_delivery.modules -- the gene-module basis C for a geometry prior (B1).

The B1 operator is a constraint on the *decoder Jacobian*, not on the likelihood:
every latent axis must move expression along a combination of gene modules rather
than an arbitrary gene-space direction. That is the VAE realisation of GEDI's
`z_k ~ N(C a_k, ...)` prior, with C the gene-level prior matrix.

C is built from TRRUST by spectral clustering of the gene co-target graph:

    genes i, j are linked if some TF targets both
    L = D - A  (normalised), take the P smallest eigenvectors, k-means on rows

The permutation control hands the same module *sizes* to random gene sets, so
real and control have identical rank and identical column sparsity. Any difference
between them is the content of the GRN, not the dimension of the constraint.

Why P = 32 and not more: UNAGI's latent is 64-dimensional. A 32-dimensional gene
subspace constrains half of the directions. P close to 64 would make the
constraint vacuous; P far below 32 would make it trivially destructive.
"""
from __future__ import annotations

import os
import sys
import json
import argparse

import numpy as np


def _trrust_edges(path, genes_upper):
    import pandas as pd

    tr = pd.read_csv(path, sep="\t", header=None, comment="#",
                     names=["TF", "target", "dir", "pmid"])
    tr["t"] = tr["target"].astype(str).str.upper()
    tr["f"] = tr["TF"].astype(str).str.upper()
    tr = tr[tr["t"].isin(genes_upper)]
    return tr


def build_modules(gene_names, trrust_path, p=32, seed=0, min_module=20):
    """Spectral embedding of the gene co-target graph.

    The basis is NOT the cluster indicators: an indicator basis forces every latent
    axis to move a module's genes with one sign and equal magnitude, which no
    biological program does. The spectral embedding (normalised-Laplacian
    eigenvectors 2..p+1, signed) spans a richer subspace with the same dimension
    and, more importantly, admits a clean permutation control: permuting rows keeps
    every column's values and norm but destroys which gene carries which loading.
    """
    from sklearn.preprocessing import normalize
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    genes = [str(g).upper() for g in gene_names]
    idx = {g: i for i, g in enumerate(genes)}
    tr = _trrust_edges(trrust_path, set(idx))

    rows = tr["t"].map(idx).to_numpy()
    tfs = tr["f"].astype("category").cat.codes.to_numpy()
    n_tf = int(tfs.max()) + 1
    B = sp.coo_matrix((np.ones(rows.size), (rows, tfs)),
                      shape=(len(genes), n_tf)).tocsr()
    B.data[:] = 1.0
    A = (B @ B.T).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()

    deg = np.asarray(A.sum(1)).ravel()
    keep = deg > 0
    n_keep = int(keep.sum())
    if n_keep < p * min_module:
        raise SystemExit(f"[modules] only {n_keep} genes have any TRRUST edge; "
                         f"cannot build {p} dimensions")

    sub = A[keep][:, keep]
    d = np.asarray(sub.sum(1)).ravel()
    Dm = sp.diags(1.0 / np.sqrt(np.maximum(d, 1e-12)))
    Lsym = sp.eye(sub.shape[0]) - Dm @ sub @ Dm
    vals, vecs = eigsh(Lsym.astype(np.float64), k=p + 1, which="SM")
    order = np.argsort(vals)
    emb = normalize(vecs[:, order[1:p + 1]])          # (n_keep, p), orthonormal

    C = np.zeros((len(genes), p), dtype=np.float64)
    C[np.where(keep)[0], :] = emb

    # module sizes for reference: k-means labels on the embedding
    from sklearn.cluster import KMeans
    lab = KMeans(n_clusters=p, n_init=10, random_state=seed).fit_predict(emb)
    sizes = np.bincount(lab, minlength=p).tolist()

    info = dict(n_genes=len(genes), n_genes_with_edge=n_keep,
                n_tf=n_tf, n_edges=int(tr.shape[0]), p=p,
                min_size=int(min(sizes)), max_size=int(max(sizes)),
                rank=int(np.linalg.matrix_rank(C)),
                n_covered=int((np.abs(C).sum(1) > 0).sum()),
                col_norm_min=float(np.linalg.norm(C, axis=0).min()),
                col_norm_max=float(np.linalg.norm(C, axis=0).max()))
    return C, sizes, info


def permute_modules(C, seed=0):
    """Same values in every column, reassigned to random genes among the covered set.

    This is the content control for B1: rank, column norms and the number of
    touched genes are untouched; only the gene-to-loading alignment is destroyed.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros_like(C)
    covered = np.where(np.abs(C).sum(1) > 0)[0]
    perm = rng.permutation(covered)
    out[covered, :] = C[perm, :]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes-from",
                    default=r"F:\UNAGI_Project\prior_delivery_runs\dose20_e45\s42\real\support.npz")
    ap.add_argument("--trrust", default=r"F:\UNAGI_Project\unagi_custom_data\trrust_raw_human.txt")
    ap.add_argument("--p", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=r"F:\UNAGI_Project\prior_delivery_runs\b1\modules_p32.npz")
    a = ap.parse_args()

    d = np.load(a.genes_from, allow_pickle=True)
    gene_names = np.asarray(d["gene_names"]).astype(str)
    C, sizes, info = build_modules(gene_names, a.trrust, p=a.p, seed=a.seed)

    P = permute_modules(C, seed=a.seed + 1)
    info["perm_rank"] = int(np.linalg.matrix_rank(P))
    info["perm_covered"] = int((np.abs(P).sum(1) > 0).sum())
    # (a) the two bases must be *second-order indistinguishable*: any penalty that
    #     acts on column norms or on C^T C alone sees the same thing, so a
    #     difference in the trained model cannot come from the constraint's size.
    info["basis_gram_frob_diff"] = float(np.linalg.norm(C.T @ C - P.T @ P))
    # (b) and they must be *first-order different*: the gene-gene similarity each
    #     basis implies has to actually change, otherwise the control tests nothing.
    S_c = C @ C.T
    S_p = P @ P.T
    info["gene_similarity_frob_diff"] = float(np.linalg.norm(S_c - S_p))
    info["gene_similarity_frob_scale"] = float(np.linalg.norm(S_c))
    nz = np.abs(C).sum(1) > 0
    num = float((S_c[np.ix_(nz, nz)] * S_p[np.ix_(nz, nz)]).sum())
    den = float(np.linalg.norm(S_c[np.ix_(nz, nz)]) * np.linalg.norm(S_p[np.ix_(nz, nz)]))
    info["gene_similarity_cosine"] = num / den if den else float("nan")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, C=C, P=P, sizes=np.asarray(sizes),
                        gene_names=gene_names)
    print(json.dumps(info, indent=2))
    print(f"\n[modules] wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
