#!/usr/bin/env python3
"""prior_delivery.cv_nonlinear_null -- three fixes to the readout, zero new training.

1. OUT-OF-FOLD.  `per_gene_r2` is in-sample: Ridge is fitted and scored on the same
   cells. Every per-gene number in the paper (+0.00861, the +0.0145 ceiling, the
   0.01572 decomposition) therefore carries an unknown optimism, and the weighted
   arms need not carry the *same* optimism as the unweighted one. This script
   refits with 5-fold cross-validation and reports both.

2. NONLINEAR READOUT.  kappa-C(W), the module basis and the Jacobian penalty are all
   linear objects. If a prior moves structure that a linear probe cannot see, our
   null is instrument-bound rather than prior-bound. A k-NN readout on the same
   latents tests exactly that, under the identical fold split.

3. GENE-LEVEL NULL.  Three seeds cannot resolve a mean difference whose s.d. is the
   same order as the effect. But there are 20,527 genes: the null distribution of
   "mean dR2 over 203 genes" can be built by resampling gene sets, which buys two
   orders of magnitude of resolution for free.

Usage:
    py -3.13 -u cv_nonlinear_null.py --root F:/UNAGI_Project/prior_delivery_runs/dose20_e45
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load(arm_dir):
    z = np.load(os.path.join(arm_dir, "latent.npz"), allow_pickle=True)
    return np.asarray(z["Z"], dtype=np.float64), np.asarray(z["X"], dtype=np.float64)


def standardise(A):
    mu = A.mean(0, keepdims=True)
    sd = A.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    return (A - mu) / sd


def folds(n, k=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [idx[i::k] for i in range(k)]


def ridge_oof(Z, X, alpha=1.0, k=5, seed=0):
    """Out-of-fold per-gene R2 with the same ridge the audit uses."""
    n, d = Z.shape
    sse = np.zeros(X.shape[1])
    for te in folds(n, k, seed):
        tr = np.setdiff1d(np.arange(n), te, assume_unique=False)
        Ztr, Xtr = Z[tr], X[tr]
        # centred normal equations: G = Z^T Z + alpha I ; B = Z^T X
        G = Ztr.T @ Ztr + alpha * np.eye(d)
        B = Ztr.T @ Xtr
        W = np.linalg.solve(G, B)
        pred = Z[te] @ W
        sse += ((X[te] - pred) ** 2).sum(0)
    sst = (X ** 2).sum(0)          # X is standardised, mean 0
    return 1.0 - sse / np.maximum(sst, 1e-12)


def ridge_insample(Z, X, alpha=1.0):
    n, d = Z.shape
    G = Z.T @ Z + alpha * np.eye(d)
    W = np.linalg.solve(G, Z.T @ X)
    resid = X - Z @ W
    sst = (X ** 2).sum(0)
    return 1.0 - (resid ** 2).sum(0) / np.maximum(sst, 1e-12)


def knn_oof(Z, X, k_nn=15, k=5, seed=0):
    """Out-of-fold k-NN readout: a nonlinear probe under the same split.

    Both Z and X are standardised; the prediction for a held-out cell is the mean
    expression of its `k_nn` nearest training cells in latent space. No linearity
    is assumed anywhere, so if the prior moves structure that ridge cannot see,
    this readout should see it.
    """
    n = Z.shape[0]
    sse = np.zeros(X.shape[1])
    for te in folds(n, k, seed):
        tr = np.setdiff1d(np.arange(n), te, assume_unique=False)
        Ztr, Xtr = Z[tr], X[tr]
        # squared distances via the expansion, so no (n_te, n_tr, d) temporary
        d2 = ((Z[te] ** 2).sum(1)[:, None] + (Ztr ** 2).sum(1)[None, :]
              - 2.0 * (Z[te] @ Ztr.T))
        nn = np.argpartition(d2, k_nn - 1, axis=1)[:, :k_nn]
        pred = Xtr[nn].mean(1)
        sse += ((X[te] - pred) ** 2).sum(0)
    sst = (X ** 2).sum(0)
    return 1.0 - sse / np.maximum(sst, 1e-12)


def gene_null(delta, sizes, obs, n_draw=20000, seed=0, exclude=None):
    """Empirical null for the mean of `delta` over a random gene set of each size.

    `obs[m]` is the observed mean over the real support of size m. The two-sided
    empirical p uses the +/- of the null, so a claim of "higher than chance" has to
    beat the resampling distribution rather than a between-seed s.d. estimated from
    three numbers.
    """
    rng = np.random.default_rng(seed)
    ok = np.isfinite(delta)
    pool = np.where(ok)[0]
    if exclude is not None:
        pool = np.setdiff1d(pool, np.asarray(exclude, dtype=int))
    out = {}
    for m in sizes:
        if m > pool.size:
            continue
        means = np.empty(n_draw)
        for i in range(n_draw):
            means[i] = delta[rng.choice(pool, m, replace=False)].mean()
        o = float(obs[m])
        p_up = float((1 + (means >= o).sum()) / (n_draw + 1))
        p_dn = float((1 + (means <= o).sum()) / (n_draw + 1))
        out[m] = dict(null_mean=float(means.mean()), null_sd=float(means.std()),
                      q01=float(np.quantile(means, 0.01)),
                      q99=float(np.quantile(means, 0.99)),
                      obs=o, p_upper=p_up, p_lower=p_dn,
                      z=float((o - means.mean()) / max(means.std(), 1e-12)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"F:\UNAGI_Project\prior_delivery_runs\dose20_e45")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--arms", nargs="+", default=["empty_e45", "real", "shuffled"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--knn", type=int, default=15)
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    report = {"root": a.root, "folds": a.folds, "knn": a.knn, "per_seed": {}}

    for s in a.seeds:
        base_dir = os.path.join(a.root, f"s{s}", "empty_e45")
        Zb, Xb = load(base_dir)
        Zb, Xb = standardise(Zb), standardise(Xb)

        # cache the baseline once per seed; identical cells across arms by construction
        r2b_is = ridge_insample(Zb, Xb)
        r2b_oof = ridge_oof(Zb, Xb, k=a.folds)
        r2b_knn = knn_oof(Zb, Xb, k_nn=a.knn, k=a.folds)
        row = {"empty_e45": dict(ridge_insample=float(np.nanmean(r2b_is)),
                                 ridge_oof=float(np.nanmean(r2b_oof)),
                                 knn_oof=float(np.nanmean(r2b_knn)),
                                 sum_oof=float(np.nansum(r2b_oof)))}

        for arm in a.arms:
            if arm == "empty_e45":
                continue
            d = os.path.join(a.root, f"s{s}", arm)
            if not os.path.exists(os.path.join(d, "latent.npz")):
                continue
            Z, X = load(d)
            Z, X = standardise(Z), standardise(X)
            print(f"[{s}/{arm}] fitting", flush=True)
            r2_is = ridge_insample(Z, X)
            r2_oof = ridge_oof(Z, X, k=a.folds)
            r2_knn = knn_oof(Z, X, k_nn=a.knn, k=a.folds)

            sup = np.load(os.path.join(d, "support.npz"), allow_pickle=True)
            sidx = np.asarray(sup["idx"], dtype=int)
            nsup = int(sidx.size)

            d_oof = r2_oof - r2b_oof
            d_is = r2_is - r2b_is
            d_knn = r2_knn - r2b_knn
            null = gene_null(d_oof, [nsup], {nsup: float(np.nanmean(d_oof[sidx]))},
                             n_draw=a.draws, exclude=sidx)

            row[arm] = dict(
                ridge_insample=float(np.nanmean(r2_is)),
                ridge_oof=float(np.nanmean(r2_oof)),
                knn_oof=float(np.nanmean(r2_knn)),
                sum_oof=float(np.nansum(r2_oof)),
                n_support=nsup,
                d_insample_support=float(np.nanmean(d_is[sidx])),
                d_oof_support=float(np.nanmean(d_oof[sidx])),
                d_knn_support=float(np.nanmean(d_knn[sidx])),
                d_oof_rest=float(np.nanmean(np.delete(d_oof, sidx))),
                d_knn_rest=float(np.nanmean(np.delete(d_knn, sidx))),
                null=null,
            )
            print(f"[{s}/{arm}] dR2 support: in-sample {row[arm]['d_insample_support']:+.5f} "
                  f"| oof {row[arm]['d_oof_support']:+.5f} | knn {row[arm]['d_knn_support']:+.5f} "
                  f"| null sd {null[nsup]['null_sd']:.5f}", flush=True)

        report["per_seed"][str(s)] = row

    out = a.out or os.path.join(a.root, "cv_nonlinear_null.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {out}")

    # summary across seeds
    print(f"\n{'arm':>10} {'R2 in':>8} {'R2 oof':>8} {'knn oof':>8} "
          f"{'dR2 sup oof':>12} {'dR2 knn':>9}")
    for arm in a.arms:
        vals = [report["per_seed"][str(s)][arm] for s in a.seeds
                if arm in report["per_seed"][str(s)]
                and "d_oof_support" in report["per_seed"][str(s)][arm]]
        if not vals:
            continue
        print(f"{arm:>10} {np.mean([v['ridge_insample'] for v in vals]):8.4f} "
              f"{np.mean([v['ridge_oof'] for v in vals]):8.4f} "
              f"{np.mean([v['knn_oof'] for v in vals]):8.4f} "
              f"{np.mean([v['d_oof_support'] for v in vals]):+12.5f} "
              f"{np.mean([v['d_knn_support'] for v in vals]):+9.5f}")


if __name__ == "__main__":
    main()
