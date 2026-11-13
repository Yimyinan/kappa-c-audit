#!/usr/bin/env python3
"""prior_delivery.pergene_fit -- did the prior fit its own genes better?

Every earlier probe used an *aggregate*. kappa-C(W) is sum_g R2_g over all 20,527
genes, so a prior touching 1% of them cannot move it -- and the ladders confirmed
that. But an aggregate can stay flat while its parts move in opposite directions.
The mechanism is a per-gene weight on the reconstruction term, so the place to look
for an effect is per gene.

The test is paired on gene identity:

    arm A (real)     203 TRRUST target genes at 5x, other genes at 0.96
    arm B (shuffled) 203 random genes          at 5x, other genes at 0.96

Same cells, seed, schedule, support size and weight; only *which* genes carry it
differs. For every gene g we take d_g = R2_g(A) - R2_g(B), where R2_g is the gene's
ridge-readout R2 from the latent -- i.e. that gene's contribution to kappa-C(W),
using the identical estimator (standardise per column, Ridge alpha=1, no intercept).

If the prior moves predictability toward its own genes while the total stays flat,
the 203 real targets should have d > 0 and the 203 shuffled targets d < 0, because
the two sets swap roles between the arms. If the prior is decorative at the per-gene
level too, both are centred on zero.

Why not reconstruction error: UNAGI's decoder mean is not a usable predictor at this
training budget (45 epochs). Checked directly -- the decoder mean reconstructs the
input with median per-gene R2 of about -100, i.e. far worse than predicting the gene
mean, because the model lowers the ZILN loss by inflating its variance rather than
tracking the mean. A per-gene comparison on that quantity measures the fit failure,
not the prior.
"""
from __future__ import annotations

import os
import sys
import json
import argparse

import numpy as np

BASE = r"F:\UNAGI_Project\b1_expansion"
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "prior_delivery"))


def per_gene_r2(Z, X, alpha: float = 1.0):
    """Per-gene ridge R2 with the exact preprocessing of kappa_c_audit.

    Mirrors ``predictability_total`` and returns the per-gene vector it sums, so
    ``r2.sum()`` equals the numerator of kappa-C(W).
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    Zs = StandardScaler().fit_transform(np.asarray(Z, dtype=np.float64))
    Xs = StandardScaler().fit_transform(np.asarray(X, dtype=np.float64))
    fit = Ridge(alpha=alpha, fit_intercept=False).fit(Zs, Xs)
    resid = Xs - fit.predict(Zs)
    return 1.0 - (resid ** 2).sum(axis=0) / Xs.shape[0]


def _gene_names(arm_dir):
    import anndata as ad
    p = os.path.join(arm_dir, "0", "stagedata", "0.h5ad")
    a = ad.read_h5ad(p, backed="r")
    g = np.asarray(a.var_names).astype(str)
    a.file.close()
    return g


def support_of(arm_dir):
    """Gene positions carrying weight in this arm, with gene names.

    run_arm.py writes support.npz; for arms trained before that existed, rebuild the
    curated support from the TRRUST database against the same gene list.
    """
    p = os.path.join(arm_dir, "support.npz")
    if os.path.exists(p):
        d = np.load(p, allow_pickle=True)
        return np.asarray(d["idx"], dtype=int), np.asarray(d["gene_names"]).astype(str)
    # An arm with no prior attached must report an empty support, not the curated set.
    mp = os.path.join(arm_dir, "arm_meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp, encoding="utf-8"))
        if int(meta.get("prior", {}).get("n_boosted", 0) or 0) == 0:
            return np.array([], dtype=int), None
    import arms as A
    from pg_unagi.config import HEMATOPOIETIC_TFS
    from pg_unagi.tf_database import build_tf_database

    genes = _gene_names(arm_dir)
    cwd = os.getcwd()
    try:
        os.chdir(arm_dir if os.path.isdir(os.path.join(arm_dir, "unagi_custom_data")) else cwd)
        db = build_tf_database(HEMATOPOIETIC_TFS, force_rebuild=False)
    finally:
        os.chdir(cwd)
    w = A.build_graded(genes, db)
    return np.where(w > 1.0)[0], genes


def load_arm(arm_dir, cache=True):
    d = np.load(os.path.join(arm_dir, "latent.npz"), allow_pickle=True)
    Z, X = d["Z"], d["X"]
    cp = os.path.join(arm_dir, "pergene_readout_r2.npy")
    if cache and os.path.exists(cp):
        r2 = np.load(cp)
    else:
        print(f"[fit] fitting readout for {arm_dir}", flush=True)
        r2 = per_gene_r2(Z, X)
        np.save(cp, r2)
    return Z, X, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="arm directory with the real prior")
    ap.add_argument("--shuffled", required=True, help="arm directory with the shuffled prior")
    ap.add_argument("--no-cache", dest="cache", action="store_false", default=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    Za, Xa, r2a = load_arm(a.real, a.cache)
    Zb, Xb, r2b = load_arm(a.shuffled, a.cache)
    if r2a.shape != r2b.shape:
        print("[fit] FATAL: different gene counts")
        return 2
    if not np.allclose(Xa, Xb):
        print("[fit] WARNING: the two arms audited different expression values")

    idx_real, genes = support_of(a.real)
    idx_shuf, _ = support_of(a.shuffled)
    set_real, set_shuf = set(idx_real.tolist()), set(idx_shuf.tolist())
    only_real = np.array(sorted(set_real - set_shuf), dtype=int)
    only_shuf = np.array(sorted(set_shuf - set_real), dtype=int)
    rest = np.array(sorted(set(range(r2a.size)) - set_real - set_shuf), dtype=int)

    d = r2a - r2b
    # The two arms differ by a global offset as well as by which genes carry the
    # weight: at these kappa values almost every gene can drift slightly. Content is
    # what moves a *subset* relative to the rest, so the subset statistics are also
    # reported after subtracting the genome-wide median shift.
    d_centered = d - np.nanmedian(d)

    def stat(sel, label):
        v = d[sel] if sel.size else np.array([])
        v = v[np.isfinite(v)]
        if v.size == 0:
            return None
        vc = d_centered[sel]
        vc = vc[np.isfinite(vc)]
        sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
        return dict(label=label, n=int(v.size), mean=float(v.mean()),
                    median=float(np.median(v)), sd=sd,
                    frac_pos=float((v > 0).mean()),
                    mean_over_sd_x=float(v.mean() / (sd / np.sqrt(v.size)))
                    if sd > 0 else 0.0,
                    mean_centered=float(vc.mean()) if vc.size else 0.0,
                    median_centered=float(np.median(vc)) if vc.size else 0.0,
                    frac_pos_centered=float((vc > 0).mean()) if vc.size else 0.0,
                    # R2 of the arm each gene is weighted in, for context
                    r2_weighted_mean=float(r2a[sel][np.isfinite(r2a[sel])].mean())
                    if label.startswith("real") else
                    float(r2b[sel][np.isfinite(r2b[sel])].mean()))

    rows = [x for x in (stat(only_real, "real targets (weighted in A, not B)"),
                        stat(only_shuf, "shuffled targets (weighted in B, not A)"),
                        stat(rest, "neither (unweighted in both)")) if x]

    agg = float(np.nansum(r2a) / np.nansum(r2b))
    print(f"\n[fit] paired delta R2 = R2({os.path.basename(a.real.rstrip(chr(92)+'/'))}) "
          f"- R2({os.path.basename(a.shuffled.rstrip(chr(92)+'/'))})")
    print(f"[fit] support: real {len(set_real)}, shuffled {len(set_shuf)}, "
          f"overlap {len(set_real & set_shuf)}; aggregate ratio {agg:.6f}")
    hdr = (f"{'subset':46s} {'n':>5s} {'mean dR2':>10s} {'median':>9s} {'sd':>8s} "
           f"{'frac>0':>7s} {'mean/(sd/sqrt n)':>17s} {'mean dR2 - med':>15s} "
           f"{'frac>0 (ctr)':>13s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label']:46s} {r['n']:5d} {r['mean']:+10.5f} {r['median']:+9.5f} "
              f"{r['sd']:8.5f} {r['frac_pos']:7.3f} {r['mean_over_sd_x']:+17.2f} "
              f"{r['mean_centered']:+15.5f} {r['frac_pos_centered']:13.3f}")

    out = a.out or os.path.join(os.path.dirname(a.real.rstrip("\\/")),
                                f"pergene_fit_{os.path.basename(a.real.rstrip(chr(92)+'/'))}"
                                f"_vs_{os.path.basename(a.shuffled.rstrip(chr(92)+'/'))}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(real=a.real, shuffled=a.shuffled, aggregate_ratio=agg,
                       n_overlap=len(set_real & set_shuf), rows=rows), f, indent=2)
    print(f"\n[fit] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
