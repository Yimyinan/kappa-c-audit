"""S1 -- re-score RegVelo's with/without-GRN pair under THIS paper's definition.

Why this exists
---------------
The NCS-era audit reported kappa = ||B_grn - B_vanilla||_F / (eig1 - eig2) = 0.42, a
Frobenius-norm ratio against the eigen-gap, with a threshold of kappa > 2. That is not
the quantity used anywhere in paper 1 or paper 3, where

    kappa-C(W) = sum_g R2_g(arm) / sum_g R2_g(reference)

with R2_g from a linear readout of the latent to the gene. Quoting 0.42 next to our
numbers would be comparing two rulers. This script recomputes the pair under ours.

What it does
------------
1. Rebuild both arms from the saved state dicts (custom torch.save: model_state_dict,
   var_names, attr_dict) -- no retraining.
2. Latent codes Z for every cell, per arm.
3. Ridge readout Z -> X_spliced, per arm. Per-gene R2 in-sample and 5-fold held-out,
   with the SAME folds across arms so the contrast is paired.
4. kappa-C(W) = sum R2(grn) / sum R2(vanilla), reported in both conventions of
   thesis 2.24 (all genes / genes that vary), plus the elementwise difference.

Output: JSON next to this script. Nothing is attributed to content: the two arms differ
only in the GRN regulariser, so whatever changes is what that regulariser did.
"""
import os
import sys
import json
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

SRC = r"F:\UNAGI_Project\regvelo-main\regvelo-main\src"
ROOT = r"F:\UNAGI_Project\NCS_submission\paper1_kc_diagnostics\code\regvelo_audit\full2k"
H5AD = r"C:\Users\12901\Downloads\adata_preprocessed_full.h5ad"
TFCSV = r"C:\Users\12901\Downloads\allTFs_hg38.csv"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rescore_regvelo.json")

sys.path.insert(0, SRC)
from regvelo import REGVELOVI            # noqa: E402
import anndata as ad                     # noqa: E402
from sklearn.linear_model import Ridge   # noqa: E402
from scipy.sparse import issparse        # noqa: E402


def load_arm(tag, adata, cfg):
    """Instantiate the arm and load the stored weights into it."""
    p = os.path.join(ROOT, {"vanilla": "regvelo_2k_vanilla",
                            "grn": "regvelo_2k_grn"}[tag], "model.pt")
    blob = torch.load(p, map_location="cpu", weights_only=False)
    sd, var_names, attr = blob["model_state_dict"], blob["var_names"], blob["attr_dict"]
    sub = adata[:, list(var_names)].copy()
    REGVELOVI.setup_anndata(sub, spliced_layer="Ms", unspliced_layer="Mu")
    model = REGVELOVI(sub, **cfg)
    # REGVELOVI is a container, not an nn.Module: the weights live in .module (VELOVAE)
    target = model.module
    missing, unexpected = target.load_state_dict(sd, strict=False)
    target.eval()
    model.is_trained_ = True          # scvi guards the inference methods on this flag
    info = dict(attr={k: str(v) for k, v in list(attr.items())[:12]},
                missing=len(missing), unexpected=len(unexpected))
    return model, sub, info


def per_gene_r2(Z, X, folds, alpha=1.0):
    """In-sample and k-fold held-out per-gene R2 for a linear readout."""
    n = Z.shape[0]
    # in-sample
    # intercept is required for R2: without it the readout predicts ~0 while the
    # counts have a large positive mean, and every gene scores hugely negative.
    r = Ridge(alpha=alpha, fit_intercept=True).fit(Z, X)
    pred = r.predict(Z)
    sst = ((X - X.mean(0, keepdims=True)) ** 2).sum(0)
    sse = ((X - pred) ** 2).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_in = np.where(sst > 1e-12, 1.0 - sse / sst, 1.0)   # constant gene convention

    # held-out
    pred_oof = np.empty_like(X)
    for te in folds:
        tr = np.setdiff1d(np.arange(n), te)
        rr = Ridge(alpha=alpha, fit_intercept=True).fit(Z[tr], X[tr])
        pred_oof[te] = rr.predict(Z[te])
    sse_oof = ((X - pred_oof) ** 2).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_oof = np.where(sst > 1e-12, 1.0 - sse_oof / sst, 1.0)
    return r2_in, r2_oof, r.coef_


def main():
    t0 = time.time()
    print(f"[S1] reading {os.path.basename(H5AD)}", flush=True)
    adata = ad.read_h5ad(H5AD)
    print(f"[S1] data {adata.n_obs} x {adata.n_vars}; layers "
          f"{list(adata.layers.keys())[:6]}", flush=True)

    arms = {
        "vanilla": dict(soft_constraint=False, lam=0.0, lam2=0.0),
        "grn": dict(soft_constraint=True, lam=0.1, lam2=0.01),
    }
    reps, subs, meta, out_eig = {}, {}, {}, {}
    for tag, cfg in arms.items():
        model, sub, info = load_arm(tag, adata, cfg)
        reps[tag] = np.asarray(model.get_latent_representation(sub), dtype=np.float64)
        subs[tag] = sub
        meta[tag] = info
        print(f"[S1] {tag}: Z {reps[tag].shape}  missing={info['missing']} "
              f"unexpected={info['unexpected']}", flush=True)

    # both arms must be on the same genes in the same order
    g = list(subs["vanilla"].var_names)
    assert g == list(subs["grn"].var_names), "gene order differs between arms"
    X = subs["vanilla"].layers["Ms"]
    X = np.asarray(X.toarray() if issparse(X) else X, dtype=np.float64)

    n = X.shape[0]
    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    folds = np.array_split(order, 5)

    out = dict(data=os.path.basename(H5AD), n_cells=int(n), n_genes=int(X.shape[1]),
               readout="Ridge(Z -> Ms), intercept on, alpha=1",
               latent_dim={k: int(v.shape[1]) for k, v in reps.items()},
               load=meta, note="kappa-C(W) = sum R2(grn) / sum R2(vanilla), this paper's definition")

    r2 = {}
    for tag in arms:
        r2[tag] = per_gene_r2(reps[tag], X, folds)
        B = r2[tag][2]
        sv = np.linalg.svd(B, compute_uv=False)
        out_eig[tag] = dict(eig1=float(sv[0] ** 2), eig2=float(sv[1] ** 2),
                            fro=float(np.linalg.norm(B)))
        print(f"[S1] {tag}: mean R2 in-sample {np.nanmean(r2[tag][0]):.5f} "
              f"held-out {np.nanmean(r2[tag][1]):.5f}", flush=True)

    out["decoder"] = out_eig
    out["decoder"]["fro_delta"] = float(abs(out_eig["grn"]["fro"] - out_eig["vanilla"]["fro"]))
    print(f"[S1] decoder eig1 vanilla {out_eig['vanilla']['eig1']:.2f} "
          f"grn {out_eig['grn']['eig1']:.2f}", flush=True)

    var_ok = X.var(0) > 1e-12
    conv = {}
    for name, mask in (("all_genes", np.ones(X.shape[1], bool)), ("varying_only", var_ok)):
        for kind, idx in (("insample", 0), ("heldout", 1)):
            s_v = float(np.nansum(r2["vanilla"][idx][mask]))
            s_g = float(np.nansum(r2["grn"][idx][mask]))
            conv[f"{name}/{kind}"] = dict(
                sum_vanilla=s_v, sum_grn=s_g,
                kappa_CW=(s_g / s_v) if s_v else None,
                delta_kappa=(1 - s_g / s_v) if s_v else None,
                n=int(mask.sum()))
    out["conventions"] = conv
    out["n_constant_genes"] = int((~var_ok).sum())

    # elementwise: does the GRN arm beat the vanilla arm gene by gene?
    for kind, idx in (("insample", 0), ("heldout", 1)):
        d = r2["grn"][idx] - r2["vanilla"][idx]
        out[f"per_gene_delta_{kind}"] = dict(
            mean=float(np.nanmean(d)), frac_pos=float(np.nanmean(d > 0)),
            mean_varying=float(np.nanmean(d[var_ok])))
        print(f"[S1] per-gene delta ({kind}): mean {np.nanmean(d):+.6f} "
              f"frac>0 {np.nanmean(d > 0):.3f}", flush=True)

    out["seconds"] = round(time.time() - t0, 1)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\n[S1] wrote {OUT}  ({out['seconds']} s)")
    for k, v in conv.items():
        print(f"   {k:22s} kappa-C(W) = {v['kappa_CW']:.4f}   "
              f"(sum {v['sum_vanilla']:.1f} -> {v['sum_grn']:.1f})")


if __name__ == "__main__":
    main()
