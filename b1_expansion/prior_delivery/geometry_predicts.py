#!/usr/bin/env python3
"""prior_delivery.geometry_predicts -- does any geometric quantity predict the readout?

The structured-decoder literature assumes that shaping the decoder's geometry is a route to
a better readout. Sections 3 and 6 of paper3 show the aggregate readout does not move. This
script asks the sharper, gene-level version of the same question on decoders that already
exist: for a given gene, does anything about its Jacobian row --- its magnitude, or how much
of it lies in the module basis --- predict how well that gene is reconstructed?

If nothing does, then "geometry -> readout" is not a weak effect but an empty premise for
annotation-derived bases, and every claim of the form "the prior improved the geometry, so
the model is better" lacks a mechanism.

    py -3.13 -u geometry_predicts.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pergene_fit import per_gene_r2  # noqa: E402

MODULES = r"F:\UNAGI_Project\prior_delivery_runs\b1\modules_p32.npz"
OUT = r"F:\UNAGI_Project\prior_delivery_runs\b1\geometry_predicts.json"
ARMS = {
    "annot/real_a": r"F:\UNAGI_Project\prior_delivery_runs\b1\ganoff\s42\real_a",
    "annot/real_b": r"F:\UNAGI_Project\prior_delivery_runs\b1\ganoff\s42\real_b",
    "annot/perm": r"F:\UNAGI_Project\prior_delivery_runs\b1\ganoff\s42\perm",
    "expr/real": r"F:\UNAGI_Project\prior_delivery_runs\b1\expr2\s42\real",
    "expr/perm": r"F:\UNAGI_Project\prior_delivery_runs\b1\expr2\s42\perm",
}


def ckpt(arm_dir):
    p = [q for q in sorted(glob.glob(os.path.join(arm_dir, "model_save", "*_1.pth")))
         if "_dis_" not in os.path.basename(q)]
    return p[-1] if p else None


def jacobian_rows(arm_dir):
    """Per-gene mean row norm and mean in-basis share over cells."""
    sd = torch.load(ckpt(arm_dir), map_location="cpu", weights_only=False)
    sd = {str(k): v for k, v in sd.items()}

    def pick(suf):
        for k in sd:
            if k.endswith(suf):
                return sd[k]
        raise KeyError(suf)
    W3 = pick("fc3.weight").double().numpy()
    b3 = pick("fc3.bias").double().numpy()
    W4 = pick("fc4.weight").double().numpy()
    z = np.asarray(np.load(os.path.join(arm_dir, "latent.npz"),
                           allow_pickle=True)["Z"], dtype=np.float64)
    s = 1.0 / (1.0 + np.exp(-(z @ W3.T + b3)))
    J = (W4[:, None, :] * s.T[:, :, None]) @ W3          # (genes, cells, latent)
    # sum over cells of |J_g(c)|^2, as a (genes, cells) array
    norm2 = (J ** 2).sum(-1)
    return np.sqrt(norm2.mean(1)), J, s


def alignment_with(J, s, W3, Q):
    """Mean fraction of each gene's Jacobian energy that lies in span(Q)."""
    # J_g(c) is a 64-vector; project the (genes, cells, latent) tensor
    # trick: rotate latent coordinates once via Q^T W3^T, no per-gene loop needed
    return None


def main():
    mod = np.load(MODULES, allow_pickle=True)
    C, cov = mod["C"], np.where(np.abs(mod["C"]).sum(1) > 0)[0]
    Qc = np.linalg.qr(C[cov])[0]

    # projector in latent space: for a fixed activation profile s_c, the map is
    # J_g(c) = W4[g] · diag(s_c) · W3, so projecting the *gene* axis is not a simple
    # rotation. Compute the per-gene share directly but on a subsample of cells.
    out = {}
    for name, d in ARMS.items():
        if ckpt(d) is None:
            print(f"{name:14s} pending")
            continue
        sd = torch.load(ckpt(d), map_location="cpu", weights_only=False)
        sd = {str(k): v for k, v in sd.items()}
        pick = lambda suf: next(v for k, v in sd.items() if k.endswith(suf))
        W3 = pick("fc3.weight").double().numpy()
        b3 = pick("fc3.bias").double().numpy()
        W4 = pick("fc4.weight").double().numpy()
        z = np.asarray(np.load(os.path.join(d, "latent.npz"),
                               allow_pickle=True)["Z"], dtype=np.float64)
        s = 1.0 / (1.0 + np.exp(-(z @ W3.T + b3)))

        cells = np.arange(0, s.shape[0], 10)             # 1-in-10 subsample
        W4c = W4[cov]
        gn2 = np.zeros(cov.size)
        inb = np.zeros(cov.size)
        for c in cells:
            Jc = (W4c * s[c][None, :]) @ W3               # (covered, latent)
            n2 = (Jc ** 2).sum(1)
            P = Qc @ (Qc.T @ Jc)
            gn2 += n2
            inb += (P ** 2).sum(1)

        norm = np.sqrt(gn2 / cells.size)
        # A gene whose Jacobian row is numerically zero makes the ratio explode and
        # inflates any mean; report the pooled ratio and restrict the rank statistic
        # to genes with a non-degenerate row.
        pooled = float(inb.sum() / max(gn2.sum(), 1e-30))
        floor = np.quantile(gn2, 0.10)
        share = np.where(gn2 > floor, inb / np.maximum(gn2, 1e-30), np.nan)

        cache = os.path.join(d, "pergene_r2.npy")
        if os.path.exists(cache):
            r2 = np.load(cache)
        else:
            zz = np.load(os.path.join(d, "latent.npz"), allow_pickle=True)
            r2 = per_gene_r2(zz["Z"], zz["X"])
            np.save(cache, r2)

        X = np.asarray(np.load(os.path.join(d, "latent.npz"),
                               allow_pickle=True)["X"], dtype=np.float64)
        varying = X.var(0) > 1e-12
        m = varying[cov] & np.isfinite(r2[cov])
        rho_norm, p_norm = spearmanr(norm[m], r2[cov][m])
        ms = m & np.isfinite(share)
        rho_share, p_share = spearmanr(share[ms], r2[cov][ms])
        out[name] = dict(n_genes=int(m.sum()),
                         rho_norm=float(rho_norm), p_norm=float(p_norm),
                         rho_share=float(rho_share), p_share=float(p_share),
                         pooled_share=pooled,
                         mean_share=float(np.nanmean(share[m])),
                         mean_r2=float(r2[cov][m].mean()))
        print(f"{name:14s} n={m.sum():4d}  rho(|J_g|, R2) = {rho_norm:+.3f}  "
              f"rho(in-basis share, R2) = {rho_share:+.3f}  "
              f"(pooled share {pooled:.3f})", flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
