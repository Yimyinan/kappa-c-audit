#!/usr/bin/env python3
"""prior_delivery.recon_pass -- per-gene reconstruction R2 from a saved checkpoint.

Why this cannot reuse the saved latent
--------------------------------------
UNAGI's ``obsm['z']`` -- and ``trainer.get_latent_representation``, which is where it
comes from -- stores ``mu + logvar``, not a sample from the posterior and not ``mu``.
That is the quantity paper1 audits, so the kappa statistics are on the same definition
as the paper. But it is *not* the code the decoder was trained on: feeding it back
through the decoder reconstructs nothing (median per-gene R2 ~ -41, checked).

So reconstruction needs one real forward pass: encoder(mu) -> decoder mean. The encoder
needs the cell graph, so this script rebuilds the merged iteration dataset from the
stage files UNAGI wrote at iteration 0 -- which is exactly what UNAGI's own mergeAdata
consumes -- and then runs the forward in batches, keeping only the audited cells.

Output per arm directory: ``pergene_r2.npy`` (length = n_genes), reconstruction R2 of
the decoder mean against the input expression, over the same cells the kappa audit used.
"""
from __future__ import annotations

import os
import sys
import json
import argparse

import numpy as np
import anndata as ad
import scipy.sparse as sp

BASE = r"F:\UNAGI_Project\b1_expansion"
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "UNAGI_source"))

import torch


def load_merged(arm_dir, iteration=0):
    """Rebuild the merged dataset UNAGI trained on, from its stage files."""
    sd = os.path.join(arm_dir, str(iteration), "stagedata")
    files = sorted(f for f in os.listdir(sd) if f.endswith(".h5ad"))
    if not files:
        raise FileNotFoundError(f"no stage files under {sd}")
    Xs, coos, names = [], [], None
    for f in files:
        a = ad.read_h5ad(os.path.join(sd, f))
        Xs.append(sp.csr_matrix(a.X))
        coos.append(a.obsp["gcn_connectivities"].tocoo())
        if names is None:
            names = np.asarray(a.var_names).astype(str)
    X = sp.vstack(Xs, format="csr")
    offs = np.cumsum([0] + [c.shape[0] for c in coos])
    rows, cols, data = [], [], []
    for i, c in enumerate(coos):
        rows.append(c.row + offs[i])
        cols.append(c.col + offs[i])
        data.append(c.data)
    n = X.shape[0]
    gcn = sp.csr_matrix((np.concatenate(data),
                         (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return X, gcn, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--iteration", type=int, default=None,
                    help="checkpoint iteration; default = max_iter - 1")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from UNAGI.model.models import VAE
    from UNAGI.utils.gcn_utils import setup_graph
    from UNAGI.utils.h5adReader import H5ADDataSet
    from torch.utils.data import DataLoader

    d = np.load(os.path.join(a.arm, "latent.npz"), allow_pickle=True)
    audit_idx = np.asarray(d["idx"], dtype=int)
    audit_X = d["X"]

    p = json.load(open(os.path.join(a.arm, "model_save", "training_parameters.json"),
                       encoding="utf-8"))
    it = a.iteration if a.iteration is not None else int(p.get("max_iter", 2)) - 1
    ckpt = os.path.join(a.arm, "model_save", f"{p['task']}_{it}.pth")

    X, gcn, names = load_merged(a.arm, 0)
    print(f"[recon] merged {X.shape[0]} cells x {X.shape[1]} genes; "
          f"checkpoint {os.path.basename(ckpt)}", flush=True)

    # sanity: the rebuilt matrix must match the one the kappa audit used on those rows
    Xa = X[audit_idx]
    Xa = Xa.toarray() if sp.issparse(Xa) else np.asarray(Xa)
    same = np.allclose(Xa.astype(np.float32), audit_X, atol=1e-5)
    print(f"[recon] rebuilt X matches audited X on {audit_idx.size} cells: {same}", flush=True)
    if not same:
        print("[recon] FATAL: cell/gene order mismatch; refusing to produce R2", flush=True)
        return 2

    model = VAE(input_dim=int(p["input_dim"]), hidden_dim=int(p["hidden_dim"]),
                graph_dim=int(p["graph_dim"]), latent_dim=int(p["latent_dim"]),
                beta=float(p.get("beta", 1)), distribution=p.get("dist", "ziln"))
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval().to(a.device)

    dev = torch.device(a.device)
    X_t = torch.from_numpy(np.asarray(X.toarray(), dtype=np.float32))
    adj = setup_graph(gcn.tocoo()).to(dev)  # setup_graph reads .row/.col, i.e. COO

    holder = ad.AnnData(X=X.copy())
    holder.obsp["gcn_connectivities"] = gcn
    loader = DataLoader(H5ADDataSet(holder), batch_size=a.batch, num_workers=0)

    wanted = {int(c): k for k, c in enumerate(audit_idx)}
    rec = np.zeros((audit_idx.size, X.shape[1]), dtype=np.float32)
    buf = torch.zeros_like(X_t)
    done = 0
    with torch.no_grad():
        for bi, (x, neigh, idx) in enumerate(loader):
            nb = [j for sub in neigh for j in sub]
            buf[nb] = X_t[nb]
            xb = buf.to(dev)
            idxb = idx.to(dev) if torch.is_tensor(idx) else torch.as_tensor(idx).to(dev)
            mu, _ = model.encoder(xb, adj, idxb)
            de_mean, _ = model.decoder(mu)
            de = de_mean.cpu().numpy()
            for local, cell in enumerate(np.asarray(idx)):
                k = wanted.get(int(cell))
                if k is not None:
                    rec[k] = de[local]
                    done += 1
            buf[nb] = 0.0
            print(f"\r[recon] batch {bi+1}, cells captured {done}/{audit_idx.size}",
                  end="", flush=True)
    print("", flush=True)

    Xf = np.asarray(audit_X, dtype=np.float64)
    R = np.asarray(rec, dtype=np.float64)
    ss_res = ((Xf - R) ** 2).sum(axis=0)
    ss_tot = ((Xf - Xf.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)

    np.save(os.path.join(a.arm, "pergene_r2.npy"), r2)
    np.save(os.path.join(a.arm, "recon_matrix.npy"), rec.astype(np.float32))
    print(f"[recon] per-gene R2: median {np.nanmedian(r2):.4f}, "
          f"IQR {np.nanpercentile(r2, 25):.3f}..{np.nanpercentile(r2, 75):.3f}, "
          f"frac>0 {np.nanmean(r2 > 0):.3f}", flush=True)
    print(f"[recon] mean squared error {np.mean(ss_res / Xf.shape[0]):.5f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
