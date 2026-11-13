# -*- coding: utf-8 -*-
"""checkpoint_spectrum.py -- per-iteration latent spectrum of an existing arm.

The B-family arm on the stratified subsample cannot be re-staged past the third
iteration (buildEdges raises 'No edges found' between stage 1 and stage 2), so a
five-iteration trajectory cannot be produced by raising --max-iter. This script
takes what the aborted run left on disk instead: the saved checkpoints
(iterations 0, 1, 2) are loaded and evaluated on ONE fixed dataset, so the
differences between them are model differences and nothing else.

Flow (the order matters -- UNAGI refuses to run if model_save already exists in
the data folder, and the wrapper deletes model_save in the *current* directory):
  1. fresh data dir A with only hemato.h5ad in it
  2. wrapper.setup_data()  -> creates A/0..4 and an untrained A/model_save
  3. replace A/model_save with the real checkpoints
  4. get_latent_representation(A.adata, k, A) for each k

Reports participation ratio (effective rank) of the latent covariance, its trace
and leading-component share.

Usage:
    py -3.13 checkpoint_spectrum.py --src <arm workdir> --scratch <empty dir> \
        [--iterations 0,1,2] [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, r"F:\UNAGI_Project\b1_expansion")
sys.path.insert(0, r"F:\UNAGI_Project\b1_expansion\prior_delivery")
sys.path.insert(0, r"F:\UNAGI_Project\b1_expansion\UNAGI_source")
sys.path.insert(0, r"F:\UNAGI_Project\paper1\paper1_kc_diagnostic\kappa_c_audit")

from kappa_c_audit import eff_rank, participation_ratio          # noqa: E402
from pg_unagi.config import HEMATOPOIETIC_TFS                    # noqa: E402
from pg_unagi.tf_database import build_tf_database                # noqa: E402
from pg_unagi.wrapper import CustomUNAGIWrapper                   # noqa: E402


def spectrum(Z):
    C = np.cov(np.asarray(Z, dtype=np.float64), rowvar=False)
    ev = np.linalg.eigvalsh(C)
    tr = float(np.trace(C))
    return dict(eff_rank=float(eff_rank(C)),
                participation_ratio=float(participation_ratio(C)),
                trace=tr,
                pc1_share=float(ev[-1] / tr) if tr > 0 else float("nan"),
                n_cells=int(Z.shape[0]), latent_dim=int(Z.shape[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="arm workdir holding model_save/")
    ap.add_argument("--scratch", required=True, help="fresh directory for the audit run")
    ap.add_argument("--task", default="prior_binary")
    ap.add_argument("--iterations", default="0,1,2")
    ap.add_argument("--h5ad", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    iters = [int(s) for s in a.iterations.split(",") if s.strip() != ""]
    src_models = os.path.join(a.src, "model_save")
    for k in iters:
        p = os.path.join(src_models, "%s_%d.pth" % (a.task, k))
        if not os.path.exists(p):
            raise SystemExit("missing checkpoint: %s" % p)

    data_dir = os.path.join(a.scratch, "data")
    cwd_dir = os.path.join(a.scratch, "cwd")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(cwd_dir, exist_ok=True)
    h5 = a.h5ad or os.path.join(a.src, "hemato.h5ad")
    dst = os.path.join(data_dir, "hemato.h5ad")
    if not os.path.exists(dst):
        shutil.copy2(h5, dst)
        print("[scratch] copied %s -> %s" % (h5, dst), flush=True)

    os.chdir(cwd_dir)                      # the wrapper rmtree()s model_save here
    torch.manual_seed(42)
    tf_db = build_tf_database(HEMATOPOIETIC_TFS, force_rebuild=False)
    w = CustomUNAGIWrapper(h5ad_path=dst, curriculum_config={"mode": "linear"},
                           tf_list=HEMATOPOIETIC_TFS, tf_prior_database=tf_db,
                           device="cuda:0", disable_curriculum=False,
                           disable_mixed_dist=False, disable_tf_prior=False)
    w.setup_data()                          # builds stage files + an empty model_save

    # swap in the real checkpoints
    ms = os.path.join(data_dir, "model_save")
    if os.path.isdir(ms):
        shutil.rmtree(ms)
    shutil.copytree(src_models, ms)
    print("[scratch] model_save <- %s" % src_models, flush=True)

    w.setup_training(task=a.task, max_iter=max(iters) + 1, epoch_initial=30,
                     epoch_iter=15, GCN=True)

    # get_latent_representation needs adata.obsp['gcn_connectivities']; the merged
    # object does not carry it, so one fixed kNN graph is built and reused for every
    # checkpoint. The graph is therefore constant across k, which is what makes the
    # comparison a model comparison; the absolute values are not identical to the
    # published arm values, which used each iteration's own staging graph.
    import scanpy as sc
    if "X_pca" not in w.adata.obsm:
        sc.pp.pca(w.adata)
    sc.pp.neighbors(w.adata, n_neighbors=30, use_rep="X_pca")
    w.adata.obsp["gcn_connectivities"] = w.adata.obsp["connectivities"]
    print("[scratch] fixed kNN graph attached (k=30, %d x %d)"
          % w.adata.obsp["gcn_connectivities"].shape, flush=True)

    rows = []
    for k in iters:
        _, _, Z = w.unagi.unagi_trainer.get_latent_representation(w.adata, k, data_dir)
        s = spectrum(Z)
        s["iteration"] = k
        rows.append(s)
        print("iter %d  PR(eff rank) %.4f  trace %.4f  pc1_share %.4f  (%d cells)"
              % (k, s["participation_ratio"], s["trace"], s["pc1_share"], s["n_cells"]),
              flush=True)

    base = rows[0]["participation_ratio"]
    print("\ndPR relative to iteration 0:")
    for r in rows[1:]:
        print("  iter %d: %+.4f  (%+.2f%%)" % (
            r["iteration"], r["participation_ratio"] - base,
            100 * (r["participation_ratio"] - base) / base))

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)
        print("[saved]", a.json)


if __name__ == "__main__":
    main()
