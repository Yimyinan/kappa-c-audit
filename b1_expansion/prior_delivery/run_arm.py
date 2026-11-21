#!/usr/bin/env python3
"""prior_delivery.run_arm -- train one arm of the prior-delivery experiment.

Each arm is an independent UNAGI run in its own directory with the same seed,
the same cells and the same schedule. The only difference is the per-gene prior
that gets written into the ``geneWeight`` layer (see ``arms.py``).

Two iterations are the minimum that makes the experiment meaningful:

  iteration 0  trains with no gene weights (stagedata does not exist yet), then
               writes stagedata/geneWeight and applies the prior to it
  iteration 1  merges those stage files and trains *with* gene weights, so the
               prior is inside the reconstruction term of the loss

The audited model is therefore the one saved at iteration 1. Iteration 0 is not
audited: it cannot have seen the prior.

Output per arm, in ``--workdir``:
  latent.npz        Z (cells_selected x latent), X (cells_selected x genes), idx
  arm_meta.json     prior description, geneWeight delivery stats, timings
"""
from __future__ import annotations

import os
import sys
import json
import time
import shutil
import argparse
import warnings

BASE = r"F:\UNAGI_Project\b1_expansion"
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "UNAGI_source"))

import numpy as np
import anndata as ad
import scipy.sparse as sp
import torch

warnings.filterwarnings("ignore")

# joblib's physical-core probe shells out to `wmic`, which no longer exists on
# current Windows builds; CPO calls into joblib and dies with a CreateProcess
# traceback before any clustering happens. Pinning the count skips the probe.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 8))

from pg_unagi.config import HEMATOPOIETIC_TFS
from pg_unagi.tf_database import build_tf_database
from pg_unagi.wrapper import CustomUNAGIWrapper

import arms as A


def stratified_subsample(dat, per_stage: int, seed: int = 0):
    """Take at most ``per_stage`` cells from each stage.

    Random subsampling was tried first and broke the pipeline: at 6k cells the
    CPO clustering left stages with no edges between them and UNAGI raised
    'No edges found between stage 1 and stage 2'. Sampling within each stage
    keeps every stage populated, which is what the temporal graph needs.
    """
    if per_stage <= 0:
        return dat
    stage = dat.obs["stage"].astype(int).values
    rng = np.random.default_rng(seed)
    keep = []
    for s in np.unique(stage):
        idx = np.where(stage == s)[0]
        if idx.size > per_stage:
            idx = rng.choice(idx, per_stage, replace=False)
        keep.append(np.sort(idx))
    keep = np.sort(np.concatenate(keep))
    print(f"[arm] stratified subsample: {keep.size} of {dat.n_obs} cells "
          f"({per_stage}/stage)", flush=True)
    return dat[keep].copy()


def stage_gene_weight_stats(workdir: str, iterations):
    """What landed in stagedata/geneWeight, per iteration that has one.

    The layer is written at the end of iteration k for iteration k+1, so the
    audited iteration `last` has no stagedata of its own; the informative files
    are the ones written by iteration `last - 1`.
    """
    out = {}
    for it in iterations:
        sd = os.path.join(workdir, str(it), "stagedata")
        if not os.path.isdir(sd):
            continue
        rows = []
        for f in sorted(os.listdir(sd)):
            if not f.endswith(".h5ad"):
                continue
            a = ad.read_h5ad(os.path.join(sd, f), backed="r")
            W = a.layers.get("geneWeight")
            if W is None:
                rows.append(dict(file=f, present=False))
            else:
                nnz = int(W.nnz) if sp.issparse(W) else int((W != 0).sum())
                mx = float(W.max()) if nnz else 0.0
                rows.append(dict(file=f, present=True, shape=list(W.shape),
                                 nnz=nnz, max=mx,
                                 fmt=getattr(W, "format", "dense")))
            a.file.close()
        out[str(it)] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["empty", "graded", "binary", "shuffle", "shuffleflat"])
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--h5ad", default=r"F:\UNAGI_Project\hemato_input\GSE150774_hemato_20466.h5ad")
    ap.add_argument("--idrem", default=r"F:\UNAGI_Project\idrem")
    ap.add_argument("--cache", default=r"F:\UNAGI_Project\unagi_custom_data")
    ap.add_argument("--max-iter", type=int, default=2)
    ap.add_argument("--epoch-initial", type=int, default=20)
    ap.add_argument("--epoch-iter", type=int, default=10)
    ap.add_argument("--per-stage", type=int, default=0, help="0 = use every cell")
    ap.add_argument("--cpo", choices=["on", "off"], default="on")
    ap.add_argument("--edge-cutoff", type=float, default=0.05,
                    help="connectNodes threshold for the stage-to-stage graph. UNAGI's "
                         "0.05 assumes the full 32k-cell dataset; on a stratified "
                         "subsample too few cluster pairs fall that far into the left "
                         "tail and buildEdges raises 'No edges found'.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle-seed", type=int, default=7)
    ap.add_argument("--weight-mode", choices=["rank", "direct"], default="rank",
                    help="rank: UNAGI's 1+1/sqrt(rank) transform (per-gene ratio capped at "
                         "~1.98). direct: use the stored value as the per-gene weight, which "
                         "is the only way to reach a dose above that ceiling.")
    ap.add_argument("--prior-level", type=float, default=2.0,
                    help="flat factor for --arm binary; also the dose in --weight-mode direct")
    ap.add_argument("--support-k", type=int, default=0,
                    help="widen the prior to k genes (curated core first, then highest "
                         "TRRUST in-degree). 0 = keep the arm's own support.")
    ap.add_argument("--no-normalize-total", dest="normalize_total",
                    action="store_false", default=True,
                    help="do not rescale weights to keep total reconstruction mass fixed")
    ap.add_argument("--audit-cells", type=int, default=3000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume", action="store_true",
                    help="reuse an existing trained run and only re-extract")
    ap.add_argument("--stop-after-last-train", dest="stop_after_last_train",
                    action="store_true", default=True,
                    help="stop once the last audited iteration's model is saved, "
                         "instead of preparing an iteration that is never run")
    ap.add_argument("--no-early-stop", dest="stop_after_last_train",
                    action="store_false")
    # ---- B1: geometry prior on the decoder Jacobian ----------------------
    ap.add_argument("--basis", choices=["none", "real", "perm"], default="none",
                    help="real: gene-module basis from TRRUST; perm: the same basis "
                         "with gene labels permuted (content control). Requires "
                         "--arm empty (B1 replaces the weight operator, not adds to it)")
    ap.add_argument("--lambda-align", type=float, default=0.0,
                    help="strength of the Jacobian subspace penalty; 0 = off")
    ap.add_argument("--basis-file",
                    default=r"F:\UNAGI_Project\prior_delivery_runs\b1\modules_p32.npz")
    ap.add_argument("--gcn", choices=["on", "off"], default="on",
                    help="graph-conv encoder (on) vs plain MLP encoder (off); B factorial")
    ap.add_argument("--no-adversarial", action="store_true",
                    help="train the VAE without the discriminator loss. Used to test "
                         "whether the run-to-run scatter in A comes from the GAN "
                         "branch, which is the prescription P3 gives its readers.")
    a = ap.parse_args()

    os.makedirs(a.workdir, exist_ok=True)
    os.environ.setdefault("PYTHONHASHSEED", str(a.seed))

    h5 = os.path.join(a.workdir, "hemato.h5ad")
    if not os.path.exists(h5):
        dat = ad.read_h5ad(a.h5ad)
        dat = stratified_subsample(dat, a.per_stage, a.seed)
        print(f"[arm] data {dat.shape[0]} cells x {dat.shape[1]} genes", flush=True)
        dat.write_h5ad(h5)
    else:
        print("[arm] reusing existing hemato.h5ad", flush=True)

    cache_dst = os.path.join(a.workdir, "unagi_custom_data")
    if not os.path.isdir(cache_dst):
        shutil.copytree(a.cache, cache_dst)

    os.chdir(a.workdir)
    import random
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    tf_db = build_tf_database(HEMATOPOIETIC_TFS, force_rebuild=False)
    print(f"[arm] TF db: {len(tf_db)} TFs, "
          f"{sum(len(v) for v in tf_db.values())} edges", flush=True)

    t_start = time.time()
    wrapper = CustomUNAGIWrapper(
        h5ad_path=h5,
        curriculum_config={"mode": "linear"},
        tf_list=HEMATOPOIETIC_TFS,
        tf_prior_database=tf_db,
        device=a.device,
        disable_curriculum=False,
        disable_mixed_dist=False,
        disable_tf_prior=(a.arm == "empty"),
    )
    t0 = time.time()
    wrapper.setup_data()
    print(f"[arm] setup_data {time.time()-t0:.1f}s", flush=True)

    wrapper.setup_training(task=f"prior_{a.arm}", max_iter=a.max_iter,
                           epoch_initial=a.epoch_initial,
                           epoch_iter=a.epoch_iter,
                           GCN=(a.gcn == "on"))

    # Supply gene weights from outside instead of letting UNAGI derive them from
    # iDREM trajectories + ENCODE fold-changes. That route writes an all-zero layer
    # on this dataset (paper1's central observation) and crashes on a subsampled
    # stage graph. See UNAGI_runner._write_gene_weights_from_prior.
    wrapper.unagi._direct_gene_weights = True

    tr0 = getattr(wrapper.unagi, "unagi_trainer", None)
    if tr0 is not None and a.weight_mode == "direct":
        # Read the stored value as the per-gene weight, bypassing the rank transform
        # that caps the ratio at ~1.98. Set on the instance the runner will use.
        tr0.direct_gene_weights = True
        print(f"[arm] weight mode = direct (per-gene factor used as-is, level {a.prior_level:g})",
              flush=True)

    # ---- B1: attach the gene-module basis before training starts ---------
    basis_info = None
    if a.basis != "none":
        if a.arm != "empty":
            print("[arm] FATAL: --basis requires --arm empty; B1 replaces the weight "
                  "operator and must not be stacked on it", flush=True)
            return 2
        if a.lambda_align < 0:
            print("[arm] FATAL: --lambda-align must be >= 0 "
                  "(0 keeps the basis attached for measurement without penalising)",
                  flush=True)
            return 2
        if tr0 is None:
            print("[arm] FATAL: no trainer instance to attach the basis to", flush=True)
            return 2
        _d = np.load(a.basis_file, allow_pickle=True)
        C = _d["C"] if a.basis == "real" else _d["P"]
        cov = np.where(np.abs(C).sum(1) > 0)[0]
        tr0.set_module_basis(C[cov], cov, a.lambda_align)
        basis_info = dict(kind=a.basis, file=a.basis_file, p=int(C.shape[1]),
                          n_covered=int(cov.size), lambda_align=a.lambda_align)

    # Replace the wrapper-built prior with this arm's variant. graded is what the
    # wrapper already produced; the other arms derive from it.
    prior = None
    if a.arm != "empty":
        graded = wrapper.tf_prior_weights
        if graded is None:
            print("[arm] FATAL: wrapper produced no prior", flush=True)
            return 2
        if a.arm == "graded":
            prior = graded
        elif a.arm == "binary":
            prior = A.arm_binary(graded, level=a.prior_level)
        elif a.arm == "shuffle":
            prior = A.arm_shuffle(graded, seed=a.shuffle_seed)
        elif a.arm == "shuffleflat":
            # content control matched to a flat dose: same support size, same value,
            # different genes. Paired on gene identity with the flat arm.
            prior = A.arm_shuffle_flat(graded, level=a.prior_level,
                                       seed=a.shuffle_seed)
        wrapper.tf_prior_weights = prior
        wrapper.unagi._tf_prior_weights = prior

    # ---- support ladder: widen the prior to k genes -------------------------
    n_genes = int(wrapper.n_genes)
    support = 0
    if a.support_k > 0 and prior is not None:
        ranked, avail = A.trrust_degree_ranking(
            list(wrapper.adata.var_names),
            r"F:\UNAGI_Project\unagi_custom_data\trrust_raw_human.txt")
        prior, support = A.expand_support(prior, list(wrapper.adata.var_names),
                                          ranked, a.support_k, level=a.prior_level)
        print(f"[arm] support ladder: requested k={a.support_k}, realised {support} "
              f"(TRRUST targets available in this dataset: {avail})", flush=True)
        wrapper.tf_prior_weights = prior
        wrapper.unagi._tf_prior_weights = prior

    # ---- keep total reconstruction mass fixed -------------------------------
    if prior is not None and a.weight_mode == "direct" and a.normalize_total:
        sel = prior > 1.0
        support = int(sel.sum())
        if support:
            # Concentrating the same total is the manipulation; letting the total grow
            # as well would confound the prior with a global reduction of KL pressure.
            # Computed from the actual vector, so it is correct for graded arms too.
            mean_w = (float(prior[sel].sum()) + (n_genes - support)) / n_genes
            scale = 1.0 / mean_w
            wrapper.unagi._gene_weight_scale = scale
            if tr0 is not None:
                tr0.direct_weight_baseline = scale
            vals = np.unique(np.round(prior[sel] * scale, 4))
            vtxt = (f"{vals[0]:.3f}" if vals.size == 1
                    else f"{vals.min():.3f}-{vals.max():.3f}")
            print(f"[arm] weights normalised: selected {vtxt}, baseline {scale:.3f} "
                  f"(mean weight 1.0, was {mean_w:.3f})", flush=True)

    if prior is None:
        prof = dict(n_boosted=0)
        print("[arm] arm=empty: no prior attached", flush=True)
    else:
        # Record which gene positions carry weight, so a later per-gene analysis can
        # pair arms on gene identity without re-deriving the shuffle.
        sel = np.where(np.asarray(prior) > 1)[0]
        np.savez(os.path.join(a.workdir, "support.npz"),
                 idx=sel, gene_names=np.asarray(wrapper.adata.var_names).astype(str))
        prof = A.ranking_profile(prior)
        prof["weight_mode"] = a.weight_mode
        prof["prior_level"] = a.prior_level
        prof["support_k"] = a.support_k
        prof["support_realised"] = int((prior > 1).sum()) if prior is not None else 0
        if a.weight_mode == "direct":
            n_sel = prof["support_realised"]
            base = float(getattr(tr0, "direct_weight_baseline", 1.0)) if tr0 else 1.0
            frac = 100.0 * n_sel / n_genes
            vals = np.unique(np.round(prior[prior > 1] * base, 4)) if n_sel else np.array([1.0])
            vtxt = (f"{vals[0]:.3f}" if vals.size == 1
                    else f"{vals.min():.3f}-{vals.max():.3f}")
            print(f"[arm] prior {a.arm}: {n_sel} genes ({frac:.2f}% of {n_genes}) at "
                  f"weight {vtxt}, baseline {base:.3f} (direct mode)", flush=True)
        else:
            print(f"[arm] prior {a.arm}: {prof['n_boosted']} genes boosted, "
                  f"raw {prof['raw_min']:.2f}-{prof['raw_max']:.2f}, "
                  f"effective rank score {prof['eff_bottom_boosted']:.3f}-{prof['eff_top']:.3f} "
                  f"vs unboosted {prof['eff_unboosted']:.4f}", flush=True)

    # per-epoch timing, so the budget for the remaining arms can be set from data
    tr = getattr(wrapper.unagi, "unagi_trainer", None)
    epochs = []
    captured = {}

    class _AuditReady(Exception):
        """Raised once the last iteration's model is saved.

        After iteration `last` finishes training, UNAGI goes on to re-cluster,
        rebuild the temporal graph, run iDREM (a few minutes of Java) and rewrite
        the gene weights -- all of it only to prepare iteration `last + 1`, which
        this experiment never runs. Stopping here saves that work without touching
        the audited model or its training data.
        """

    if tr is not None:
        _orig = tr.train_model

        def _timed(*args, **kw):
            t = time.time()
            r = _orig(*args, **kw)
            dt = time.time() - t
            epochs.append(dt)
            print(f"[arm] epoch {len(epochs)-1:03d} {dt:.1f}s", flush=True)
            return r

        tr.train_model = _timed

        # Capture the merged iteration dataset: UNAGI builds it in memory
        # (mergeAdata writes only stagedata/attribute.pkl, never an .h5ad), so the
        # object handed to the trainer is the only copy of the audited data.
        _orig_train = tr.train

        def _cap(adata, iteration, *args, **kw):
            captured[iteration] = adata
            if a.no_adversarial:
                # train(adata, iteration, target_dir, adversarial, is_iterative)
                args = list(args)
                if len(args) >= 2:
                    args[1] = False
                else:
                    kw["adversarial"] = False
                args = tuple(args)
            out = _orig_train(adata, iteration, *args, **kw)
            if a.stop_after_last_train and iteration == a.max_iter - 1:
                raise _AuditReady(iteration)
            return out

        tr.train = _cap

    if not a.resume:
        t0 = time.time()
        try:
            wrapper.unagi.run_UNAGI(idrem_dir=a.idrem, CPO=(a.cpo == "on"),
                                    connect_edges_cutoff=a.edge_cutoff)
        except _AuditReady as done:
            print(f"[arm] stopped after iteration {done} training "
                  f"(model saved; skipped iteration {int(str(done)) + 1} prep)", flush=True)
        print(f"[arm] training+graph finished in {time.time()-t0:.1f}s", flush=True)

    last = a.max_iter - 1
    stats = stage_gene_weight_stats(a.workdir, range(a.max_iter))

    # ---- latent extraction on the iteration-`last` dataset -------------------
    dmat = captured.get(last)
    if dmat is None:
        ds_path = os.path.join(a.workdir, f"{last}.h5ad")
        if not os.path.exists(ds_path):
            print(f"[arm] FATAL: no captured dataset and {ds_path} missing", flush=True)
            return 2
        dmat = ad.read_h5ad(ds_path)
    print(f"[arm] audit dataset {dmat.shape[0]} x {dmat.shape[1]}", flush=True)

    t0 = time.time()
    z_locs, z_scales, TZ = wrapper.unagi.unagi_trainer.get_latent_representation(
        dmat, last, a.workdir)
    print(f"[arm] latent extraction {time.time()-t0:.1f}s, Z {TZ.shape}", flush=True)

    n = TZ.shape[0]
    rng = np.random.default_rng(1234)  # same cells for every arm
    sel = np.sort(rng.choice(n, min(a.audit_cells, n), replace=False))
    X = dmat.X[sel]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    X = X.astype(np.float32)
    np.savez_compressed(os.path.join(a.workdir, "latent.npz"),
                        Z=TZ[sel].astype(np.float32), X=X, idx=sel,
                        obs_names=np.asarray(dmat.obs_names)[sel].astype(str))

    align_stats = None
    if tr0 is not None and getattr(tr0, "module_basis", None) is not None:
        try:
            # mean penalty per batch over the whole run, and the inside-energy share
            # of the last batch — enough to confirm the switch was actually connected
            align_stats = tr0.pop_align_stats()
        except Exception as exc:                     # never fail a finished run on a stat
            align_stats = dict(error=repr(exc))

    meta = dict(
        arm=a.arm, seed=a.seed, shuffle_seed=a.shuffle_seed,
        weight_mode=a.weight_mode, prior_level=a.prior_level,
        support_k=a.support_k, support_realised=prof.get("support_realised", 0),
        normalized_total=bool(a.normalize_total and a.support_k > 0),
        gcn=a.gcn, max_iter=a.max_iter, epoch_initial=a.epoch_initial,
        epoch_iter=a.epoch_iter, per_stage=a.per_stage, cpo=a.cpo,
        n_cells=int(dmat.shape[0]), n_genes=int(dmat.shape[1]),
        prior=prof,
        basis=basis_info,
        adversarial=not a.no_adversarial,
        align_stats=align_stats,
        stagedata_final=stats,
        epoch_seconds=epochs,
        total_seconds=time.time() - t_start,
    )
    with open(os.path.join(a.workdir, "arm_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[arm] DONE {a.arm} in {meta['total_seconds']:.0f}s "
          f"({len(epochs)} epochs, mean {np.mean(epochs) if epochs else 0:.1f}s/epoch)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
