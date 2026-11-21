#!/usr/bin/env python3
"""prior_delivery.audit_arms -- dual-spectrum audit of the four arms.

For each arm the audit is run against ``empty`` as the family-matched reference,
which is the role the vanilla model plays in paper1: same architecture, same
cells, same schedule, only no prior.

Reports, per arm:
  kappa_C_W          sum_g R2(arm) / sum_g R2(empty)   -- decoder readout spectrum
  kappa_C_Z          tr(cov Z_arm)/tr(cov Z_empty)      -- latent spectrum
  eff_rank_*         effective rank of the latent covariance
  verdict            the package's banded classification

Also reports the latent-side quantities that do not depend on a readout (mean
latent variance, participation ratio), because a readout-only comparison is the
exact failure mode paper1 documents.
"""
from __future__ import annotations

import os
import sys
import json
import glob
import argparse

import numpy as np

sys.path.insert(0, r"F:\UNAGI_Project\paper1\paper1_kc_diagnostic\kappa_c_audit")

from kappa_c_audit import audit, eff_rank, participation_ratio  # noqa: E402

ARM_ORDER = ["empty", "graded", "binary", "shuffle"]


def load_arm(root, arm):
    p = os.path.join(root, arm, "latent.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    meta_p = os.path.join(root, arm, "arm_meta.json")
    meta = json.load(open(meta_p, encoding="utf-8")) if os.path.exists(meta_p) else {}
    return dict(Z=d["Z"], X=d["X"], idx=d["idx"],
                obs=np.asarray(d["obs_names"]).astype(str), meta=meta)


def latent_extras(Z):
    C = np.cov(np.asarray(Z, dtype=np.float64), rowvar=False)
    ev = np.linalg.eigvalsh(C)
    return dict(
        eff_rank=float(eff_rank(C)),
        participation_ratio=float(participation_ratio(C)),
        mean_var=float(np.mean(np.diag(C))),
        trace=float(np.trace(C)),
        # fraction of total latent variance carried by the leading component
        pc1_share=float(ev[-1] / np.trace(C)) if np.trace(C) > 0 else float("nan"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="directory holding <arm>/latent.npz")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--arms", nargs="*", default=None,
                    help="arm directory names; default is the four rank-mode arms")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    order = a.arms if a.arms else ARM_ORDER
    if a.ref is None:
        a.ref = order[0]

    arms = {}
    for arm in order:
        d = load_arm(a.root, arm)
        if d is not None:
            arms[arm] = d
    if a.ref not in arms:
        print(f"[audit] reference arm '{a.ref}' missing under {a.root}")
        return 2

    ref = arms[a.ref]
    ref_idx = ref["idx"]
    rows = []
    for arm, d in arms.items():
        if not np.array_equal(d["idx"], ref_idx):
            print(f"[audit] WARNING: {arm} audited a different cell set; skipping")
            continue
        if not np.array_equal(d["obs"], ref["obs"]):
            print(f"[audit] WARNING: {arm} obs order differs; skipping")
            continue
        if arm == a.ref:
            res = audit(ref["Z"], ref["X"])
            kCW = kCZ = 1.0
            er_change = 0.0
        else:
            res = audit(ref["Z"], ref["X"], d["Z"], d["X"])
            kCW = float(res["kappa_C_W"])
            kCZ = float(res["kappa_C_Z"])
            er_change = float(res["eff_rank_change_percent"])
        ex = latent_extras(d["Z"])
        # what the loss actually saw: nnz of the geneWeight layer per stage file
        st = d["meta"].get("stagedata_final") or {}
        per_iter = {it: [r.get("nnz", 0) for r in rows if r.get("present")]
                    for it, rows in st.items()}
        nnz_all = [v for vals in per_iter.values() for v in vals]
        rows.append(dict(
            arm=arm,
            kappa_C_W=kCW,
            C_percent=float((1 - kCW) * 100),
            kappa_C_Z=kCZ,
            verdict=res.get("verdict"),
            eff_rank=ex["eff_rank"],
            eff_rank_change_percent=er_change,
            participation_ratio=ex["participation_ratio"],
            mean_var=ex["mean_var"],
            pc1_share=ex["pc1_share"],
            n_boosted=d["meta"].get("prior", {}).get("n_boosted", 0),
            weight_mode=d["meta"].get("weight_mode", "rank"),
            prior_level=d["meta"].get("prior_level"),
            geneWeight_nnz=nnz_all,
            geneWeight_nnz_by_iter=per_iter,
            mean_epoch_s=float(np.mean(d["meta"].get("epoch_seconds") or [0]) or 0),
        ))

    print(f"\n[audit] reference = {a.ref}  ({ref['Z'].shape[0]} cells, "
          f"{ref['Z'].shape[1]} latent dims, {ref['X'].shape[1]} genes)\n")
    hdr = (f"{'arm':10s} {'mode':6s} {'level':>6s} {'kappa_C_W':>10s} {'C%':>7s} "
           f"{'kappa_C_Z':>10s} {'effrank':>8s} {'pc1share':>9s} {'nBoost':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        lvl = r["prior_level"]
        lvl_s = f"{lvl:6g}" if isinstance(lvl, (int, float)) else f"{'--':>6s}"
        print(f"{r['arm']:10s} {r['weight_mode']:6s} {lvl_s} "
              f"{r['kappa_C_W']:10.4f} {r['C_percent']:7.2f} "
              f"{r['kappa_C_Z']:10.4f} {r['eff_rank']:8.3f} "
              f"{r['pc1_share']:9.4f} {r['n_boosted']:7d}")

    out = a.out or os.path.join(a.root, "audit_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(reference=a.ref, rows=rows), f, indent=2)
    print(f"\n[audit] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
