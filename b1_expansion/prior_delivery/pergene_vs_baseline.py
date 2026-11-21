#!/usr/bin/env python3
"""prior_delivery.pergene_vs_baseline -- the absolute test, not the paired one.

Every per-gene number in Parts 5-8 compares the real prior against a weight-matched
shuffled control. That establishes that *content* matters but not whether the prior
beats having no prior at all: if both arms sat below the baseline, "real beats
shuffled" would still be a loss. Part 8's matched 75-epoch `empty` arm makes the
absolute comparison possible for the first time at this budget.

Reports, per seed, the mean per-gene readout dR2 of each weighted set against the
no-prior model:

    real targets      R2(real)     - R2(empty)
    shuffled targets  R2(shuffled) - R2(empty)
    unweighted genes  R2(real)     - R2(shuffled)   (paired content contrast)

The first two answer "does weighting help these genes at all"; the third is the
control from earlier parts, kept for continuity.
"""
from __future__ import annotations

import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pergene_fit import per_gene_r2, support_of  # noqa: E402


def load_r2(arm_dir, cache=True):
    p = os.path.join(arm_dir, "pergene_readout_r2.npy")
    if cache and os.path.exists(p):
        return np.load(p)
    d = np.load(os.path.join(arm_dir, "latent.npz"), allow_pickle=True)
    print(f"[abs] fitting readout for {arm_dir}", flush=True)
    r2 = per_gene_r2(d["Z"], d["X"])
    np.save(p, r2)
    return r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"F:\UNAGI_Project\prior_delivery_runs\dose20_e45")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    ap.add_argument("--empty-name", default="empty_e45")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    summary = {}
    print(f"{'seed':>5s} {'real vs empty':>14s} {'shuf vs empty':>14s} "
          f"{'real vs shuf':>13s} {'n_tgt':>6s}")
    print("-" * 58)
    for seed in a.seeds:
        sdir = os.path.join(a.root, f"s{seed}")
        real_dir = os.path.join(sdir, "real")
        shuf_dir = os.path.join(sdir, "shuffled")
        base_dir = os.path.join(sdir, a.empty_name)

        r2r = load_r2(real_dir)
        r2s = load_r2(shuf_dir)
        r2b = load_r2(base_dir)

        idx_r, _ = support_of(real_dir)
        idx_s, _ = support_of(shuf_dir)
        set_r, set_s = set(idx_r.tolist()), set(idx_s.tolist())
        only_r = np.array(sorted(set_r - set_s), dtype=int)
        only_s = np.array(sorted(set_s - set_r), dtype=int)
        # unweighted in both: the genes neither arm touches
        rest = np.array(sorted(set(range(r2r.size)) - set_r - set_s), dtype=int)

        def m(v, sel):
            x = v[sel]
            x = x[np.isfinite(x)]
            return float(x.mean())

        row = dict(
            n_target=int(only_r.size),
            real_vs_empty=m(r2r - r2b, only_r),
            shuffled_vs_empty=m(r2s - r2b, only_s),
            real_vs_shuffled=m(r2r - r2s, only_r),
            rest_real_vs_empty=m(r2r - r2b, rest),
            rest_shuffled_vs_empty=m(r2s - r2b, rest),
            # what the two arms each do to the genes they *do not* name
            baseline_r2_targets=m(r2b, only_r),
            baseline_r2_shuffled=m(r2b, only_s),
        )
        summary[seed] = row
        print(f"{seed:5d} {row['real_vs_empty']:+14.5f} {row['shuffled_vs_empty']:+14.5f} "
              f"{row['real_vs_shuffled']:+13.5f} {row['n_target']:6d}")

    import statistics as st
    keys = ["real_vs_empty", "shuffled_vs_empty", "real_vs_shuffled",
            "rest_real_vs_empty", "rest_shuffled_vs_empty"]
    print("-" * 58)
    agg = {}
    for k in keys:
        v = [summary[s][k] for s in a.seeds]
        agg[k] = dict(mean=st.mean(v), sd=st.stdev(v) if len(v) > 1 else 0.0,
                      n_positive=sum(1 for x in v if x > 0), values=v)
        print(f"{k:24s} {agg[k]['mean']:+.5f} +/- {agg[k]['sd']:.5f}  "
              f"({agg[k]['n_positive']}/{len(v)} positive)")

    p = a.out or os.path.join(a.root, "pergene_vs_baseline.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(dict(per_seed={str(k): v for k, v in summary.items()}, summary=agg),
                  f, indent=2)
    print(f"\n[abs] wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
