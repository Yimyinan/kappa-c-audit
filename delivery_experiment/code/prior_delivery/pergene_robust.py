#!/usr/bin/env python3
"""prior_delivery.pergene_robust -- is the 20x per-gene tilt a content effect, or
just "genes that were already easier to read out gain more"?

The real support is 203 curated TRRUST targets; the shuffled support is 203
uniformly random genes. Both carry the identical weight after normalisation, so
the paired contrast is clean with respect to the intervention. It is not clean
with respect to baseline predictability: the target genes sit at R2 ~ 0.30-0.32
while the random genes sit at ~0.25-0.28. If easier genes simply respond more to
any reweighting, the +0.016 could be that, not content.

Two controls, both within the same run pair:

1. **Decile matching.** Strate the 20,122 genes weighted in neither arm by their
   baseline R2 (read from the shuffled arm, where they are unweighted), then ask
   what an unweighted gene in the same decile gained. The excess of the real
   targets over their own decile is the content effect at matched baseline.

2. **Linear control.** Fit d_g ~ a + b * R2_base,g on the unweighted genes and
   report the mean residual of the real targets. If b captures the "easier genes
   move more" channel, the residual is what is left.

Neither is a p-value. Both answer the same question: does the sign and size
survive when baseline predictability is held fixed?
"""
from __future__ import annotations

import os
import sys
import json
import argparse

import numpy as np


def load_r2(arm_dir):
    p = os.path.join(arm_dir, "pergene_readout_r2.npy")
    if not os.path.exists(p):
        raise SystemExit(f"[robust] missing cached readout: {p}")
    return np.load(p)


def support_idx(arm_dir, fallback=None):
    """Gene positions carrying weight in this arm.

    Older arms (the 5x `k203` run) predate support.npz. The support is identical
    across levels by construction, so `fallback` may point at the same arm from
    the other level; that is a provenance-preserving substitution, not a guess.
    """
    p = os.path.join(arm_dir, "support.npz")
    if not os.path.exists(p):
        if fallback is None:
            raise SystemExit(f"[robust] no support.npz in {arm_dir} and no fallback")
        p = os.path.join(fallback, "support.npz")
        print(f"[robust] {arm_dir}: using support from {fallback}")
    d = np.load(p, allow_pickle=True)
    return set(d["idx"].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"F:\UNAGI_Project\prior_delivery_runs\dose20")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--real-name", default="real",
                    help="arm directory name for the real prior (k203 in the 5x root)")
    ap.add_argument("--shuf-name", default="shuffled",
                    help="arm directory name for the control (shuffleflat in the 5x root)")
    ap.add_argument("--real-support-from", default=None,
                    help="path template (with {seed}) holding an arm whose support.npz "
                         "is the same real support, for arms that predate support.npz")
    ap.add_argument("--shuf-support-from", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    summary = {}
    print(f"{'seed':>5s} {'n_tgt':>6s} {'raw d':>9s} {'decile excess':>14s} "
          f"{'ols resid':>10s} {'b (slope)':>10s}")
    print("-" * 62)
    for seed in a.seeds:
        sdir = os.path.join(a.root, f"s{seed}")
        real_dir = os.path.join(sdir, a.real_name)
        shuf_dir = os.path.join(sdir, a.shuf_name)
        r2a = load_r2(real_dir)
        r2b = load_r2(shuf_dir)
        rf = a.real_support_from.format(seed=seed) if a.real_support_from else None
        sf = a.shuf_support_from.format(seed=seed) if a.shuf_support_from else None
        set_real = support_idx(real_dir, rf)
        set_shuf = support_idx(shuf_dir, sf)
        real_i = np.array(sorted(set_real - set_shuf), dtype=int)
        shuf_i = np.array(sorted(set_shuf - set_real), dtype=int)
        rest_i = np.array(sorted(set(range(r2a.size)) - set_real - set_shuf), dtype=int)

        d = r2a - r2b
        base = r2b  # these genes carry no weight in the shuffled arm

        # --- 1. decile matching on baseline R2 -----------------------------
        edges = np.quantile(base[rest_i], np.linspace(0, 1, a.bins + 1))
        edges[-1] += 1e-12
        bin_of = np.clip(np.digitize(base, edges) - 1, 0, a.bins - 1)
        rest_mean = {b: float(np.nanmean(d[rest_i[bin_of[rest_i] == b]]))
                     for b in range(a.bins)
                     if np.any(bin_of[rest_i] == b)}
        excess = [d[i] - rest_mean[bin_of[i]] for i in real_i if bin_of[i] in rest_mean]
        excess = np.array(excess, dtype=float)
        excess = excess[np.isfinite(excess)]

        # --- 2. linear control on the unweighted genes ---------------------
        x = base[rest_i]
        y = d[rest_i]
        ok = np.isfinite(x) & np.isfinite(y)
        A = np.column_stack([np.ones(ok.sum()), x[ok]])
        coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
        pred = coef[0] + coef[1] * base[real_i]
        resid = d[real_i] - pred
        resid = resid[np.isfinite(resid)]

        summary[seed] = dict(
            n_target=int(real_i.size),
            raw_d=float(np.nanmean(d[real_i])),
            decile_excess_mean=float(excess.mean()),
            decile_excess_sd=float(excess.std(ddof=1)) if excess.size > 1 else 0.0,
            decile_excess_frac_pos=float((excess > 0).mean()),
            ols_resid_mean=float(resid.mean()),
            ols_slope=float(coef[1]),
            baseline_r2_real=float(np.nanmean(base[real_i])),
            baseline_r2_rest=float(np.nanmean(base[rest_i])),
        )
        s = summary[seed]
        print(f"{seed:5d} {s['n_target']:6d} {s['raw_d']:+9.5f} "
              f"{s['decile_excess_mean']:+14.5f} {s['ols_resid_mean']:+10.5f} "
              f"{s['ols_slope']:+10.3f}")

    vals = [v["decile_excess_mean"] for v in summary.values()]
    resid = [v["ols_resid_mean"] for v in summary.values()]
    print("-" * 62)
    print(f"decile-matched excess: {np.mean(vals):+.5f} "
          f"+/- {np.std(vals, ddof=1) if len(vals) > 1 else 0:.5f}  "
          f"({sum(1 for v in vals if v > 0)}/{len(vals)} positive)")
    print(f"OLS residual:          {np.mean(resid):+.5f} "
          f"+/- {np.std(resid, ddof=1) if len(resid) > 1 else 0:.5f}  "
          f"({sum(1 for v in resid if v > 0)}/{len(resid)} positive)")

    p = a.out or os.path.join(a.root, "pergene_robust.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[robust] wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
