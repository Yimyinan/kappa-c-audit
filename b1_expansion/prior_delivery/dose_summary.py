#!/usr/bin/env python3
"""prior_delivery.dose_summary -- read the dose ladder off disk and apply the
decision rules that were registered before each level was trained.

Levels are named explicitly, because the arms were not produced by one driver:

    5x   prior_delivery_runs/support      k203       vs shuffleflat   (Part 4/5)
    20x  prior_delivery_runs/dose20       real       vs shuffled      (Part 6)
    40x  prior_delivery_runs/dose40       real       vs shuffled      (Part 7)

Two statistics per level, both per-seed and both paired on gene identity:

    raw d      mean_centered(real targets) - mean_centered(shuffled targets)
    matched d  the same after matching each target to unweighted genes in its own
               baseline-R2 decile (pergene_robust.py). Stricter; used for verdicts.

Registered rules, both fixed before their data existed:

    Part 6 (20x vs 5x):  supported  iff  mean(d20-d5) > 0
                                and  mean(d20-d5) > sd(d20-d5)
                                and  d20 > d5 in >= 2 of 3 seeds
    Part 7 (40x vs 20x): scale    iff  d_m(40)/d_m(20) > 1.5
                         flat     iff  0.85 <= ratio <= 1.5
                         collapse iff ratio < 0.85
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import statistics as st

PAIR_FILES = ("pergene_fit_real_vs_shuffled.json", "pergene_k203_vs_shuffleflat.json")
ROBUST_FILES = ("pergene_robust.json", "pergene_robust_5x.json")


def read_pair(seed_dir):
    for name in PAIR_FILES:
        p = os.path.join(seed_dir, name)
        if not os.path.exists(p):
            continue
        dat = json.load(open(p, encoding="utf-8"))
        real = next((r for r in dat["rows"] if r["label"].startswith("real")), None)
        shuf = next((r for r in dat["rows"] if r["label"].startswith("shuffled")), None)
        if not (real and shuf):
            continue
        return dict(real=real["mean_centered"], shuffled=shuf["mean_centered"],
                    d=real["mean_centered"] - shuf["mean_centered"],
                    frac_pos=real["frac_pos_centered"], n=real["n"],
                    aggregate_ratio=dat.get("aggregate_ratio"))
    return None


def read_matched(root, seed):
    """The decile-matched excess for this seed, if a robustness file exists."""
    for name in ROBUST_FILES:
        p = os.path.join(root, name)
        if not os.path.exists(p):
            continue
        dat = json.load(open(p, encoding="utf-8"))
        row = dat.get(str(seed)) or dat.get(seed)
        if row:
            return row.get("decile_excess_mean")
    return None


def read_audit(seed_dir, arm_names):
    p = os.path.join(seed_dir, "audit_results.json")
    if not os.path.exists(p):
        return None
    rows = json.load(open(p, encoding="utf-8"))["rows"]
    for r in rows:
        if r["arm"] in arm_names:
            return dict(kappa_C_W=r["kappa_C_W"], kappa_C_Z=r["kappa_C_Z"],
                        C_percent=r["C_percent"], eff_rank=r["eff_rank"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="*", default=[
        r"5=F:\UNAGI_Project\prior_delivery_runs\support",
        r"20=F:\UNAGI_Project\prior_delivery_runs\dose20",
        r"40=F:\UNAGI_Project\prior_delivery_runs\dose40",
    ], help="LEVEL=ROOT pairs, in ascending dose order")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    ap.add_argument("--real-arm", default="real",
                    help="arm dir name used for the audit lookup above 5x")
    ap.add_argument("--out", default=r"F:\UNAGI_Project\prior_delivery_runs\dose_ladder.json")
    a = ap.parse_args()

    levels = []
    for spec in a.levels:
        lv, root = spec.split("=", 1)
        levels.append((float(lv), root))
    levels.sort()

    data = {}
    for lv, root in levels:
        per = {}
        for s in a.seeds:
            sdir = os.path.join(root, f"s{s}")
            arm = "k203" if lv == 5.0 else a.real_arm
            per[s] = dict(pair=read_pair(sdir), matched=read_matched(root, s),
                          audit=read_audit(sdir, (arm,)))
        data[lv] = per

    hdr = (f"{'level':>6s} {'seed':>5s} {'raw d':>9s} {'matched d':>10s} "
           f"{'kCW':>7s} {'kCZ':>7s} {'agg ratio':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for lv, per in data.items():
        for s, t in per.items():
            p, m, au = t["pair"], t["matched"], t["audit"]
            raw = f"{p['d']:+9.5f}" if p else f"{'--':>9s}"
            mat = f"{m:+10.5f}" if m is not None else f"{'--':>10s}"
            kcw = f"{au['kappa_C_W']:7.4f}" if au else f"{'--':>7s}"
            kcz = f"{au['kappa_C_Z']:7.4f}" if au else f"{'--':>7s}"
            agg = (f"{p['aggregate_ratio']:10.6f}"
                   if p and p.get("aggregate_ratio") else f"{'--':>10s}")
            print(f"{lv:6.0f} {s:5d} {raw} {mat} {kcw} {kcz} {agg}")
        print("-" * len(hdr))

    def col(lv, key):
        out = []
        for s in a.seeds:
            v = data.get(lv, {}).get(s, {}).get(key)
            if isinstance(v, dict):
                v = v.get("d")
            if v is not None:
                out.append(v)
        return out

    print("\nacross-seed means")
    print(f"{'level':>6s} {'raw d':>18s} {'matched d':>18s} {'kCW':>16s} {'kCZ':>16s}")
    means = {}
    for lv in data:
        raw = col(lv, "pair")
        mat = [v for v in col(lv, "matched") if v is not None]
        kcw = [t["audit"]["kappa_C_W"] for t in data[lv].values() if t["audit"]]
        kcz = [t["audit"]["kappa_C_Z"] for t in data[lv].values() if t["audit"]]

        def fmt(v):
            if not v:
                return f"{'--':>18s}"
            sd = st.stdev(v) if len(v) > 1 else 0.0
            return f"{st.mean(v):+.5f} +/- {sd:.5f}"

        means[lv] = dict(raw=raw, matched=mat, kcw=kcw, kcz=kcz)
        print(f"{lv:6.0f} {fmt(raw):>18s} {fmt(mat):>18s} {fmt(kcw):>16s} {fmt(kcz):>16s}")

    verdicts = {}
    # ---- Part 6 rule: 20x vs 5x on the raw estimator ---------------------
    if 5.0 in means and 20.0 in means and len(means[5.0]["raw"]) == len(means[20.0]["raw"]):
        d5, d20 = means[5.0]["raw"], means[20.0]["raw"]
        step = [b - a_ for a_, b in zip(d5, d20)]
        m, sd = st.mean(step), (st.stdev(step) if len(step) > 1 else 0.0)
        ok = m > 0 and m > sd and sum(1 for x in step if x > 0) >= 2
        verdicts["part6_20_vs_5"] = dict(step_mean=m, step_sd=sd,
                                         n_positive=sum(1 for x in step if x > 0),
                                         verdict="supported" if ok else "unresolved")
        print(f"\nPart 6 rule (20x vs 5x): step {m:+.5f} +/- {sd:.5f}, "
              f"{sum(1 for x in step if x > 0)}/{len(step)} positive -> "
              f"{verdicts['part6_20_vs_5']['verdict'].upper()}")

    # ---- Part 7 rule: 40x vs 20x on the matched estimator ----------------
    if 20.0 in means and 40.0 in means and means[20.0]["matched"] and means[40.0]["matched"]:
        r20, r40 = st.mean(means[20.0]["matched"]), st.mean(means[40.0]["matched"])
        ratio = r40 / r20 if r20 else float("nan")
        if ratio > 1.5:
            v = "still scaling"
        elif ratio >= 0.85:
            v = "saturated"
        else:
            v = "collapsed"
        verdicts["part7_40_vs_20"] = dict(d_m20=r20, d_m40=r40, ratio=ratio, verdict=v)
        print(f"Part 7 rule (40x vs 20x, matched): {r40:+.5f} / {r20:+.5f} "
              f"= {ratio:.2f} -> {v.upper()}")

    out = dict(levels=[lv for lv, _ in levels],
               roots={str(lv): root for lv, root in levels},
               means={str(k): dict(raw=v["raw"], matched=v["matched"],
                                   kcw=v["kcw"], kcz=v["kcz"])
                      for k, v in means.items()},
               verdicts=verdicts)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[dose] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
