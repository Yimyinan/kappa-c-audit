#!/usr/bin/env python3
"""prior_delivery.run_seeds -- replicate the four-arm contrast across seeds.

Seed 42 already exists in `prior_delivery_runs/r2`, so only 43 and 44 are run here
and the three are aggregated together. Every seed trains all four arms with the
same cells, budget and edge cutoff; only the prior differs within a seed, and only
the seed differs between seeds.

The point is not a p-value. It is to find out whether the 0.028 gap between the
real 203-gene support and an in-degree-matched random support survives a change of
seed. With three seeds the honest summary is a range, which is what the aggregate
prints.
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["empty", "graded", "binary", "shuffle"]


def run_one(root, seed, a, log):
    wd = os.path.join(root, f"s{seed}")
    cmd = [a.python, "-u", os.path.join(HERE, "run_all.py"),
           "--root", wd,
           "--per-stage", str(a.per_stage),
           "--epoch-initial", str(a.epoch_initial),
           "--epoch-iter", str(a.epoch_iter),
           "--max-iter", str(a.max_iter),
           "--edge-cutoff", str(a.edge_cutoff),
           "--cpo", a.cpo,
           "--seed", str(seed),
           "--idrem", a.idrem]
    print(f"\n===== seed {seed} -> {wd} @ {time.strftime('%H:%M:%S')} =====",
          file=log, flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    print(f"===== seed {seed} rc={p.returncode} in {dt/60:.1f} min =====", file=log,
          flush=True)
    return p.returncode, dt


def aggregate(roots):
    """Collect per-seed audit rows into one table of (arm, seed, kappa) triples."""
    per = {}
    for seed, root in roots:
        p = os.path.join(root, "audit_results.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        for r in d["rows"]:
            per.setdefault(r["arm"], {})[seed] = r
    return per


def summarise(per):
    import statistics as st

    def col(rows, key):
        v = [r[key] for r in rows]
        return v

    print("\n" + "=" * 88)
    print("across-seed summary (kappa against that seed's own empty arm)")
    print("=" * 88)
    hdr = (f"{'arm':10s} {'n':>3s} {'kCW mean':>9s} {'sd':>7s} {'range':>17s} "
           f"{'kCZ mean':>9s} {'sd':>7s} {'range':>17s}")
    print(hdr)
    print("-" * len(hdr))
    out = {}
    for arm in ARMS:
        rows = list(per.get(arm, {}).values())
        if not rows:
            continue
        kcw = col(rows, "kappa_C_W")
        kcz = col(rows, "kappa_C_Z")
        out[arm] = dict(
            n=len(rows),
            kappa_C_W_mean=st.mean(kcw), kappa_C_W_sd=st.stdev(kcw) if len(kcw) > 1 else 0.0,
            kappa_C_W_min=min(kcw), kappa_C_W_max=max(kcw),
            kappa_C_Z_mean=st.mean(kcz), kappa_C_Z_sd=st.stdev(kcz) if len(kcz) > 1 else 0.0,
            kappa_C_Z_min=min(kcz), kappa_C_Z_max=max(kcz),
            per_seed={s: dict(kappa_C_W=r["kappa_C_W"], kappa_C_Z=r["kappa_C_Z"],
                              C_percent=r["C_percent"], eff_rank=r["eff_rank"])
                      for s, r in sorted(per[arm].items())},
        )
        o = out[arm]
        print(f"{arm:10s} {o['n']:3d} {o['kappa_C_W_mean']:9.4f} {o['kappa_C_W_sd']:7.4f} "
              f"[{o['kappa_C_W_min']:.4f},{o['kappa_C_W_max']:.4f}] "
              f"{o['kappa_C_Z_mean']:9.4f} {o['kappa_C_Z_sd']:7.4f} "
              f"[{o['kappa_C_Z_min']:.4f},{o['kappa_C_Z_max']:.4f}]")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"F:\UNAGI_Project\prior_delivery_runs\seeds")
    ap.add_argument("--seeds", nargs="*", type=int, default=[43, 44])
    ap.add_argument("--baseline-root", default=r"F:\UNAGI_Project\prior_delivery_runs\r2",
                    help="root already holding the seed-42 run")
    ap.add_argument("--baseline-seed", type=int, default=42)
    ap.add_argument("--per-stage", type=int, default=2400)
    ap.add_argument("--epoch-initial", type=int, default=30)
    ap.add_argument("--epoch-iter", type=int, default=15)
    ap.add_argument("--max-iter", type=int, default=2)
    ap.add_argument("--edge-cutoff", type=float, default=0.3)
    ap.add_argument("--cpo", choices=["on", "off"], default="off")
    ap.add_argument("--idrem", default=r"F:\UNAGI_Project\idrem")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    os.makedirs(a.root, exist_ok=True)
    log = open(os.path.join(a.root, "seeds.log"), "a", encoding="utf-8")
    status = {}
    for seed in a.seeds:
        rc, dt = run_one(a.root, seed, a, log)
        status[str(seed)] = dict(returncode=rc, minutes=round(dt / 60, 1))

    with open(os.path.join(a.root, "run_status.json"), "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    roots = [(a.baseline_seed, a.baseline_root)] + \
            [(s, os.path.join(a.root, f"s{s}")) for s in a.seeds]
    per = aggregate(roots)
    summary = summarise(per)
    with open(os.path.join(a.root, "seeds_summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(seeds=[s for s, _ in roots], summary=summary), f, indent=2)
    log.close()
    print(json.dumps(status, indent=2))
    return 0 if all(v["returncode"] == 0 for v in status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
