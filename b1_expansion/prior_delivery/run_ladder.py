#!/usr/bin/env python3
"""prior_delivery.run_ladder -- dose ladder above the rank ceiling.

The rank form of the weight (`1 + 1/sqrt(rank)`) caps the per-gene ratio at
`2 / (1 + 1/sqrt(N))`, i.e. ~1.98 for 20,527 genes, and the observed response was
already saturated at a 1.09x dose. So nothing inside that ceiling can test whether
compression appears at a *higher* dose.

This driver switches the weighting to `direct`, where the stored value is used as
the per-gene multiplier on the reconstruction term, and walks the dose up:

    d1    factor 1  -- identical to no weighting; the reference, mode-consistent
    d2    factor 2  -- the rank ceiling, reached directly
    d5    factor 5
    d20   factor 20

All arms keep the same 203-gene support, the same cells, seed and schedule, so the
only thing that moves is the factor. Prediction under test: if paper1's dose-ceiling
reading is right, the decoder spectrum stays flat at d2 and only compresses once the
factor clears ~2x, with the latent spectrum moving first.
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"F:\UNAGI_Project\prior_delivery_runs\ladder1")
    ap.add_argument("--levels", nargs="*", type=float, default=[1.0, 2.0, 5.0, 20.0])
    ap.add_argument("--per-stage", type=int, default=2400)
    ap.add_argument("--epoch-initial", type=int, default=30)
    ap.add_argument("--epoch-iter", type=int, default=15)
    ap.add_argument("--max-iter", type=int, default=2)
    ap.add_argument("--edge-cutoff", type=float, default=0.3)
    ap.add_argument("--cpo", choices=["on", "off"], default="off")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--idrem", default=r"F:\UNAGI_Project\idrem")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    os.makedirs(a.root, exist_ok=True)
    names = [f"d{lv:g}" for lv in a.levels]
    log = open(os.path.join(a.root, "run.log"), "a", encoding="utf-8")
    status = {}

    for lv, name in zip(a.levels, names):
        wd = os.path.join(a.root, name)
        cmd = [a.python, "-u", os.path.join(HERE, "run_arm.py"),
               "--arm", "binary", "--workdir", wd,
               "--per-stage", str(a.per_stage),
               "--max-iter", str(a.max_iter),
               "--epoch-initial", str(a.epoch_initial),
               "--epoch-iter", str(a.epoch_iter),
               "--edge-cutoff", str(a.edge_cutoff),
               "--cpo", a.cpo,
               "--seed", str(a.seed),
               "--weight-mode", "direct",
               "--prior-level", str(lv),
               "--idrem", a.idrem]
        t0 = time.time()
        print(f"\n===== {name} (factor {lv:g}) @ {time.strftime('%H:%M:%S')} =====",
              file=log, flush=True)
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        status[name] = dict(factor=lv, returncode=p.returncode,
                            seconds=round(dt, 1))
        print(f"===== {name} rc={p.returncode} in {dt/60:.1f} min =====", file=log,
              flush=True)

    with open(os.path.join(a.root, "run_status.json"), "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    ok = [k for k, v in status.items() if v["returncode"] == 0]
    if len(ok) >= 2:
        print("\n===== audit =====", file=log, flush=True)
        cmd = [a.python, "-u", os.path.join(HERE, "audit_arms.py"),
               "--root", a.root, "--ref", names[0], "--arms", *names]
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        print(f"===== audit rc={p.returncode} =====", file=log, flush=True)

    log.close()
    print(json.dumps(status, indent=2))
    return 0 if all(v["returncode"] == 0 for v in status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
