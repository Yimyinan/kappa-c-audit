#!/usr/bin/env python3
"""prior_delivery.run_all -- run the four arms in order, then audit.

Order matters: `empty` and `graded` come first because they answer the primary
question (does delivering the prior move the spectra at all), and `binary` /
`shuffle` are the content controls. If the run is interrupted, the primary
contrast is already on disk.

Each arm is a separate process so that one arm's failure does not take the
others down; the driver records the exit status in run_status.json.
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["empty", "graded", "binary", "shuffle"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"F:\UNAGI_Project\prior_delivery_runs\r1")
    ap.add_argument("--per-stage", type=int, default=2400)
    ap.add_argument("--epoch-initial", type=int, default=30)
    ap.add_argument("--epoch-iter", type=int, default=15)
    ap.add_argument("--max-iter", type=int, default=2)
    ap.add_argument("--edge-cutoff", type=float, default=0.3)
    ap.add_argument("--cpo", choices=["on", "off"], default="off")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--idrem", default=r"F:\UNAGI_Project\idrem")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()

    os.makedirs(a.root, exist_ok=True)
    log = open(os.path.join(a.root, "run.log"), "a", encoding="utf-8")
    status = {}

    arms = a.only or ORDER
    for arm in arms:
        wd = os.path.join(a.root, arm)
        cmd = [a.python, "-u", os.path.join(HERE, "run_arm.py"),
               "--arm", arm, "--workdir", wd,
               "--per-stage", str(a.per_stage),
               "--max-iter", str(a.max_iter),
               "--epoch-initial", str(a.epoch_initial),
               "--epoch-iter", str(a.epoch_iter),
               "--edge-cutoff", str(a.edge_cutoff),
               "--cpo", a.cpo,
               "--seed", str(a.seed),
               "--idrem", a.idrem]
        t0 = time.time()
        print(f"\n===== {arm} @ {time.strftime('%H:%M:%S')} =====", file=log, flush=True)
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        status[arm] = dict(returncode=p.returncode, seconds=round(dt, 1),
                           started=time.strftime("%Y-%m-%d %H:%M:%S"))
        print(f"===== {arm} rc={p.returncode} in {dt/60:.1f} min =====", file=log, flush=True)

    with open(os.path.join(a.root, "run_status.json"), "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    ok = [k for k, v in status.items() if v["returncode"] == 0]
    if len(ok) >= 2:
        print("\n===== audit =====", file=log, flush=True)
        cmd = [a.python, "-u", os.path.join(HERE, "audit_arms.py"), "--root", a.root]
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        print(f"===== audit rc={p.returncode} =====", file=log, flush=True)

    log.close()
    print(json.dumps(status, indent=2))
    return 0 if all(v["returncode"] == 0 for v in status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
