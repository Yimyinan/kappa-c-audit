# -*- coding: utf-8 -*-
"""Experiment B -- same-data single-factor factorial.

The thesis's 3.12 says the latent-side movement is "content-agnostic but associated with
the injection event", and that which injection-side factor causes it cannot be identified
from the retained artefacts, because tissue / dataset / batch move together. This runs the
design that separates them: one dataset, one pipeline, one budget, two factors.

    factor 1  prior arm     binary (real prior) | shuffleflat (degree-matched shuffled prior) | empty
    factor 2  encoder       GCN on | GCN off (plain MLP encoder)

The graph axis is UNAGI's own `GCN` flag (Graph_VAE vs Plain_VAE); nothing in the vendored
code is modified.

Why `empty` doubles as "no prior": run_arm calls setup_data(disable_tf_prior=(arm=="empty")),
so it runs with no prior mechanism at all; the thesis already states that the empty-vector
carrier and the no-prior arm are functionally identical in this source.

GCN-on cells already exist at the matched operating point (prior_delivery_runs/dose20_e45,
45 epochs, 12,000 cells, 3 seeds), so only the nine GCN-off cells are trained here.

Resumable: a cell whose arm_meta.json exists is skipped.

Usage:
    py -3.13 b_factorial.py --dry-run
    py -3.13 b_factorial.py                 # all nine cells, ~45 min each
    py -3.13 b_factorial.py --arms binary --seeds 42 --max-iter 1 --epoch-initial 2 --epoch-iter 2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = r"F:\UNAGI_Project"
RUNNER = os.path.join(ROOT, "b1_expansion", "prior_delivery", "run_arm.py")
OUTROOT = os.path.join(ROOT, "prior_delivery_runs", "b_factorial_gcn_off")

# matched to prior_delivery_runs/dose20_e45/s42/{real,empty_e45,shuffled}/arm_meta.json
CELLS = {
    "binary":      dict(weight_mode="direct", prior_level=20.0, support_k=203),
    "shuffleflat": dict(weight_mode="direct", prior_level=20.0, support_k=0),
    "empty":       dict(),
}
EPOCH_INITIAL = 30
EPOCH_ITER = 45
MAX_ITER = 2
PER_STAGE = 2400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="binary,shuffleflat,empty")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--gcn", choices=["on", "off"], default="off")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-iter", type=int, default=MAX_ITER)
    ap.add_argument("--epoch-initial", type=int, default=EPOCH_INITIAL)
    ap.add_argument("--epoch-iter", type=int, default=EPOCH_ITER)
    ap.add_argument("--per-stage", type=int, default=PER_STAGE)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    todo, done = [], []
    for arm in arms:
        for seed in seeds:
            wd = os.path.join(OUTROOT, f"s{seed}", arm)
            if os.path.exists(os.path.join(wd, "arm_meta.json")):
                done.append((arm, seed))
            else:
                todo.append((arm, seed))

    print(f"cells: {len(todo) + len(done)} total, {len(done)} done, {len(todo)} to run "
          f"(gcn={a.gcn})")
    for arm, seed in todo:
        wd = os.path.join(OUTROOT, f"s{seed}", arm)
        cmd = [sys.executable, RUNNER, "--arm", arm, "--workdir", wd,
               "--seed", str(seed), "--shuffle-seed", "7",
               "--max-iter", str(a.max_iter), "--epoch-initial", str(a.epoch_initial),
               "--epoch-iter", str(a.epoch_iter), "--per-stage", str(a.per_stage),
               "--cpo", "off", "--gcn", a.gcn, "--device", a.device]
        for k, v in CELLS[arm].items():
            cmd += [f"--{k.replace('_', '-')}", str(v)]
        print("  " + " ".join(cmd))
        if a.dry_run:
            continue
        os.makedirs(wd, exist_ok=True)
        log = open(os.path.join(wd, "run.log"), "w", encoding="utf-8")
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        log.close()
        print(f"  -> {arm} s{seed} rc={rc}", flush=True)
        if rc != 0:
            print(f"  !! failed, see {wd}/run.log")
            return 1
    print("done" if not a.dry_run else "dry run only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
