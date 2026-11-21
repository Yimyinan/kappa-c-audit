#!/usr/bin/env python3
"""prior_delivery.run_empty_e45 -- the matched-epoch no-prior reference.

Part 8 registered this as a contingent step and its trigger fired
(`d_m(e45)/d_m(e15) = 1.26 > 1.2`).

Why it is needed: every `empty` arm in the project was trained at 45 total epochs
(30 unweighted + 15 weighted). The e45 arms run 75. Comparing a 75-epoch weighted
model against a 45-epoch unweighted one would credit the prior with 30 epochs of
extra training, which is exactly the confound this project keeps catching. So the
geometry question at this budget needs an empty trained at 75 epochs, on the same
cells, seed and schedule as the e45 arms.

Three arms, ~42 min each. Audits are written to a separate file so the
e15-referenced `audit_results.json` of each seed survives.
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def trained_at(arm_dir, epoch_initial, epoch_iter):
    """True only for an empty arm trained with this exact schedule.

    A bare `latent.npz` check is not enough: `run_dose.py` copies the e15 reference
    into every seed directory it creates, so the file exists before this script
    runs and has the wrong schedule. The first version of this guard did exactly
    that and silently audited 45-epoch arms against a 45-epoch baseline.
    """
    lp = os.path.join(arm_dir, "latent.npz")
    mp = os.path.join(arm_dir, "arm_meta.json")
    if not (os.path.exists(lp) and os.path.exists(mp)):
        return False
    try:
        m = json.load(open(mp, encoding="utf-8"))
    except Exception:
        return False
    return (m.get("arm") == "empty"
            and int(m.get("epoch_initial", -1)) == epoch_initial
            and int(m.get("epoch_iter", -1)) == epoch_iter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"F:\UNAGI_Project\prior_delivery_runs\dose20_e45")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    ap.add_argument("--per-stage", type=int, default=2400)
    ap.add_argument("--epoch-initial", type=int, default=30)
    ap.add_argument("--epoch-iter", type=int, default=45)
    ap.add_argument("--max-iter", type=int, default=2)
    ap.add_argument("--edge-cutoff", type=float, default=0.3)
    ap.add_argument("--cpo", choices=["on", "off"], default="off")
    ap.add_argument("--idrem", default=r"F:\UNAGI_Project\idrem")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    log_path = os.path.join(a.root, "empty_e45.log")
    log = open(log_path, "a", encoding="utf-8")
    status = {}

    for seed in a.seeds:
        sdir = os.path.join(a.root, f"s{seed}")
        # NOT `empty`: that directory already holds the 45-epoch reference that
        # run_dose.py copied in from the e15 run, and reusing it silently would
        # make this whole part measure 15 weighted epochs instead of 45.
        edir = os.path.join(sdir, "empty_e45")
        if trained_at(edir, a.epoch_initial, a.epoch_iter):
            print(f"[empty-e45] s{seed} already trained at "
                  f"{a.epoch_initial}+{a.epoch_iter}; reusing", file=log, flush=True)
            status[str(seed)] = dict(reused=True)
        else:
            if os.path.isdir(edir) and os.path.exists(os.path.join(edir, "latent.npz")):
                print(f"[empty-e45] s{seed}: existing model has the wrong schedule; "
                      f"retraining", file=log, flush=True)
            cmd = [a.python, "-u", os.path.join(HERE, "run_arm.py"),
                   "--arm", "empty", "--workdir", edir,
                   "--per-stage", str(a.per_stage),
                   "--max-iter", str(a.max_iter),
                   "--epoch-initial", str(a.epoch_initial),
                   "--epoch-iter", str(a.epoch_iter),
                   "--edge-cutoff", str(a.edge_cutoff),
                   "--cpo", a.cpo, "--seed", str(seed), "--idrem", a.idrem]
            print(f"\n===== s{seed}/empty_e45 (no prior, {a.epoch_initial}+{a.epoch_iter} "
                  f"epochs) @ {time.strftime('%H:%M:%S')} =====", file=log, flush=True)
            t0 = time.time()
            p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            dt = (time.time() - t0) / 60
            print(f"===== s{seed}/empty_e45 rc={p.returncode} in {dt:.1f} min =====",
                  file=log, flush=True)
            status[str(seed)] = dict(returncode=p.returncode, minutes=round(dt, 1))
            if p.returncode != 0:
                break

        cmd = [a.python, "-u", os.path.join(HERE, "audit_arms.py"),
               "--root", sdir, "--ref", "empty_e45",
               "--arms", "empty_e45", "real", "shuffled",
               "--out", os.path.join(sdir, "audit_vs_e45empty.json")]
        print(f"\n===== s{seed} audit vs e45 empty @ {time.strftime('%H:%M:%S')} =====",
              file=log, flush=True)
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        print(f"===== s{seed} audit rc={p.returncode} =====", file=log, flush=True)

    with open(os.path.join(a.root, "empty_e45_status.json"), "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    log.close()
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
