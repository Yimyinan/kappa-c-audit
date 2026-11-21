#!/usr/bin/env python3
"""prior_delivery.run_b1 -- the B1 geometry-prior runs (see b1_expansion/B1_DESIGN.md).

Two phases, deliberately separated:

    probe     one seed, real basis, a small lambda ladder. Purpose is only to pick
              a lambda that leaves reconstruction alive while actually pulling the
              decoder Jacobian into the module span. Not evidence.
    contrast  3 seeds x {real, perm} at that lambda. This is the experiment.

The control is a row permutation of the basis: identical rank, identical column
norms, identical covered-gene count, `C^T C` identical to 1e-13 — and the induced
gene-gene similarity completely different (cosine 0.17). So any difference between
the arms cannot come from how strong the constraint is.

B1 arms are `--arm empty` by construction: B1 replaces the likelihood-weight
operator, it does not stack on top of it.
"""
from __future__ import annotations

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, log, tag):
    t0 = time.time()
    print(f"\n===== {tag} @ {time.strftime('%H:%M:%S')} =====", file=log, flush=True)
    print(" ".join(cmd), file=log, flush=True)
    p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    dt = (time.time() - t0) / 60
    print(f"===== {tag} rc={p.returncode} in {dt:.1f} min =====", file=log, flush=True)
    return p.returncode, dt


def already(arm_dir, basis, lam, epoch_initial, epoch_iter):
    lp = os.path.join(arm_dir, "latent.npz")
    mp = os.path.join(arm_dir, "arm_meta.json")
    if not (os.path.exists(lp) and os.path.exists(mp)):
        return False
    try:
        m = json.load(open(mp, encoding="utf-8"))
    except Exception:
        return False
    b = m.get("basis") or {}
    return (b.get("kind") == basis
            and abs(float(b.get("lambda_align", -1)) - lam) < 1e-9
            and int(m.get("epoch_initial", -1)) == epoch_initial
            and int(m.get("epoch_iter", -1)) == epoch_iter)


def base_cmd(a, workdir, seed, basis, lam):
    return [a.python, "-u", os.path.join(HERE, "run_arm.py"),
            "--arm", "empty", "--workdir", workdir,
            "--per-stage", str(a.per_stage),
            "--max-iter", str(a.max_iter),
            "--epoch-initial", str(a.epoch_initial),
            "--epoch-iter", str(a.epoch_iter),
            "--edge-cutoff", str(a.edge_cutoff),
            "--cpo", a.cpo, "--seed", str(seed),
            "--basis", basis, "--lambda-align", str(lam),
            "--basis-file", a.basis_file,
            "--idrem", a.idrem]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"F:\UNAGI_Project\prior_delivery_runs\b1")
    ap.add_argument("--phase", choices=["probe", "contrast", "both"], default="probe")
    ap.add_argument("--lambdas", nargs="*", type=float, default=[0.1, 1.0, 10.0])
    ap.add_argument("--lambda-final", type=float, default=None,
                    help="lambda used by the contrast phase; required for phase=contrast")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    ap.add_argument("--probe-seeds", nargs="*", type=int, default=[42])
    ap.add_argument("--basis-file",
                    default=r"F:\UNAGI_Project\prior_delivery_runs\b1\modules_p32.npz")
    ap.add_argument("--empty-root", default=r"F:\UNAGI_Project\prior_delivery_runs\support",
                    help="where the 30+15 epoch no-prior arms live, for the audit ref")
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
    log = open(os.path.join(a.root, "run.log"), "a", encoding="utf-8")
    status = {}

    if a.phase in ("probe", "both"):
        for seed in a.probe_seeds:
            for lam in a.lambdas:
                tag = f"probe/real_l{lam:g}"
                wd = os.path.join(a.root, "probe", f"s{seed}_real_l{lam:g}")
                if already(wd, "real", lam, a.epoch_initial, a.epoch_iter):
                    print(f"[b1] {tag} s{seed} exists; reusing", file=log, flush=True)
                    status[f"{tag}/s{seed}"] = dict(reused=True)
                    continue
                rc, dt = run(base_cmd(a, wd, seed, "real", lam), log, f"{tag} s{seed}")
                status[f"{tag}/s{seed}"] = dict(returncode=rc, minutes=round(dt, 1))
                if rc != 0:
                    break

    if a.phase in ("contrast", "both"):
        if a.lambda_final is None:
            print("[b1] FATAL: --lambda-final required for the contrast phase")
            return 2
        for seed in a.seeds:
            sdir = os.path.join(a.root, "contrast", f"s{seed}")
            os.makedirs(sdir, exist_ok=True)
            ref = os.path.join(a.empty_root, f"s{seed}", "empty")
            dref = os.path.join(sdir, "empty")
            if not os.path.exists(os.path.join(dref, "latent.npz")):
                if not os.path.exists(os.path.join(ref, "latent.npz")):
                    print(f"[b1] no reference empty for seed {seed}; skipping",
                          file=log, flush=True)
                    continue
                os.makedirs(dref, exist_ok=True)
                shutil.copy2(os.path.join(ref, "latent.npz"), dref)
                if os.path.exists(os.path.join(ref, "arm_meta.json")):
                    shutil.copy2(os.path.join(ref, "arm_meta.json"), dref)

            for basis in ("real", "perm"):
                wd = os.path.join(sdir, basis)
                if already(wd, basis, a.lambda_final, a.epoch_initial, a.epoch_iter):
                    print(f"[b1] s{seed}/{basis} exists; reusing", file=log, flush=True)
                    status[f"contrast/s{seed}/{basis}"] = dict(reused=True)
                    continue
                rc, dt = run(base_cmd(a, wd, seed, basis, a.lambda_final), log,
                             f"contrast s{seed}/{basis} l={a.lambda_final:g}")
                status[f"contrast/s{seed}/{basis}"] = dict(returncode=rc,
                                                           minutes=round(dt, 1))
                if rc != 0:
                    break

            cmd = [a.python, "-u", os.path.join(HERE, "audit_arms.py"),
                   "--root", sdir, "--ref", "empty",
                   "--arms", "empty", "real", "perm"]
            run(cmd, log, f"audit s{seed}")

    with open(os.path.join(a.root, "run_status.json"), "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    log.close()
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
