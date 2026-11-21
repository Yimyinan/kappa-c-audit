#!/usr/bin/env python3
"""prior_delivery.run_support -- the support ladder.

Parts 1-3 showed that dose does nothing when the prior covers 203 of 20,527 genes
(1.1% of the reconstruction mass). The arithmetic says that is not a dose property
but a support property: kappa-C(W) is an aggregate over all genes, so reweighting 1%
of the sum cannot move it however hard you push.

This driver tests that directly, by widening a *real* TRRUST prior while holding the
factor fixed at 5x:

    k203    the curated 11-TF core (the same support as Parts 1 and 2)
    k1000   core + highest in-degree TRRUST targets, to 1,000 genes (4.9%)
    k2027   every TRRUST target present in the dataset (9.9%) -- the natural ceiling

Only 2,027 of the 20,527 genes are targets of any TRRUST TF, so 5,000 is not
available with a real prior. Weights are rescaled so the total reconstruction mass
stays fixed: the manipulation is *concentration*, not a global loss rescale.

Each seed reuses the `empty` arm already trained for that seed as its reference, so
only three new runs per seed are needed. Prediction under test: kappa-C(W) should
begin to fall only once the support reaches a few percent of the transcriptome.
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
RUNGS = [203, 1000, 2027]


def reference_for(seed, baseline_root, seeds_root):
    """Where the already-trained `empty` arm for this seed lives."""
    if seed == 42 and baseline_root:
        p = os.path.join(baseline_root, "empty")
        if os.path.exists(os.path.join(p, "latent.npz")):
            return p
    p = os.path.join(seeds_root, f"s{seed}", "empty")
    return p if os.path.exists(os.path.join(p, "latent.npz")) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"F:\UNAGI_Project\prior_delivery_runs\support")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    ap.add_argument("--k", nargs="*", type=int, default=RUNGS)
    ap.add_argument("--level", type=float, default=5.0)
    ap.add_argument("--per-stage", type=int, default=2400)
    ap.add_argument("--epoch-initial", type=int, default=30)
    ap.add_argument("--epoch-iter", type=int, default=15)
    ap.add_argument("--max-iter", type=int, default=2)
    ap.add_argument("--edge-cutoff", type=float, default=0.3)
    ap.add_argument("--cpo", choices=["on", "off"], default="off")
    ap.add_argument("--idrem", default=r"F:\UNAGI_Project\idrem")
    ap.add_argument("--baseline-root", default=r"F:\UNAGI_Project\prior_delivery_runs\r2")
    ap.add_argument("--seeds-root", default=r"F:\UNAGI_Project\prior_delivery_runs\seeds")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    os.makedirs(a.root, exist_ok=True)
    log = open(os.path.join(a.root, "run.log"), "a", encoding="utf-8")
    status = {}
    names = [f"k{k}" for k in a.k]

    for seed in a.seeds:
        sdir = os.path.join(a.root, f"s{seed}")
        os.makedirs(sdir, exist_ok=True)

        ref = reference_for(seed, a.baseline_root, a.seeds_root)
        if ref is None:
            print(f"[support] no reference empty for seed {seed}; skipping", file=log, flush=True)
            status[f"s{seed}"] = dict(error="no reference")
            continue
        dref = os.path.join(sdir, "empty")
        if not os.path.exists(os.path.join(dref, "latent.npz")):
            os.makedirs(dref, exist_ok=True)
            shutil.copy2(os.path.join(ref, "latent.npz"), dref)
            m = os.path.join(ref, "arm_meta.json")
            if os.path.exists(m):
                shutil.copy2(m, dref)
        print(f"\n===== seed {seed} reference <- {ref} =====", file=log, flush=True)

        for k, name in zip(a.k, names):
            cmd = [a.python, "-u", os.path.join(HERE, "run_arm.py"),
                   "--arm", "binary", "--workdir", os.path.join(sdir, name),
                   "--per-stage", str(a.per_stage),
                   "--max-iter", str(a.max_iter),
                   "--epoch-initial", str(a.epoch_initial),
                   "--epoch-iter", str(a.epoch_iter),
                   "--edge-cutoff", str(a.edge_cutoff),
                   "--cpo", a.cpo,
                   "--seed", str(seed),
                   "--weight-mode", "direct",
                   "--prior-level", str(a.level),
                   "--support-k", str(k),
                   "--idrem", a.idrem]
            t0 = time.time()
            print(f"\n===== s{seed}/{name} (k={k}, {a.level:g}x) @ "
                  f"{time.strftime('%H:%M:%S')} =====", file=log, flush=True)
            p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            dt = time.time() - t0
            status[f"s{seed}/{name}"] = dict(k=k, returncode=p.returncode,
                                             minutes=round(dt / 60, 1))
            print(f"===== s{seed}/{name} rc={p.returncode} in {dt/60:.1f} min =====",
                  file=log, flush=True)

        cmd = [a.python, "-u", os.path.join(HERE, "audit_arms.py"),
               "--root", sdir, "--ref", "empty", "--arms", "empty", *names]
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        print(f"===== s{seed} audit rc={p.returncode} =====", file=log, flush=True)

    with open(os.path.join(a.root, "run_status.json"), "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    # ---- across-seed summary ------------------------------------------------
    per = {}
    for seed in a.seeds:
        p = os.path.join(a.root, f"s{seed}", "audit_results.json")
        if not os.path.exists(p):
            continue
        for r in json.load(open(p, encoding="utf-8"))["rows"]:
            per.setdefault(r["arm"], {})[seed] = r

    summary = {}
    print("\n" + "=" * 82, file=log, flush=True)
    print("support ladder, across seeds", file=log, flush=True)
    print("=" * 82, file=log, flush=True)
    hdr = (f"{'arm':8s} {'n':>3s} {'support':>8s} {'kCW mean':>9s} {'sd':>7s} "
           f"{'kCZ mean':>9s} {'sd':>7s} {'effrank':>8s}")
    print(hdr, file=log, flush=True)
    for arm in ["empty", *names]:
        rows = list(per.get(arm, {}).values())
        if not rows:
            continue
        kcw = [r["kappa_C_W"] for r in rows]
        kcz = [r["kappa_C_Z"] for r in rows]
        sup = rows[0]["n_boosted"]
        summary[arm] = dict(n=len(rows), support=sup,
                            kappa_C_W_mean=st.mean(kcw),
                            kappa_C_W_sd=st.stdev(kcw) if len(kcw) > 1 else 0.0,
                            kappa_C_Z_mean=st.mean(kcz),
                            kappa_C_Z_sd=st.stdev(kcz) if len(kcz) > 1 else 0.0,
                            eff_rank=st.mean([r["eff_rank"] for r in rows]),
                            per_seed={s: dict(kappa_C_W=r["kappa_C_W"],
                                              kappa_C_Z=r["kappa_C_Z"],
                                              C_percent=r["C_percent"])
                                      for s, r in sorted(per[arm].items())})
        o = summary[arm]
        print(f"{arm:8s} {o['n']:3d} {sup:8d} {o['kappa_C_W_mean']:9.4f} "
              f"{o['kappa_C_W_sd']:7.4f} {o['kappa_C_Z_mean']:9.4f} "
              f"{o['kappa_C_Z_sd']:7.4f} {o['eff_rank']:8.3f}", file=log, flush=True)

    with open(os.path.join(a.root, "support_summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(seeds=a.seeds, level=a.level, summary=summary), f, indent=2)
    log.close()
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
