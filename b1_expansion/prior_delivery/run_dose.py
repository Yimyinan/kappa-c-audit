#!/usr/bin/env python3
"""prior_delivery.run_dose -- the dose-response that Part 5 asked for.

Part 5 left one thing open: at 5x the real 203-gene prior raises its own genes'
readout R2 by +0.0029 relative to a capacity- and weight-matched random set,
2 of 3 seeds, not resolved at n = 3. More seeds cannot settle that -- the noise
is between-seed, not within. A dose does settle it, because "the effect grows
with the factor" is a within-design prediction that noise does not reproduce.

So this driver trains exactly six arms: real and shuffled, at 20x, on the three
seeds that already carry the 5x pair. Nothing else varies -- same 12,000 cells
per seed, same schedule, same 203-gene support, same shuffle draw
(--shuffle-seed 7, so the random gene set is the one Part 5 used), same
normalisation (mean weight held at 1.0, so the manipulation stays
*concentration* and not a global rescale of the reconstruction term).

Prediction under test
---------------------
    if the per-gene tilt is real,  mean dR2(real - shuffled)  should grow from
    +0.003 at 5x to materially more at 20x. If it stays at +0.003 or shrinks,
    the 5x value was seed noise and Parts 1-5 close on a negative result.

What is deliberately NOT done here
----------------------------------
No extra seeds at 5x, no wider support, no new architecture. Those ladders are
flat (Parts 2-4) and repeating them would only widen the noise band.
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


def find_empty(seed, paired_root, seeds_root, baseline_root, baseline_seed=42):
    """Where the already-trained `empty` arm for *this* seed lives.

    Order matters: the copy under the 5x paired root is the exact reference the
    5x numbers were audited against, so reusing it keeps the two levels paired on
    the identical cell set. Falling back to r2/empty for seeds 43/44 would audit
    against another seed's baseline, which is the bug this ordering prevents.
    """
    cands = [os.path.join(paired_root, f"s{seed}", "empty"),
             os.path.join(seeds_root, f"s{seed}", "empty")]
    if seed == baseline_seed:
        cands.append(os.path.join(baseline_root, "empty"))
    for p in cands:
        if os.path.exists(os.path.join(p, "latent.npz")):
            return p
    return None


def already_trained(arm_dir, level):
    """True only if this exact arm (level included) finished."""
    lp = os.path.join(arm_dir, "latent.npz")
    mp = os.path.join(arm_dir, "arm_meta.json")
    if not (os.path.exists(lp) and os.path.exists(mp)):
        return False
    try:
        m = json.load(open(mp, encoding="utf-8"))
    except Exception:
        return False
    return abs(float(m.get("prior_level", -1)) - level) < 1e-9


def run(cmd, log, tag):
    t0 = time.time()
    print(f"\n===== {tag} @ {time.strftime('%H:%M:%S')} =====", file=log, flush=True)
    print(" ".join(cmd), file=log, flush=True)
    p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    dt = (time.time() - t0) / 60
    print(f"===== {tag} rc={p.returncode} in {dt:.1f} min =====", file=log, flush=True)
    return p.returncode, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"F:\UNAGI_Project\prior_delivery_runs\dose20")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    ap.add_argument("--level", type=float, default=20.0)
    ap.add_argument("--support-k", type=int, default=203)
    ap.add_argument("--shuffle-seed", type=int, default=7,
                    help="must match Part 5, so the random gene set is the same one")
    ap.add_argument("--per-stage", type=int, default=2400)
    ap.add_argument("--epoch-initial", type=int, default=30)
    ap.add_argument("--epoch-iter", type=int, default=15)
    ap.add_argument("--max-iter", type=int, default=2)
    ap.add_argument("--edge-cutoff", type=float, default=0.3)
    ap.add_argument("--cpo", choices=["on", "off"], default="off")
    ap.add_argument("--idrem", default=r"F:\UNAGI_Project\idrem")
    ap.add_argument("--baseline-root", default=r"F:\UNAGI_Project\prior_delivery_runs\r2")
    ap.add_argument("--seeds-root", default=r"F:\UNAGI_Project\prior_delivery_runs\seeds")
    ap.add_argument("--paired-root", default=r"F:\UNAGI_Project\prior_delivery_runs\support",
                    help="root holding the 5x pair this design is the dose extension of")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    os.makedirs(a.root, exist_ok=True)
    log = open(os.path.join(a.root, "run.log"), "a", encoding="utf-8")
    status = {}

    for seed in a.seeds:
        sdir = os.path.join(a.root, f"s{seed}")
        os.makedirs(sdir, exist_ok=True)

        ref = find_empty(seed, a.paired_root, a.seeds_root, a.baseline_root)
        if ref is None:
            print(f"[dose] no empty reference for seed {seed}; skipping", file=log, flush=True)
            status[f"s{seed}"] = dict(error="no empty reference")
            continue
        dref = os.path.join(sdir, "empty")
        if not os.path.exists(os.path.join(dref, "latent.npz")):
            os.makedirs(dref, exist_ok=True)
            shutil.copy2(os.path.join(ref, "latent.npz"), dref)
            m = os.path.join(ref, "arm_meta.json")
            if os.path.exists(m):
                shutil.copy2(m, dref)
        print(f"\n===== seed {seed} empty <- {ref} =====", file=log, flush=True)

        # real arm: same support and code path as Part 5's k203, only the factor moves
        real_dir = os.path.join(sdir, "real")
        if already_trained(real_dir, a.level):
            print(f"[dose] s{seed}/real already at {a.level:g}x; reusing", file=log, flush=True)
            status[f"s{seed}/real"] = dict(reused=True)
        else:
            cmd = [a.python, "-u", os.path.join(HERE, "run_arm.py"),
                   "--arm", "binary", "--workdir", real_dir,
                   "--per-stage", str(a.per_stage),
                   "--max-iter", str(a.max_iter),
                   "--epoch-initial", str(a.epoch_initial),
                   "--epoch-iter", str(a.epoch_iter),
                   "--edge-cutoff", str(a.edge_cutoff),
                   "--cpo", a.cpo, "--seed", str(seed),
                   "--weight-mode", "direct", "--prior-level", str(a.level),
                   "--support-k", str(a.support_k),
                   "--idrem", a.idrem]
            rc, dt = run(cmd, log, f"s{seed}/real ({a.level:g}x, k={a.support_k})")
            status[f"s{seed}/real"] = dict(returncode=rc, minutes=round(dt, 1))
            if rc != 0:
                print(f"[dose] real arm failed for seed {seed}; stopping", file=log, flush=True)
                break

        # shuffled control: same support size, same weight, different genes
        shuf_dir = os.path.join(sdir, "shuffled")
        if already_trained(shuf_dir, a.level):
            print(f"[dose] s{seed}/shuffled already at {a.level:g}x; reusing", file=log,
                  flush=True)
            status[f"s{seed}/shuffled"] = dict(reused=True)
        else:
            cmd = [a.python, "-u", os.path.join(HERE, "run_arm.py"),
                   "--arm", "shuffleflat", "--workdir", shuf_dir,
                   "--per-stage", str(a.per_stage),
                   "--max-iter", str(a.max_iter),
                   "--epoch-initial", str(a.epoch_initial),
                   "--epoch-iter", str(a.epoch_iter),
                   "--edge-cutoff", str(a.edge_cutoff),
                   "--cpo", a.cpo, "--seed", str(seed),
                   "--weight-mode", "direct", "--prior-level", str(a.level),
                   "--shuffle-seed", str(a.shuffle_seed),
                   "--idrem", a.idrem]
            rc, dt = run(cmd, log, f"s{seed}/shuffled ({a.level:g}x)")
            status[f"s{seed}/shuffled"] = dict(returncode=rc, minutes=round(dt, 1))
            if rc != 0:
                print(f"[dose] shuffled arm failed for seed {seed}; stopping", file=log,
                      flush=True)
                break

        cmd = [a.python, "-u", os.path.join(HERE, "audit_arms.py"),
               "--root", sdir, "--ref", "empty", "--arms", "empty", "real", "shuffled"]
        run(cmd, log, f"s{seed} audit")

        cmd = [a.python, "-u", os.path.join(HERE, "pergene_fit.py"),
               "--real", real_dir, "--shuffled", shuf_dir,
               "--out", os.path.join(sdir, "pergene_fit_real_vs_shuffled.json")]
        run(cmd, log, f"s{seed} per-gene fit")

    with open(os.path.join(a.root, "run_status.json"), "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)

    # ---- dose-response summary ---------------------------------------------
    def read_fit(path):
        """The registered statistic: real-subset mean minus shuffled-subset mean.

        Both subset means are already expressed as Δ against the other arm after
        subtracting the genome-wide median shift, so their difference is exactly
        the difference-of-differences Part 5 reports. Reading only the real row
        would drop the control and inflate every fold change.
        """
        if not os.path.exists(path):
            return None
        d = json.load(open(path, encoding="utf-8"))
        real = next((r for r in d["rows"] if r["label"].startswith("real")), None)
        shuf = next((r for r in d["rows"] if r["label"].startswith("shuffled")), None)
        if real is None or shuf is None:
            return None
        return dict(real=real["mean_centered"], shuffled=shuf["mean_centered"],
                    d=real["mean_centered"] - shuf["mean_centered"],
                    frac_pos_centered=real["frac_pos_centered"],
                    n=real["n"], aggregate_ratio=d.get("aggregate_ratio"))

    def read_audit(path, arm):
        if not os.path.exists(path):
            return None
        for r in json.load(open(path, encoding="utf-8"))["rows"]:
            if r["arm"] == arm:
                return dict(kappa_C_W=r["kappa_C_W"], kappa_C_Z=r["kappa_C_Z"],
                            C_percent=r["C_percent"], eff_rank=r["eff_rank"],
                            n_boosted=r["n_boosted"])
        return None

    table = {}
    for seed in a.seeds:
        s5 = os.path.join(a.paired_root, f"s{seed}")
        s20 = os.path.join(a.root, f"s{seed}")
        table[seed] = dict(
            x5=read_fit(os.path.join(s5, "pergene_k203_vs_shuffleflat.json")),
            x20=read_fit(os.path.join(s20, "pergene_fit_real_vs_shuffled.json")),
            audit5_k203=read_audit(os.path.join(s5, "audit_results.json"), "k203"),
            audit20_real=read_audit(os.path.join(s20, "audit_results.json"), "real"),
        )

    print("\n" + "=" * 88, file=log, flush=True)
    print(f"dose-response, {a.level:g}x against the existing 5x pair", file=log, flush=True)
    print("=" * 88, file=log, flush=True)
    hdr = (f"{'seed':>5s} {'5x mean dR2':>12s} {'20x mean dR2':>13s} "
           f"{'fold':>7s} {'5x frac>0':>10s} {'20x frac>0':>11s}")
    print(hdr, file=log, flush=True)
    print("-" * len(hdr), file=log, flush=True)
    folds = []
    for seed, t in table.items():
        a5, a20 = t["x5"], t["x20"]
        if not (a5 and a20):
            print(f"{seed:5d} {'--':>12s} {'--':>13s}", file=log, flush=True)
            continue
        m5, m20 = a5["d"], a20["d"]
        fold = (m20 / m5) if m5 not in (0, None) else float("nan")
        if m5 and m5 > 0:
            folds.append(fold)
        print(f"{seed:5d} {m5:+12.5f} {m20:+13.5f} {fold:7.2f} "
              f"{a5['frac_pos_centered']:10.3f} {a20['frac_pos_centered']:11.3f}",
              file=log, flush=True)

    summary = {}
    v5 = [t["x5"]["d"] for t in table.values() if t["x5"]]
    v20 = [t["x20"]["d"] for t in table.values() if t["x20"]]
    if v5:
        summary["x5"] = dict(mean=st.mean(v5), sd=st.stdev(v5) if len(v5) > 1 else 0.0,
                             n=len(v5), values=v5)
    if v20:
        summary["x20"] = dict(mean=st.mean(v20), sd=st.stdev(v20) if len(v20) > 1 else 0.0,
                              n=len(v20), values=v20)
    if v5 and v20 and len(v5) == len(v20):
        paired = [b - a_ for a_, b in zip(v5, v20)]
        summary["delta_20_minus_5"] = dict(
            mean=st.mean(paired), sd=st.stdev(paired) if len(paired) > 1 else 0.0,
            n_positive=sum(1 for x in paired if x > 0), values=paired)
    with open(os.path.join(a.root, "dose_summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(level=a.level, per_seed=table, summary=summary), f, indent=2,
                  ensure_ascii=False)
    print("\n" + json.dumps(summary, indent=2), file=log, flush=True)
    log.close()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
