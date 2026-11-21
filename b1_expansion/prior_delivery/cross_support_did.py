#!/usr/bin/env python3
"""prior_delivery.cross_support_did -- is the gain about *these* genes or about *a* gene set?

Every arm improves its own 203 genes out-of-fold. That is not evidence for content:
a weighted arm has 203 genes it was asked to fit better, and it fits them better.
The question is whether the *real* prior improves the real support more than the
*shuffled* prior improves the same genes. That is a difference-in-differences with a
gene-level permutation null, and it needs no new training.

    DiD = [d_real - d_shuf] on support(real)  -  [d_real - d_shuf] on support(shuffled)

    py -3.13 -u cross_support_did.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cv_nonlinear_null import load, standardise, ridge_oof, knn_oof, folds  # noqa: E402

ROOT = r"F:\UNAGI_Project\prior_delivery_runs\dose20_e45"
OUT = os.path.join(ROOT, "cross_support_did.json")


def support_idx(arm_dir):
    d = np.load(os.path.join(arm_dir, "support.npz"), allow_pickle=True)
    return np.asarray(d["idx"], dtype=int)


def null_did(d_real, d_shuf, nr, ns, n_draw=20000, seed=0):
    """Null for the DiD under random gene sets of the two observed sizes."""
    rng = np.random.default_rng(seed)
    ok = np.isfinite(d_real) & np.isfinite(d_shuf)
    pool = np.where(ok)[0]
    diff = d_real - d_shuf
    out = np.empty(n_draw)
    for i in range(n_draw):
        a = rng.choice(pool, nr, replace=False)
        b = rng.choice(pool, ns, replace=False)
        out[i] = diff[a].mean() - diff[b].mean()
    return out


def main():
    rep = {}
    for s in (42, 43, 44):
        base = os.path.join(ROOT, f"s{s}", "empty_e45")
        Zb, Xb = standardise(load(base)[0]), standardise(load(base)[1])
        r2b = ridge_oof(Zb, Xb)
        kbb = knn_oof(Zb, Xb)

        rows = {}
        for arm in ("real", "shuffled"):
            d = os.path.join(ROOT, f"s{s}", arm)
            Z, X = standardise(load(d)[0]), standardise(load(d)[1])
            print(f"[{s}/{arm}] oof + knn", flush=True)
            rows[arm] = dict(
                d=ridge_oof(Z, X) - r2b,
                k=knn_oof(Z, X) - kbb,
                sup=support_idx(d),
            )

        sr, ss = rows["real"]["sup"], rows["shuffled"]["sup"]
        out = {"n_real": int(sr.size), "n_shuf": int(ss.size),
               "overlap": int(np.intersect1d(sr, ss).size)}

        for tag, key in (("linear_oof", "d"), ("knn", "k")):
            dr, ds = rows["real"][key], rows["shuffled"][key]
            a = float(np.nanmean(dr[sr]))          # real prior, its own genes
            b = float(np.nanmean(dr[ss]))          # real prior, the other's genes
            c = float(np.nanmean(ds[sr]))          # shuffled prior, real's genes
            d_ = float(np.nanmean(ds[ss]))         # shuffled prior, its own genes
            did = (a - c) - (b - d_)
            nul = null_did(dr, ds, sr.size, ss.size, n_draw=20000, seed=s)
            out[tag] = dict(real_on_real=a, real_on_shuf=b, shuf_on_real=c,
                            shuf_on_shuf=d_,
                            specificity_real=a - b, specificity_shuf=d_ - c,
                            did=did, null_mean=float(nul.mean()),
                            null_sd=float(nul.std()),
                            z=float((did - nul.mean()) / max(nul.std(), 1e-12)),
                            p_two_sided=float((1 + (np.abs(nul - nul.mean())
                                                   >= abs(did - nul.mean())).sum())
                                              / (nul.size + 1)))
            print(f"  {tag:11s} real_on_real {a:+.5f} shuf_on_real {c:+.5f} "
                  f"| spec(real) {a-b:+.5f} spec(shuf) {d_-c:+.5f} "
                  f"| DiD {did:+.5f} z {out[tag]['z']:+.2f} p {out[tag]['p_two_sided']:.4f}",
                  flush=True)

        rep[str(s)] = out

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {OUT}")

    print(f"\n{'seed':>5} {'DiD linear':>11} {'z':>6} {'DiD knn':>9} {'z':>6}")
    for s, o in rep.items():
        print(f"{s:>5} {o['linear_oof']['did']:+11.5f} {o['linear_oof']['z']:+6.2f} "
              f"{o['knn']['did']:+9.5f} {o['knn']['z']:+6.2f}")


if __name__ == "__main__":
    main()
