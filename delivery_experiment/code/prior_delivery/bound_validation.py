#!/usr/bin/env python3
"""prior_delivery.bound_validation -- is the resolution bound a law or a derivation?

Section 5 of paper3 states

    dkappa = [ n_s * d_local + (N - n_s) * d_global ] / S

and uses it to argue that a support-limited prior cannot move the aggregate. The support
ladder (5x, k = 203 / 1,000 / 2,027) is a three-point test of that statement: the observed
dkappa is already in `support_summary.json`, and d_local can be measured per arm. If the
prediction tracks the observation across a tenfold range of n_s, the bound is validated
rather than asserted.

Second question, which can invalidate our own headline: S is a sum over all 20,527 genes,
but most of them are never predicted. If the "effective transcriptome" is only the genes
the decoder can actually fit, the support fraction is much larger than 1%, and the bound
is correspondingly looser. This script reports both definitions so the paper can state
which one it uses.

    py -3.13 -u bound_validation.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pergene_fit import per_gene_r2  # noqa: E402

ROOT = r"F:\UNAGI_Project\prior_delivery_runs\support"
SUMMARY = os.path.join(ROOT, "support_summary.json")
OUT = os.path.join(ROOT, "bound_validation.json")
ARMS = ["k203", "k1000", "k2027"]


def load_r2(arm_dir, cache=True):
    p = os.path.join(arm_dir, "pergene_r2.npy")
    if cache and os.path.exists(p):
        return np.load(p)
    z = np.load(os.path.join(arm_dir, "latent.npz"), allow_pickle=True)
    r2 = per_gene_r2(z["Z"], z["X"])
    np.save(p, r2)
    return r2


CORE = np.load(r"F:\UNAGI_Project\prior_delivery_runs\dose20_e45\s42\real\support.npz",
               allow_pickle=True)
NAMES = [str(g).upper() for g in CORE["gene_names"]]
CORE_IDX = np.asarray(CORE["idx"], dtype=int)
_RANKED = None


def _ranked():
    global _RANKED
    if _RANKED is None:
        import arms
        order, _ = arms.trrust_degree_ranking(
            NAMES, r"F:\UNAGI_Project\unagi_custom_data\trrust_raw_human.txt")
        pos = {g: i for i, g in enumerate(NAMES)}
        _RANKED = [pos[g] for g in order if g in pos]
    return _RANKED


def support_idx(arm_dir, k):
    """Rebuild the rung: the curated core first, then highest in-degree targets.

    These arms predate support.npz, so the rung has to be reconstructed with the same
    rule run_arm used; the realised count is checked against arm_meta.
    """
    sel = list(CORE_IDX)
    seen = set(sel)
    for i in _ranked():
        if len(sel) >= k:
            break
        if i not in seen:
            sel.append(i)
            seen.add(i)
    return np.asarray(sel[:k], dtype=int)


def main():
    summ = json.load(open(SUMMARY, encoding="utf-8"))
    rows = []
    for s in ("42", "43", "44"):
        base = os.path.join(ROOT, f"s{s}", "empty")
        if not os.path.exists(os.path.join(base, "latent.npz")):
            continue
        r2b = load_r2(base)
        S = float(np.nansum(r2b))
        for arm in ARMS:
            d = os.path.join(ROOT, f"s{s}", arm)
            if not os.path.exists(os.path.join(d, "latent.npz")):
                continue
            r2 = load_r2(d)
            idx = support_idx(d, {"k203": 203, "k1000": 1000, "k2027": 2027}[arm])
            delta = float(np.nanmean((r2 - r2b)[idx]))
            delta_rest = float(np.nanmean(np.delete(r2 - r2b, idx)))
            obs = 1.0 - summ["summary"][arm]["per_seed"][s]["kappa_C_W"]
            pred = (idx.size * delta) / S
            rows.append(dict(seed=int(s), arm=arm, n_s=int(idx.size),
                             d_local=delta, d_rest=delta_rest, S=S,
                             observed_dkappa=float(obs), predicted_dkappa=float(pred)))
            print(f"s{s} {arm:6s} n_s {idx.size:5d} d_local {delta:+.5f} "
                  f"d_rest {delta_rest:+.5f} | observed {obs*100:+.3f}% "
                  f"predicted {pred*100:+.3f}%", flush=True)

    # effective-transcriptome correction, on one baseline
    base = os.path.join(ROOT, "s42", "empty")
    r2b = load_r2(base)
    z = np.load(os.path.join(base, "latent.npz"), allow_pickle=True)
    X = np.asarray(z["X"], dtype=np.float64)
    var = X.var(0)
    n_all = X.shape[1]
    defs = {
        "all genes": np.ones(n_all, bool),
        "non-constant in this dataset": var > 1e-12,
        "baseline R2 > 0.05": r2b > 0.05,
        "baseline R2 > 0.10": r2b > 0.10,
        "top 5,000 by baseline R2": np.isin(np.arange(n_all),
                                            np.argsort(-r2b)[:5000]),
    }
    eff = {}
    delta_local = 0.00866                       # measured, 20x / 45-epoch arm
    for name, m in defs.items():
        N = int(m.sum())
        n_s = 203
        mean_r2 = float(np.nanmean(r2b[m]))
        # dkappa = f * d_local / meanR2, so reaching 0.5 needs f = 0.5 * meanR2 / delta
        f_need = 0.5 * mean_r2 / delta_local
        eff[name] = dict(N_eff=N, mean_baseline_r2=round(mean_r2, 4),
                         support_fraction=round(n_s / N, 4),
                         dkappa_at_100pct_readout=round(n_s / N, 4),
                         fraction_of_transcriptome_for_50pct=round(f_need, 1))
        print(f"{name:32s} N_eff {N:6d} f {n_s/N*100:6.2f}% "
              f"d_kappa(100%) {n_s/N*100:5.2f}% "
              f"| need {f_need*100:7.0f}% of it to reach -50%")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(dict(rows=rows, effective=eff), f, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
