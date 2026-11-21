# -*- coding: utf-8 -*-
"""Experiment B (revised) -- is the latent movement content or structure?

The planned second factor (GCN on/off) turned out to be unrunnable: with the plain
encoder UNAGI's iDREM loop cannot build a temporal graph at all
(`buildGraph.buildEdges` raises "No edges found" for every stage pair, at cutoff 0.05
and 0.01 alike, because `connectNodes` needs one stage pair whose distance is a strong
left-tail outlier, and the plain encoder's stage representations do not produce one).
So the arrival/non-arrival of the graph encoder is not a free factor in this codebase;
it is a precondition for the pipeline to run.

What remains -- and is what the thesis actually asked -- is whether the latent-side
movement is carried by prior *content* or by the *structure* of injecting any weighted
prior. That is a three-level content factor at matched budget, already on disk:

    empty_e45    no prior at all (the empty arm disables the TF prior mechanism)
    shuffled     degree-matched rewired prior: same support size, same degree sequence,
                 no biology -- structure without content
    real         the curated haematopoietic prior at 20x

If real ~ shuffled < empty, the movement is structural and content adds nothing.
If shuffled ~ empty < real, the movement is content.

Statistic: effective rank (participation ratio) of the covariance of the final latent Z,
the same quantity the thesis reports as the latent side of the dual spectrum.

Usage:  py -3.13 b_content_factor.py
"""
from __future__ import annotations

import json
import os

import numpy as np

ROOT = r"F:\UNAGI_Project\prior_delivery_runs\dose20_e45"
SEEDS = ("42", "43", "44")
ARMS = [("empty_e45", "no prior"), ("shuffled", "degree-matched rewired prior"),
        ("real", "real prior 20x")]
OUT = os.path.join(r"F:\UNAGI_Project\b1_expansion\prior_delivery", "b_content_factor.json")


def eff_rank(cov, eps=1e-12):
    e = np.linalg.eigvalsh(cov + eps * np.eye(cov.shape[0]))
    e = np.clip(e, eps, None)
    return float((e.sum() ** 2) / (e ** 2).sum())


def main():
    rows = {}
    for s in SEEDS:
        for arm, label in ARMS:
            p = os.path.join(ROOT, f"s{s}", arm, "latent.npz")
            d = np.load(p, allow_pickle=True)
            Z = np.asarray(d["Z"], dtype=np.float64)
            C = np.cov(Z, rowvar=False)
            rows.setdefault(arm, []).append(dict(seed=int(s), n=int(Z.shape[0]),
                                                 dim=int(Z.shape[1]),
                                                 eff_rank=round(eff_rank(C), 4),
                                                 trace=round(float(np.trace(C)), 3)))

    print(f"{'arm':12s} {'label':28s}  per-seed eff_rank           mean±sd")
    summary = {}
    for arm, label in ARMS:
        v = [r["eff_rank"] for r in rows[arm]]
        m = float(np.mean(v))
        sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        summary[arm] = dict(label=label, per_seed=v, mean=round(m, 4), sd=round(sd, 4))
        print(f"{arm:12s} {label:28s}  {['%.3f'%x for x in v]}   {m:.3f}±{sd:.3f}")

    e = summary["empty_e45"]["mean"]
    sh = summary["shuffled"]["mean"]
    re = summary["real"]["mean"]
    print()
    print("相对 no-prior 的有效秩变化（正数 = 秩升高，负数 = 秩降低）：")
    print(f"  rewired prior : {sh - e:+.3f}  ({(sh-e)/e*100:+.1f}%)")
    print(f"  real prior    : {re - e:+.3f}  ({(re-e)/e*100:+.1f}%)")
    print(f"  content 增量  : {re - sh:+.3f}，"
          f"相当于结构效应的 {(re-sh)/max(sh-e,1e-9)*100:.0f}%")

    res = dict(statistic="effective rank (participation ratio) of cov(Z), final latent",
               family="dose20_e45", seeds=list(SEEDS), arms=summary,
               delta_vs_no_prior=dict(rewired=round(e - sh, 4), real=round(e - re, 4)),
               content_increment=round(sh - re, 4))
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("->", OUT)


if __name__ == "__main__":
    main()
