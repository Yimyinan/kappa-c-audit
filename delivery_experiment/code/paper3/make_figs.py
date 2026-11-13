#!/usr/bin/env python3
"""paper3.make_figs -- Figures 2 and 3.

Fig 2  the ceiling: matched per-gene effect against dose (A) and against the length of
       the weighted phase (B). Per-seed points, mean line.
Fig 3  the basis audit: variance share of each basis within its own covered block, in
       the two datasets.

Reads only files that already exist; no training. Run from anywhere.
"""
from __future__ import annotations

import json
import os
import statistics as st

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = r"F:\UNAGI_Project\prior_delivery_runs"
OUT = r"F:\UNAGI_Project\paper3\fig"

C_REAL = "#D55E00"
C_CTRL = "#0072B2"
C_FILL = "#D8CFC7"
C_DATA = "#5E8C61"


def load_robust(path):
    d = json.load(open(path, encoding="utf-8"))
    return [v["decile_excess_mean"] for v in d.values()]


def fig_ceiling():
    # dose ladder: each level has its own pergene_robust.json
    dose = {}
    for lv, root in ((5, "support"), (20, "dose20"), (40, "dose40")):
        p = os.path.join(RUNS, root, "pergene_robust.json" if lv != 5 else "pergene_robust_5x.json")
        if os.path.exists(p):
            dose[lv] = load_robust(p)
    time = {15: dose.get(20, []),
            45: load_robust(os.path.join(RUNS, "dose20_e45", "pergene_robust.json"))}

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, (title, data, xlabel) in zip(
            axes,
            (("A  dose", dose, "dose (support : rest)"),
             ("B  length of the weighted phase", time, "weighted epochs at $20\\times$"))):
        xs = sorted(data)
        for x in xs:
            ys = data[x]
            ax.scatter([x] * len(ys), ys, s=18, color=C_REAL, alpha=0.55,
                       edgecolor="white", linewidth=0.4, zorder=3)
        ax.plot(xs, [st.mean(data[x]) for x in xs], color=C_REAL, lw=1.6, zorder=4)
        ax.axhline(0, color="k", lw=0.7)
        if title.startswith("A"):
            ax.set_xscale("log")
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{x}$\\times$" for x in xs])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        else:
            ax.set_xticks(xs)
        ax.set_ylim(-0.002, 0.021)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(r"matched per-gene effect $d_m$", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=8.5, loc="left")
    # mark the common ceiling
    ceil = st.mean(dose[40]) if 40 in dose else np.nan
    axes[0].axhline(ceil, color=C_CTRL, lw=0.9, ls="--")
    axes[0].text(5.2, ceil + 0.0006, f"ceiling $\\approx${ceil:+.4f}", fontsize=6.4,
                 color=C_CTRL)
    fig.tight_layout(pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig2_ceiling.{ext}"), dpi=400 if ext == "png" else None)
    print("fig2 ceiling =", round(ceil, 5))
    return ceil


def fig_basis():
    a = json.load(open(os.path.join(RUNS, "b1", "basis_variance_audit.json"), encoding="utf-8"))
    b = json.load(open(os.path.join(RUNS, "b1", "basis_audit_frangieh.json"), encoding="utf-8"))
    order = ["grn_spectral", "grn_indicator", "permuted", "pca32", "nmf32"]
    labels = ["GRN\nspectral", "GRN\nindicator", "permuted\ncontrol", "PCA-32", "NMF-32"]
    colors = [C_REAL, C_REAL, C_CTRL, C_DATA, C_DATA]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for ax, dat, name in ((axes[0], a, "haematopoiesis (12k cells)"),
                          (axes[1], b, "melanoma Perturb-CITE-seq (4k cells)")):
        vals = {r["basis"]: r["block_share"] for r in dat["rows"]}
        ys = [vals.get(k, np.nan) for k in order]
        x = np.arange(len(order))
        bars = ax.bar(x, ys, 0.62, color=colors)
        for r, v in zip(bars, ys):
            if np.isfinite(v):
                ax.text(r.get_x() + r.get_width() / 2, v + 0.012, f"{v:.3f}",
                        ha="center", fontsize=6.2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.4)
        cov = next(r["n_covered"] for r in dat["rows"] if r["basis"] == "grn_spectral")
        ax.axhline(32 / cov, color="k", lw=0.7, ls=":",
                   label="chance (dimension ratio)" if name.startswith("haemato") else None)
        ax.set_ylim(0, 0.72)
        ax.set_title(name, fontsize=8.5, loc="left")
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("variance share of the covered block", fontsize=8)
    axes[0].legend(fontsize=6, frameon=False, loc="upper left")
    fig.tight_layout(pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig3_basis.{ext}"), dpi=400 if ext == "png" else None)
    print("fig3 written")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_ceiling()
    fig_basis()
