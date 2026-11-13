#!/usr/bin/env python3
"""prior_delivery.fig_resolution -- Figure 1 of the reporting-standard paper.

Panel A: what a spectrum diagnostic can and cannot see as a function of the prior's
support fraction. The curves are exact arithmetic, not fits:

    kappa = 1 + [ n_s * delta_local + (N - n_s) * delta_global ] / S

Two conventions are drawn, because the denominator decides what the diagnostic can
report. "All genes" counts every column of the matrix; "varying" drops the genes that
are constant in the data -- they score R2 = 1 by the 0/0 convention and contribute 58%
of S without ever being predicted. Every quantity below (N, S, both deltas, the support
size, the baseline R2) is recomputed inside its own convention, so each curve is
self-consistent and the two can be compared.

Panel B: the measured decomposition for the real prior and for the weight-matched random
control, in the varying convention.

Run: py -3.13 -u fig_resolution.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

OUT = r"F:\UNAGI_Project\prior_delivery_runs\b1"
LADDER = r"F:\UNAGI_Project\prior_delivery_runs\dose_ladder.json"

# ---- two self-consistent conventions (20x / 45-epoch arms, seed 42) ----------------
CONV = {
    "varying": dict(N=17328, S=2322.3, m=0.1340, ns=191,
                    d_local=+0.00502, d_global=+0.00045,
                    c_local=-0.00955, c_global=-0.00212),
    "all": dict(N=20527, S=5521.3, m=0.2690, ns=203,
                d_local=+0.00861, d_global=+0.00017,
                c_local=+0.00188, c_global=-0.00250),
}

C_REAL = "#D55E00"
C_CTRL = "#0072B2"
C_IDEAL = "#999999"
C_BAND = "#F6D9C6"


def dkappa(c, n_s, d_local, d_global):
    return (n_s * d_local + (c["N"] - n_s) * d_global) / c["S"]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0),
                             gridspec_kw=dict(width_ratios=[1.35, 1]))

    # ---------------- panel A ----------------
    ax = axes[0]
    for name, c in CONV.items():
        primary = (name == "varying")
        f = 10 ** np.linspace(np.log10(2 / c["N"]), 0, 400)
        n_s = f * c["N"]
        lw, alpha = (1.8, 1.0) if primary else (1.0, 0.55)
        dash = "-" if primary else (0, (1, 1))

        ax.plot(f, dkappa(c, n_s, 1 - c["m"], 0.0), color=C_IDEAL, lw=lw,
                ls=":" if primary else dash, alpha=alpha,
                label=(r"ceiling, $R^2\to1$ (varying)" if primary else None))
        ax.plot(f, dkappa(c, n_s, c["d_local"], 0.0), color=C_REAL, lw=lw,
                ls=dash, alpha=alpha,
                label=(r"measured $\delta_{local}$ (varying)" if primary else None))
        ax.plot(f, dkappa(c, n_s, c["d_local"], c["c_global"]), color=C_CTRL,
                lw=lw, ls="--" if primary else dash, alpha=alpha,
                label=(r"$+\,$control's global cost" if primary else None))

    ax.axhline(0, color="k", lw=0.8)
    ax.axhspan(-0.5, -0.35, color=C_BAND, alpha=0.7, zorder=0)
    ax.text(0.00016, -0.43, "deep-compression band\n($C>50\\%$, i.e. $\\kappa<0.5$)",
            fontsize=7, va="center", color="#8A3A0F")

    pts = []
    if os.path.exists(LADDER):
        d = json.load(open(LADDER, encoding="utf-8"))
        for lv, m in d["means"].items():
            for v in m["kcw"]:
                pts.append((float(lv), 1 - v))
    fx = CONV["varying"]["ns"] / CONV["varying"]["N"]
    for lv in sorted({p[0] for p in pts}):
        ys = [y for x, y in pts if x == lv]
        ax.scatter([fx] * len(ys), ys, s=16, color=C_REAL, zorder=5,
                   edgecolor="white", linewidth=0.3)
    ax.annotate("9 trained arms\n(5x, 20x, 40x)", xy=(fx, 0.012),
                xytext=(fx * 1.2, 0.16), fontsize=6.2, color=C_REAL,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C_REAL))

    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(-0.55, 0.30)
    ax.set_xlabel("prior support fraction  $n_s/N$", fontsize=8)
    ax.set_ylabel(r"$\Delta\kappa_{C(W)}=1-\kappa$", fontsize=8)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper left", frameon=False, handlelength=2.2)
    ax.text(-0.14, 1.04, "(A)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left")

    # ---------------- panel B ----------------
    ax = axes[1]
    c = CONV["varying"]              # panel B is drawn in the varying convention
    labels = ["real\nprior", "random\ncontrol"]
    local = [c["ns"] * c["d_local"] / c["S"] * 100,
             c["ns"] * c["c_local"] / c["S"] * 100]
    glob = [(c["N"] - c["ns"]) * c["d_global"] / c["S"] * 100,
            (c["N"] - c["ns"]) * c["c_global"] / c["S"] * 100]
    x = np.arange(2)
    w = 0.34
    b1 = ax.bar(x - w / 2, local, w, color=C_REAL, label="named genes (local)")
    b2 = ax.bar(x + w / 2, glob, w, color=C_CTRL, label="everything else (global)")
    ax.axhline(0, color="k", lw=0.8)
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width() / 2, h + (0.05 if h >= 0 else -0.16),
                    f"{h:+.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(r"contribution to $\Delta\kappa$  (%)", fontsize=8)
    ax.set_ylim(-2.0, 0.7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.text(-0.16, 1.04, "(B)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left")

    fig.tight_layout(pad=0.6)
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"fig1_resolution.{ext}")
        fig.savefig(p, dpi=400 if ext == "png" else None)
        print("wrote", p)

    print()
    out = {}
    for name, cc in CONV.items():
        cap = dkappa(cc, cc["ns"], 1 - cc["m"], 0.0)
        need = 0.5 * cc["m"] / cc["d_local"]
        out[name] = dict(f=cc["ns"] / cc["N"], ceiling=cap,
                         measured=dkappa(cc, cc["ns"], cc["d_local"], cc["d_global"]),
                         control=dkappa(cc, cc["ns"], cc["c_local"], cc["c_global"]),
                         need=need)
        print(f"{name:8s} f={cc['ns']/cc['N']*100:5.2f}%  ceiling {cap*100:5.2f}%  "
              f"measured {out[name]['measured']*100:+.3f}%  "
              f"control {out[name]['control']*100:+.3f}%  "
              f"need {need*100:6.0f}%")
    with open(os.path.join(OUT, "fig1_numbers.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
