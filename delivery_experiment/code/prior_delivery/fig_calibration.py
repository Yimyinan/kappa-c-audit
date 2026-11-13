#!/usr/bin/env python3
"""prior_delivery.fig_calibration -- Figure 4: what the 50% band is a scale of.

Section 5 argues from arithmetic that a support-limited prior cannot reach the band. This
figure supplies the empirical side: it calibrates the band by degrading the model along an
axis that has nothing to do with priors --- the number of training epochs --- and asking
where kappa lands.

Five arms, same seed, same data, same schedule, no prior, no adversarial branch, no
geometric penalty. Training runs from 2 to 45 epochs, a factor of 22.

    py -3.13 -u fig_calibration.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAL = r"F:\UNAGI_Project\prior_delivery_runs\calib\s42\calibration.json"
OUT = r"F:\UNAGI_Project\prior_delivery_runs\b1"
C_CAL = "#009E73"
C_REF = "#D55E00"
C_BAND = "#F6D9C6"


def main():
    d = json.load(open(CAL, encoding="utf-8"))
    arms = sorted(d.items(), key=lambda kv: int(kv[0][1:]))
    ep = np.array([int(k[1:]) for k, _ in arms], dtype=float)
    k_var = np.array([v["k_var"] for _, v in arms])
    k_all = np.array([v["k_all"] for _, v in arms])
    m_var = np.array([v["mean_var"] for _, v in arms])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9),
                             gridspec_kw=dict(width_ratios=[1, 1.05]))

    # ---- panel A: kappa vs training budget
    ax = axes[0]
    ax.plot(ep, k_var, "o-", color=C_CAL, lw=1.6, ms=4,
            label=r"$\kappa_{C(W)}$, varying genes")
    ax.plot(ep, k_all, "s--", color=C_CAL, lw=1.0, ms=3, alpha=0.6,
            label=r"$\kappa_{C(W)}$, all genes")
    ax.axhline(0.5, color="#8A3A0F", lw=1.2)
    ax.axhspan(0.35, 0.5, color=C_BAND, alpha=0.7, zorder=0)
    ax.text(2.2, 0.515, r"deep-compression band ($\kappa<0.5$)", fontsize=7,
            color="#8A3A0F", va="bottom")
    # label only the two endpoints; five labels collide on a log axis this narrow
    ax.annotate(f"{k_var[0]:.2f}", (ep[0], k_var[0]), textcoords="offset points",
                xytext=(-14, -2), ha="center", fontsize=7, color=C_CAL)
    ax.set_xscale("log")
    ax.set_xlim(1.6, 60)
    ax.set_ylim(0.35, 1.06)
    ax.set_xlabel("training epochs", fontsize=8)
    ax.set_ylabel(r"$\kappa_{C(W)}$ vs the 45-epoch arm", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.text(-0.16, 1.04, "(A)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left")

    # ---- panel B: the same curve against the quantity that actually degrades
    ax = axes[1]
    ax.plot(m_var, k_var, "o-", color=C_CAL, lw=1.6, ms=4)
    for x, y, e in zip(m_var, k_var, ep):
        dx, dy = (-16, -2) if e == 2 else (0, 7)
        ax.annotate(f"{int(e)} ep", (x, y), textcoords="offset points",
                    xytext=(dx, dy), ha="center", fontsize=7, color=C_CAL)
    # where the band would sit, by linear extrapolation in R2
    slope, icept = np.polyfit(m_var, k_var, 1)
    r2_at_band = (0.5 - icept) / slope
    ax.axhline(0.5, color="#8A3A0F", lw=1.2)
    ax.axhspan(0.35, 0.5, color=C_BAND, alpha=0.7, zorder=0)
    ax.axvline(r2_at_band, color="#8A3A0F", lw=0.8, ls=":")
    ax.annotate("linear extrapolation:\nband at $R^2\\approx$" + f"{r2_at_band:.3f}",
                xy=(r2_at_band, 0.5), xytext=(r2_at_band + 0.004, 0.62),
                fontsize=7, color="#8A3A0F",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#8A3A0F"))
    ax.set_xlim(0.06, 0.15)
    ax.set_ylim(0.35, 1.06)
    ax.set_xlabel("mean per-gene $R^2$ on varying genes", fontsize=8)
    ax.set_ylabel(r"$\kappa_{C(W)}$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.text(-0.16, 1.04, "(B)", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left")

    fig.tight_layout(pad=0.6)
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"fig4_calibration.{ext}")
        fig.savefig(p, dpi=400 if ext == "png" else None)
        print("wrote", p)

    print(f"\nslope dkappa/dR2 = {slope:.2f}; R2 at band = {r2_at_band:.4f}; "
          f"trained R2 = {m_var[-1]:.4f}")


if __name__ == "__main__":
    main()
