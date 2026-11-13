# -*- coding: utf-8 -*-
"""Supplementary figure S20b: the trace trade-off fails, the like-for-like
effective-rank dissociation survives.

Panel (a) PREDICTABILITY level.  The constant-reconstruction-loss argument predicts
the conservation line log k-C(W) = -log k-C(Z) (grey dashed).  Plotted in the
corrected predictability statistic the points are UNCORRELATED (red fit slope
~0), so conservation is rejected and, more strongly, neither trace pins the
other -- the "positively coupled" reading of the released trace form was an
artefact of the readout's functional form (see S1F_0OF33_RECHECK.md).

Panel (b) EFFECTIVE-RANK level, like-for-like.  Same statistic (participation
ratio) on both sides.  The latent compresses (mean ratio 0.72) while the decoder
readout does not (mean ratio 0.95), and the two are uncorrelated: the dissociation
is not a statistic mismatch -- it survives the like-for-like test.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figure_palette as fp

JSON = os.path.join(HERE, "..", "data", "kappa_c_pr_comparison.json")
FIG = os.path.join(HERE, "..", "figures", "figS_trace_tradeoff")

fp.apply_sample_style()
d = json.load(open(JSON))
rows = d["rows"]
n = len(rows)

ltrW_trace = np.log(np.array([r["kappa_C_W_trace"] for r in rows]))
ltrZ = np.log(np.array([r["kappa_C_Z_trace"] for r in rows]))
lpW = np.log(np.array([r["pr_ratio_W"] for r in rows]))
lpZ = np.log(np.array([r["pr_ratio_Z"] for r in rows]))

# Panel (a) is drawn in the CORRECTED predictability statistic.  The released
# trace-form values are kept above only for the reproduction record.
S1G = os.path.abspath(os.path.join(HERE, "..", "..", "..", "outputs",
                                   "positive_result", "s1g_corrected_summaries.json"))
_s1g = {r["config"]: r["kappa_r2"] for r in json.load(open(S1G,
                                                          encoding="utf-8-sig"))["rows"]}
_missing = [r["config"] for r in rows if r["config"] not in _s1g]
if _missing:
    raise SystemExit("missing corrected kappa for: %s" % _missing)
ltrW = np.log(np.array([_s1g[r["config"]] for r in rows]))

TCOL = {"hemo": fp.C_HEMA, "neural": fp.C_NEURAL, "lung": fp.C_LUNG}
TNAME = {"hemo": "hematopoietic", "neural": "neural cortex", "lung": "lung fibrosis"}
tis = [r["tissue"] for r in rows]
cols = [TCOL[t] for t in tis]

# drawn at 	extwidth so each pt below is the size the reader sees; the old
# 9.6 in design was scaled to 0.71 and landed the 7.5 pt notes at 4.98 pt.
fig, axes = plt.subplots(1, 2, figsize=fp.figsize(170, 96))


def panel(ax, xz, yw, title, xlabel, ylabel, cons_line=False, fit=True, colors=None):
    if cons_line:
        lo = min(xz.min(), yw.min()) - 0.5
        hi = max(xz.max(), yw.max()) + 0.5
        xs = np.linspace(lo, hi, 50)
        ax.plot(xs, -xs, ls="--", lw=1.4, color=fp.GRAY, zorder=1,
                label=r"conservation  $y=-x$")
    ax.axhline(0, lw=0.8, color=fp.GRAY, alpha=0.55, zorder=0)
    ax.axvline(0, lw=0.8, color=fp.GRAY, alpha=0.55, zorder=0)
    if colors is None:
        ax.scatter(xz, yw, s=44, color=fp.C_HEMA, edgecolor="white",
                   linewidth=0.8, zorder=3, label="prior-injected")
    else:
        for t in sorted(set(tis)):
            m = np.array([x == t for x in tis])
            ax.scatter(xz[m], yw[m], s=46, color=TCOL[t], edgecolor="white",
                       linewidth=0.8, zorder=3, label=TNAME[t])
    if fit and len(xz) > 2:
        s, b = np.polyfit(xz, yw, 1)
        r = float(np.corrcoef(xz, yw)[0, 1])
        xr = np.linspace(xz.min(), xz.max(), 20)
        ax.plot(xr, s * xr + b, lw=1.8, color=fp.RED, zorder=2,
                label="fit: slope %+.2f, $r$=%+.2f" % (s, r))
    ax.set_xlabel(xlabel, fontsize=fp.FS_AXIS)
    ax.set_ylabel(ylabel, fontsize=fp.FS_AXIS)
    ax.set_title(title, fontsize=fp.FS_TITLE)
    return ax


# ---- (a) trace level -------------------------------------------------------
ax = axes[0]
panel(ax, ltrZ, ltrW,
      "(a) predictability level:\nconservation rejected, traces uncoupled",
      r"$\log\,\kappa$-C(Z)  (latent trace)",
      r"$\log\,\kappa$-C(W)  (decoder predictability)",
      cons_line=True, colors=cols)
prods = np.array([_s1g[r["config"]] * r["kappa_C_Z_trace"] for r in rows])
within = int(np.sum(np.abs(np.log(prods)) < np.log(1.25)))
ax.text(0.03, 0.97, "median product %.2f\n%d/%d within $\\pm$25%%"
        % (np.median(prods), within, n),
        transform=ax.transAxes, va="top", ha="left", fontsize=fp.FS_NOTE, color=fp.RED,
        bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=fp.RED, lw=0.7, alpha=0.92))
fp.safe_legend(ax, loc="lower right", fontsize=fp.FS_NOTE,
               frameon=True, framealpha=0.92, edgecolor="none")

# ---- (b) effective-rank level ---------------------------------------------
ax = axes[1]
panel(ax, lpZ, lpW,
      "(b) like-for-like effective rank:\ndissociation survives",
      r"$\log$ pr-ratio(Z)  (latent effective rank)",
      r"$\log$ pr-ratio(W)  (decoder effective rank)", colors=cols)
mpW = float(np.mean([r["pr_ratio_W"] for r in rows]))
mpZ = float(np.mean([r["pr_ratio_Z"] for r in rows]))
r_er = float(np.corrcoef(lpZ, lpW)[0, 1])
ax.text(0.03, 0.97,
        "mean ratio  W = %.3f  (no compression)\nmean ratio  Z = %.3f  (compression)\n"
        "$r$ = %+.2f  (uncoupled)" % (mpW, mpZ, r_er),
        transform=ax.transAxes, va="top", ha="left", fontsize=fp.FS_NOTE, color=fp.RED,
        bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=fp.RED, lw=0.7, alpha=0.92))
fp.safe_legend(ax, loc="lower right", fontsize=fp.FS_NOTE,
               frameon=True, framealpha=0.92, edgecolor="none")

fig.tight_layout(pad=0.6)
fp.assert_design_width(fig)
fp.export(fig, FIG, deck=False)

print("(a) trace:      slope=%+.3f r=%+.3f  median product=%.3f  %d/%d within 25%%"
      % (np.polyfit(ltrZ, ltrW, 1)[0], np.corrcoef(ltrZ, ltrW)[0, 1],
         np.median(prods), within, n))
print("(b) eff.-rank:  mean pr-ratio W=%.3f  Z=%.3f   r=%+.3f  slope=%+.3f"
      % (mpW, mpZ, r_er, np.polyfit(lpZ, lpW, 1)[0]))
print("[saved] %s.pdf / .png" % os.path.abspath(FIG))
