# -*- coding: utf-8 -*-
"""Fig. 2 (schematic): workflow of the kappa-C audit. Single-panel overview.

V32 BBOX DECOLLISION (2026-11-15)
---------------------------------
Purely geometric follow-up to V31: two pairs of adjacent text-span bounding
boxes touched by <= 3.8 pt^2 (a math subscript vs the 03 / CLASSIFICATION
label, and two consecutive gate-chip rows). The left spectrum's two-line
formula block is tightened (leading 1.35 -> 1.10) so the Sigma_g subscript
clears the stage label below and the first line clears the box title above;
G4 is dropped by 0.04 axis units with its arrow and the serial rail attached.
No text, topology, colour or font size changed. See
V32_FIG2_BBOX_FIX_2026-11-15.md.

V31 RESTYLE + TOPOLOGY CORRECTION (2026-11-15)
----------------------------------------------
The V30 drawing was a document laid out inside rectangles: every one of 20+
strings used FS_NOTE, box edges landed on five different x coordinates, three
verdict colours were applied to statements that are not verdicts, and the
"two spectra, one audit" caption floated free of any group.

Visual fixes:
  * typographic hierarchy from the palette grid -- FS_TITLE for the group band,
    FS_AXIS for box titles / stage labels, FS_NOTE for body text, FS_PANEL for
    the one number the eye should land on (0/33);
  * one grid: LEFT = 0.15, RIGHT = 9.85, two equal 4.75-wide tracks;
  * verdict colours (FAIL/PASS) removed entirely. SEMANTIC_ROLES reserves them
    for actual FAIL/PASS verdicts, but "0/33 null configurations" is a finding,
    not a failed experiment. CAUTION is used only for the threshold chip and
    for Gate 1, both of which really are critical lines;
  * in-box text is a centred block, never hand-placed per line, so mathtext
    ascenders/descenders cannot push a line through a box edge.

Topology fixes (the drawing contradicted its own label / the caption):
  * DELETED the kappa-C(W) -> latent-spectrum arrow. The two spectra are
    parallel readouts of the same paired models, not a causal chain; the group
    band now carries the "two spectra, one audit" claim instead of an orphan
    italic line.
  * DELETED the latent-spectrum -> classification arrow. That box is explicitly
    "decoder-readout classification"; the companion kappa-C-Z audit does not
    enter it (main.tex Fig. 2 caption, Results "companion latent diagnostic").
    The Z track now runs down the right-hand column as a dashed companion lane
    straight to the verdict.
  * DELETED the classification -> result shortcut. The protocol box itself says
    "in order; all must pass", so a bypass arrow made the four gates optional.
    The left track is now strictly serial: spectra -> classification -> G1..G4
    -> result -> verdict.

All colours, line widths and font sizes come from figure_palette.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

BASE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(PKG, "code"))
import figure_palette as fp  # noqa: E402

fp.apply_sample_style()

WHITE = "#FFFFFF"  # only structural pure-white literal permitted by checker

LEFT, RIGHT = 0.15, 9.85
FULL = RIGHT - LEFT          # 9.70
RX = 5.10                    # right track origin
COL = 4.75                   # width of each of the two tracks
LC = LEFT + COL / 2          # 2.525, centre of left track
RC = RX + COL / 2            # 7.475, centre of right track


def box(ax, x, y, w, h, fc=WHITE, ec=fp.EDGE, lw=fp.LW_AXIS):
    """Rounded box on the grid, no text."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010",
                                fc=fc, ec=ec, lw=lw, zorder=1))


def text(ax, x, y, s, fs=fp.FS_NOTE, tc=fp.INK, bold=False,
         ha="left", va="center", style="normal", leading=None):
    kw = {}
    if leading is not None:
        kw["linespacing"] = leading
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", fontstyle=style,
            zorder=3, **kw)


def arr(ax, x1, y1, x2, y2, lw=fp.LW_DATA, ls="-", color=fp.EDGE):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=9, color=color, lw=lw,
                                 linestyle=ls, shrinkA=0, shrinkB=0, zorder=2))


def chip(ax, y, tag, body, accent=None, h=0.32):
    """One full-width row of the left track: coloured rail + tag + body."""
    box(ax, LEFT, y, COL, h)
    if accent is not None:
        ax.add_patch(Rectangle((LEFT + 0.01, y + 0.01), 0.055, h - 0.02,
                               fc=accent, ec="none", zorder=2))
    text(ax, LEFT + 0.18, y + h / 2, tag, fs=fp.FS_NOTE, bold=True)
    text(ax, LEFT + 1.45, y + h / 2, body, fs=fp.FS_NOTE, tc=fp.MUTED)


fig = plt.figure(figsize=fp.figsize(height_mm=95.0))
ax = fig.add_axes([0.01, 0.02, 0.98, 0.96]); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 10)

# ============================================================ 01  INPUT / MODELS
# No free-floating stage label here: the top of the axes has < 0.2 pt of
# headroom, so a va="bottom" caption at y=9.86 got clipped by the page edge.
# The input box carries the stage name instead.
box(ax, LEFT, 8.84, 3.55, 0.88)
text(ax, 1.925, 9.45, "scRNA-seq  (20k-30k cells)", fs=fp.FS_AXIS,
     bold=True, ha="center")
text(ax, 1.925, 9.08, "GSE150774 / GSE136831 / GSE153162",
     fs=fp.FS_NOTE, tc=fp.MUTED, ha="center")

box(ax, 4.05, 9.28, 5.80, 0.44, fc=fp.BAND_NULL)
text(ax, 4.23, 9.50, "vanilla VAE (no prior)", fs=fp.FS_NOTE)
box(ax, 4.05, 8.84, 5.80, 0.44, fc=fp.BAND_CROSS)
text(ax, 4.23, 9.06, "GRN-prior VAE (injected prior, 33 configs)",
     fs=fp.FS_NOTE)

arr(ax, 3.70, 9.35, 4.02, 9.50)
arr(ax, 3.70, 9.00, 4.02, 9.06)

# ============================================== 02  TWO SPECTRA (PARALLEL TRACKS)
box(ax, LEFT, 8.40, FULL, 0.36, fc=fp.BAND)
text(ax, 5.00, 8.58, "TWO SPECTRA  ·  ONE AUDIT", fs=fp.FS_AXIS,
     bold=True, ha="center")

box(ax, LEFT, 6.90, COL, 1.50)
text(ax, LC, 8.10, r"$\kappa$-$C(W)$  decoder readout",
     fs=fp.FS_AXIS, bold=True, ha="center")
text(ax, LC, 7.45,
     "Ridge proxy / analytical Jacobian  ($H=W^T W$)\n"
     "$\\kappa=\\Sigma_g R^2_g(P)/\\Sigma_g R^2_g(V)$,  "
     "$C=(1-\\kappa)\\times100\\%$",
     fs=fp.FS_NOTE, tc=fp.MUTED, ha="center", leading=1.10)

box(ax, RX, 6.90, COL, 1.50)
text(ax, RC, 8.10, r"$\kappa$-$C$-$Z$  companion latent spectrum",
     fs=fp.FS_AXIS, bold=True, ha="center")
text(ax, RC, 7.45,
     "$Z$ = encoder output;  $\\mathrm{tr}\\,Z^T Z$: effective rank\n"
     "leading-axis share  ·  not used for readout classification",
     fs=fp.FS_NOTE, tc=fp.MUTED, ha="center", leading=1.35)

# NOTE: deliberately NO arrow between the two spectra. They are parallel
# readouts, not a causal chain (the V30 drawing implied the opposite).

# ============================================== 03  CLASSIFICATION (LEFT TRACK)
text(ax, LEFT, 6.71, "03 / CLASSIFICATION", fs=fp.FS_AXIS, tc=fp.MUTED,
     bold=True)
arr(ax, 2.52, 6.90, 2.52, 6.56, lw=fp.LW_EMPH)

chip(ax, 6.20, "locked", r"$\kappa<0.5$  ($C>50\%$)  deep compression",
     accent=fp.CAUTION)
chip(ax, 5.84, "vanilla-eq.", r"$\kappa$ in [0.937, 1.064]  $\pm2\sigma$ null")
chip(ax, 5.48, "expanded", r"$\kappa>1.064$  amplified readout")

# ============================================== 04  FOUR GATES (ALL MUST PASS)
text(ax, LEFT, 5.28, "04 / FOUR GATES", fs=fp.FS_AXIS, tc=fp.MUTED,
     bold=True)
text(ax, LEFT + COL - 0.12, 5.28, "in order; all must pass", fs=fp.FS_NOTE,
     tc=fp.MUTED, ha="right")
arr(ax, 2.52, 5.48, 2.52, 5.10, lw=fp.LW_EMPH)

chip(ax, 4.76, "G1", "injection-path check  ·  prerequisite [this work]",
     accent=fp.CAUTION)
chip(ax, 4.40, "G2", "gradient / wiring check")
chip(ax, 4.04, "G3", r"prior-strength ($\lambda$) scan")
chip(ax, 3.64, "G4", r"structural bound  $\kappa_{\mathrm{gap}}\leq(1-C)\,\mathrm{tr}H_V/\mathrm{gap}_V$")
# serial rail: the four gates are consecutive, not a checklist in parallel
ax.plot([LEFT + 0.10, LEFT + 0.10], [3.72, 5.00], color=fp.EDGE,
        lw=fp.LW_REF, zorder=2)

# ============================================== RESULT (LEFT TRACK)
arr(ax, 2.52, 3.64, 2.52, 3.54, lw=fp.LW_EMPH)
box(ax, LEFT, 2.62, COL, 0.90, fc=fp.BAND)
text(ax, LEFT + 0.18, 3.07, "0/33", fs=fp.FS_PANEL, bold=True)
text(ax, LEFT + 1.45, 3.22, r"configurations reach $C>50\%$", fs=fp.FS_NOTE)
text(ax, LEFT + 1.45, 2.90, "15/15 Jacobian re-audit agree (0 locked)",
     fs=fp.FS_NOTE, tc=fp.MUTED)

# ============================================== COMPANION LANE (RIGHT TRACK)
arr(ax, RC, 6.90, RC, 5.52, lw=fp.LW_REF, ls="--", color=fp.GRAY)
box(ax, RX, 4.20, COL, 1.30)
text(ax, RC, 5.30, "The companion audit stays separate",
     fs=fp.FS_AXIS, bold=True, ha="center")
text(ax, RC, 4.70,
     "$\\kappa$-$C$-$Z$ reports latent reorganization.\n"
     "It does not enter classification or the four gates.",
     fs=fp.FS_NOTE, tc=fp.MUTED, ha="center", leading=1.35)
arr(ax, RC, 4.20, RC, 1.57, lw=fp.LW_REF, ls="--", color=fp.GRAY)

# ============================================================ VERDICT
arr(ax, 2.52, 2.62, 2.52, 1.57, lw=fp.LW_EMPH)
box(ax, LEFT, 0.45, FULL, 1.10, fc=fp.BAND)
text(ax, LEFT + 0.18, 1.00, "verdict", fs=fp.FS_TITLE, bold=True)
text(ax, LEFT + 1.65, 1.22,
     "Perturbation readiness is a measured property, not a design guarantee",
     fs=fp.FS_AXIS)
text(ax, LEFT + 1.65, 0.82, "audit per model, per seed, per prior",
     fs=fp.FS_NOTE, tc=fp.MUTED)

fp.assert_design_width(fig)
fp.export(fig, os.path.join(BASE, "fig0_workflow"))
print("saved fig0_workflow.pdf/png + deck (V32 bbox decollision)")
