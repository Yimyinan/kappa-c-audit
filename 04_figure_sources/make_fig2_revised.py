# -*- coding: utf-8 -*-
"""Redesigned Fig. 2 workflow schematic for the dual-spectrum audit.

Conceptual schematic only. Numeric labels are manuscript-reported results,
not calculated from experimental files by this plotting script.

Scientific/visual corrections relative to the supplied V32 script:
- The latent companion is described using Cov(Z), participation ratio and
  leading-axis share. tr(Z^T Z) is not mislabeled as effective rank.
- Decoder and latent spectra are parallel readouts of paired models.
- The latent companion is explicitly excluded from decoder-readout
  classification.
- G1 injection-path verification is a prerequisite; G2-G4 are considered
  only after delivery is established. The checks are ordered and interpreted
  jointly, not presented as interchangeable binary pass/fail gates.
- Jacobian-recomputed and Ridge-only configurations are explicitly separated.
- The 22/29 latent result is reported separately from the 0/33 decoder result.
"""
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figure_palette as fp  # noqa: E402

fp.apply_sample_style()

WHITE = "#FFFFFF"
BLUE = fp.C_HEMA
PURPLE = fp.PURPLE
GREEN = fp.C_DEC
DARK = fp.INK
MUTED = fp.MUTED
EDGE = fp.EDGE

fig = plt.figure(figsize=fp.figsize(height_mm=132.0))
ax = fig.add_axes([0.018, 0.012, 0.964, 0.976])
ax.set_xlim(0, 10)
ax.set_ylim(0, 13)
ax.axis("off")


def rounded(x, y, w, h, fc=WHITE, ec=EDGE, lw=fp.LW_AXIS):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.025",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=1)
    ax.add_patch(p)
    return p


def label(x, y, s, size=fp.FS_NOTE, color=DARK, weight="normal",
          ha="left", va="center", **kwargs):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va=va, zorder=4, **kwargs)


def arrow(x1, y1, x2, y2, color=EDGE, lw=fp.LW_DATA, ls="-", ms=7):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=ms, color=color, linewidth=lw,
                        linestyle=ls, shrinkA=0, shrinkB=0, zorder=3)
    ax.add_patch(a)
    return a


def rule(x1, y1, x2, y2, color=EDGE, lw=fp.LW_REF, ls="-"):
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw,
            linestyle=ls, zorder=2)


def pill(x, y, w, h, text_, fc, color=DARK, size=fp.FS_NOTE, weight="normal"):
    rounded(x, y, w, h, fc=fc, ec="none", lw=0)
    label(x + w/2, y + h/2, text_, size=size, color=color,
          weight=weight, ha="center")

# 1. Shared data and matched model pair
rounded(0.20, 12.05, 9.60, 0.82, fc=fp.BAND)
label(5.00, 12.57, "Single-cell expression datasets", size=fp.FS_TITLE,
      weight="bold", ha="center")
label(5.00, 12.26, "GSE150774  ·  GSE136831  ·  GSE153162",
      size=fp.FS_NOTE, color=MUTED, ha="center")
arrow(5.00, 12.05, 5.00, 11.88, color=EDGE)
rule(2.53, 11.88, 7.47, 11.88, color=EDGE)
arrow(2.53, 11.88, 2.53, 11.70, color=EDGE)
arrow(7.47, 11.88, 7.47, 11.70, color=EDGE)

rounded(0.20, 10.91, 4.66, 0.76, fc=fp.BAND_NULL)
label(2.53, 11.39, "Vanilla VAE", size=fp.FS_TITLE, weight="bold", ha="center")
label(2.53, 11.08, "No GRN prior", size=fp.FS_NOTE, color=MUTED, ha="center")
rounded(5.14, 10.91, 4.66, 0.76, fc=fp.BAND_CROSS)
label(7.47, 11.39, "GRN-prior-configured VAE", size=fp.FS_TITLE, weight="bold", ha="center")
label(7.47, 11.08, "Prior registered in the pipeline", size=fp.FS_NOTE,
      color=MUTED, ha="center")

rule(2.53, 10.91, 2.53, 10.76, color=EDGE)
rule(7.47, 10.91, 7.47, 10.76, color=EDGE)
rule(2.53, 10.76, 7.47, 10.76, color=EDGE)
arrow(5.00, 10.76, 5.00, 10.63, color=EDGE)
rounded(0.20, 10.29, 9.60, 0.34, fc=fp.BAND)
label(5.00, 10.46, "TWO PARALLEL READOUTS OF THE PAIRED MODELS",
      size=fp.FS_AXIS, weight="bold", ha="center")
arrow(2.53, 10.29, 2.53, 10.00, color=GREEN)
arrow(7.47, 10.29, 7.47, 10.00, color=PURPLE)

# 2. Parallel spectra
rounded(0.20, 8.45, 4.66, 1.50, fc="#F2F7F4", ec=GREEN)
label(2.53, 9.68, r"Decoder readout  $\kappa$–$C(W)$",
      size=fp.FS_TITLE, weight="bold", color=GREEN, ha="center")
label(0.42, 9.38, r"Ridge proxy or analytical Jacobian ($H=W^T W$)",
      size=fp.FS_NOTE, color=DARK)
label(0.42, 9.04, r"Per-gene $R^2$ aggregated across genes",
      size=fp.FS_NOTE, color=DARK)
label(0.42, 8.70,
      r"$\kappa=\sum_g R_g^2(P)/\sum_g R_g^2(V)$;  $C=100(1-\kappa)\%$",
      size=fp.FS_NOTE, color=MUTED)

rounded(5.14, 8.45, 4.66, 1.50, fc="#F5F2FA", ec=PURPLE)
label(7.47, 9.68, r"Latent companion  $\kappa$–$C$–$Z$",
      size=fp.FS_TITLE, weight="bold", color=PURPLE, ha="center")
label(5.36, 9.31, r"Covariance spectrum of encoder output $Z$",
      size=fp.FS_NOTE, color=DARK)
label(5.36, 8.96, "Participation ratio and leading-axis share",
      size=fp.FS_NOTE, color=DARK)
label(5.36, 8.61, "Rotation-invariant latent reorganization",
      size=fp.FS_NOTE, color=MUTED)

# 3. Separate interpretation of the two readouts
arrow(2.53, 8.45, 2.53, 8.18, color=GREEN)
arrow(7.47, 8.45, 7.47, 8.18, color=PURPLE)

rounded(0.20, 7.13, 4.66, 0.94, fc=WHITE, ec=GREEN)
label(0.40, 7.84, "DECODER-READOUT CLASSIFICATION",
      size=fp.FS_AXIS, weight="bold", color=GREEN)
pill(0.40, 7.39, 1.30, 0.25, r"$\kappa<0.5$", "#F7EEDC",
     color=fp.CAUTION, size=fp.FS_NOTE, weight="bold")
pill(1.80, 7.39, 1.70, 0.25, r"$0.937$–$1.064$", fp.BAND_NULL,
     color=DARK, size=fp.FS_NOTE, weight="bold")
pill(3.60, 7.39, 1.06, 0.25, r"$>1.064$", fp.BAND_CROSS,
     color=DARK, size=fp.FS_NOTE, weight="bold")
label(1.05, 7.20, "deep compression", size=6.8, color=MUTED, ha="center")
label(2.65, 7.20, "vanilla-equivalent", size=6.8, color=MUTED, ha="center")
label(4.13, 7.20, "expansion", size=6.8, color=MUTED, ha="center")

rounded(5.14, 7.13, 4.66, 0.94, fc=WHITE, ec=PURPLE)
label(5.36, 7.84, "LATENT COMPANION (SEPARATE)",
      size=fp.FS_AXIS, weight="bold", color=PURPLE)
label(5.36, 7.54, "Describes Cov(Z) reorganization",
      size=fp.FS_NOTE, color=DARK)
label(5.36, 7.27, "Not used for decoder-readout classification",
      size=fp.FS_NOTE, color=MUTED)

# 4. Prior-activity checks for attribution of decoder-readout changes.
label(0.20, 6.91, "PRIOR-ACTIVITY VERIFICATION",
      size=fp.FS_AXIS, weight="bold")
label(4.86, 6.91, "G1 is prerequisite", size=6.9, color=MUTED, ha="right")

card_x, card_w, card_h = 0.20, 4.66, 0.42
card_ys = [6.13, 5.57, 5.01, 4.45]
gates = [
    ("G1", "Trace source-to-objective path (reader present)", fp.CAUTION),
    ("G2", "Verify gradient path to model parameters", BLUE),
    ("G3", r"Scan response across prior strength $\lambda$", BLUE),
    ("G4", "Bound attainable change in readout geometry", BLUE),
]
for i, (y, (tag, body, accent)) in enumerate(zip(card_ys, gates)):
    rounded(card_x, y, card_w, card_h, fc=WHITE, ec=accent, lw=fp.LW_AXIS)
    ax.add_patch(Rectangle((card_x + 0.012, y + 0.012), 0.055,
                           card_h - 0.024, facecolor=accent,
                           edgecolor="none", zorder=2))
    label(card_x + 0.20, y + card_h/2, tag, size=fp.FS_NOTE,
          weight="bold", color=accent)
    label(card_x + 0.70, y + card_h/2, body, size=fp.FS_NOTE, color=DARK)
    if i < 3:
        arrow(2.53, y - 0.012, 2.53, card_ys[i+1] + card_h + 0.012,
              color=EDGE, lw=fp.LW_REF, ms=6)

# The latent companion has a separate interpretation note and does not feed G1-G4.
rounded(5.14, 4.75, 4.66, 1.90, fc="#F5F2FA", ec=PURPLE)
label(5.38, 6.33, "INTERPRETATION LIMIT",
      size=fp.FS_AXIS, weight="bold", color=PURPLE)
label(5.38, 5.95, "Latent compression is a descriptive pattern.",
      size=fp.FS_NOTE, color=DARK)
label(5.38, 5.60, "It may reflect tissue- or training-dependent",
      size=fp.FS_NOTE, color=DARK)
label(5.38, 5.25, "dynamics; it does not by itself show that",
      size=fp.FS_NOTE, color=DARK)
label(5.38, 4.90, "prior content caused the reorganization.",
      size=fp.FS_NOTE, color=DARK)

# 5. Separate observed results
arrow(2.53, 4.55, 2.53, 4.31, color=GREEN)
arrow(7.47, 4.75, 7.47, 4.31, color=PURPLE, ls="--", lw=fp.LW_REF)

rounded(0.20, 2.88, 4.66, 1.43, fc="#F2F7F4", ec=GREEN)
label(0.42, 4.05, "DECODER RESULT", size=fp.FS_AXIS,
      weight="bold", color=GREEN)
label(0.42, 3.72, "0/33 reached deep compression",
      size=fp.FS_TITLE, weight="bold")
label(0.42, 3.37, "15 checkpointed: all remained subcritical by Jacobian",
      size=fp.FS_NOTE, color=MUTED)
label(0.42, 3.06, "18 Ridge-only: all above threshold under empirical bound",
      size=fp.FS_NOTE, color=MUTED)

rounded(5.14, 2.88, 4.66, 1.43, fc="#F5F2FA", ec=PURPLE)
label(5.36, 4.05, "LATENT RESULT", size=fp.FS_AXIS,
      weight="bold", color=PURPLE)
label(5.36, 3.72, "22/29 showed directional compression",
      size=fp.FS_TITLE, weight="bold")
label(5.36, 3.37, "Among runs with computable latent spectra",
      size=fp.FS_NOTE, color=MUTED)
label(5.36, 3.06, "Does not show that prior content caused the change",
      size=fp.FS_NOTE, color=MUTED)

# 6. Scope: readiness is a downstream property
arrow(2.53, 2.88, 2.53, 2.59, color=GREEN, lw=fp.LW_REF)
arrow(7.47, 2.88, 7.47, 2.59, color=PURPLE, lw=fp.LW_REF, ls="--")
rounded(0.20, 1.66, 9.60, 0.86, fc=fp.BAND)
label(5.00, 2.20, "Neither spectrum certifies perturbation readiness",
      size=fp.FS_TITLE, weight="bold", ha="center")
label(5.00, 1.90,
      "Predictive utility requires separate evaluation on held-out perturbations",
      size=fp.FS_AXIS, color=MUTED, ha="center")
label(5.00, 1.25,
      "Schematic workflow; numerical callouts summarize the 33-configuration decoder audit and 29-run latent companion.",
      size=fp.FS_NOTE, color=MUTED, ha="center")

fp.assert_design_width(fig)
out = HERE
fig.savefig(out / "Fig2_dual_spectrum_workflow.pdf")
fig.savefig(out / "Fig2_dual_spectrum_workflow.svg")
fig.savefig(out / "Fig2_dual_spectrum_workflow_300dpi.png", dpi=300)
fig.savefig(out / "Fig2_dual_spectrum_workflow_600dpi.tif", dpi=600,
            pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("Exported Fig2 PDF, SVG, 300-dpi PNG and 600-dpi LZW TIFF to", out)
