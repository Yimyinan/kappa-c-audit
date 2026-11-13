# -*- coding: utf-8 -*-
"""
figure_palette.py -- SINGLE SOURCE OF TRUTH for paper1_kc_diagnostic figure styling
AND figure naming.

Two contracts live here, because both had drifted the same way (per-script local
copies that slowly diverged):

1. STROKE LADDER  -- every figure uses LW_* constants, never a bare ``lw=``.
      hierarchy:  reference < data < emphasis        (LW_REF < LW_DATA < LW_EMPH)
   The 2026-11-09 audit found the hierarchy inverted: threshold lines were drawn at
   lw=1.2 on top of 0.8 axes, and mean lines reached lw=2.0. Readers take the
   longest, heaviest stroke as the baseline, so the whole panel reads as "too bold".

2. NAMING REGISTRY -- every tissue / null-band / arm label comes from here.
   The audit found 6 naming defects across 5 figures: "Lung IPF" vs "Lung
   fibrosis", "Hema." vs "hemo" vs "Hematopoietic", the same null band called
   "family-matched null" in one legend and "cross-family band" in another, panel
   c labels mixing provenance ("recomputed") with verification status
   ("unverified") in the same bracket, and a per-run legend loop emitting two
   identical "vanilla (n=10)" keys.

DESIGN-WIDTH CONTRACT
---------------------
Figures are drawn at the width they are PLACED at (PAPER_WIDTH_IN = 6.69 in =
A4 with 2 cm margins). ``export()`` then emits
    <stem>.pdf            vector, 1:1 with \\textwidth
    <stem>.png            300 dpi, for review
    <stem>_deck.png       ~3240 px wide, for 1280-px slides
Drawing at 13.6 in and placing at 6.69 in (as fig2_kc and fig4 did) divides every
font by two -- 7 pt text printed at 3.4 pt -- and is the second half of the
"looks wrong" symptom.

2026-11-11 EXTENSION (FIGURE_VISUAL_AUDIT_2026-11-11.md)
--------------------------------------------------------
Three contracts were added; none changes an existing colour or constant name.

3. TYPOGRAPHIC GRID -- four landed sizes only (FS_PANEL / FS_TITLE / FS_AXIS /
   FS_NOTE). The audit found up to 11 distinct sizes in one figure because every
   script set ``fontsize=`` per artist, defeating rcParams. ``apply_sample_style()``
   layers the grid on top of ``apply_paper_style()`` for restyled figures, so
   existing figures keep their bytes until they are deliberately regenerated.

4. UNIFIED MATH FONT -- mathtext now uses Arial (rm/it/bf), not the default
   DejaVu/CM mix. 27 of 47 figures previously mixed Arial body text with DejaVu
   Sans/Oblique for kappa, lambda, sigma, R^2. Arial is installed on this machine
   and renders all symbols used in paper1.

5. EXECUTABLE WIDTH GUARD -- ``landed_pt()``, ``assert_design_width()`` and
   ``contrast_ratio()`` make the arithmetic checkable instead of a note in a
   docstring. ``check_figure_typography.py`` applies the same arithmetic to the
   exported PDFs.

Usage
-----
    import figure_palette as fp
    fp.apply_paper_style()
    ax.plot(x, y, color=fp.tissue_color("hemo"), lw=fp.LW_DATA,
            label=fp.arm_label("prior", n=18))
    fp.export(fig, "figures/fig2_kc_phase_diagram")

For a restyled figure (4-step grid + width guard):

    fp.apply_sample_style()
    fig, ax = plt.subplots(figsize=fp.figsize(height_mm=80))
    ...
    fp.assert_design_width(fig)
    fp.export(fig, "...")
"""
from __future__ import annotations

import os

import matplotlib  # noqa: F401  (kept so `import figure_palette` also primes mpl)
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Unified scientific palette
# ----------------------------------------------------------------------------
C_HEMA = "#4C72B0"    # blue   -- hematopoietic tissue / encoder readout
C_NEURAL = "#DD8452"  # orange -- neural cortex
C_LUNG = "#55A868"    # green  -- lung fibrosis / decoder readout
C_ENC = C_HEMA
C_DEC = C_LUNG
C1, C2, C3, C4 = C_HEMA, C_LUNG, "#C44E52", "#8172B2"
GRAY = "#888888"
GREY = "#9A9A9A"
RED = "#C44E52"
GREEN = "#3B7A3B"
BAND = "#EEF2F7"
VANILLA = "#2F2F2F"
PURPLE = "#8172B2"

# --- verdict semantics: RESERVED. Never use these for a series identity.
FAIL = "#C44E52"
PASS = "#3B7A3B"
CAUTION = "#CC7722"

# --- arm identity: prior-injected vs control (no verdict implied)
ARM_VANILLA = "#2F2F2F"
ARM_PRIOR = "#8172B2"
ARM_PRIOR_ALT = "#B4A7D6"

# --- band fills
FAIL_FILL = "#F7E3E3"
PASS_FILL = "#E4EFE4"
BAND_NULL = "#DCE4EF"     # family-matched null: the narrower, darker band
BAND_CROSS = "#EDF1F7"    # cross-family null: wider, lighter (drawn underneath)

INK = "#111111"
EDGE = "#333333"
MUTED = "#6B6B6B"

# ----------------------------------------------------------------------------
# 1. STROKE LADDER  (units: pt at the design width)
#
#   LW_AXIS 0.6   spines
#   LW_TICK 0.6   tick marks
#   LW_REF  0.6   thresholds / zero lines / guide lines  -- must stay <= LW_AXIS
#   LW_DATA 0.8   ordinary data series
#   LW_EMPH 1.0   mean / fit / the one series the eye should land on
#   LW_EDGE 0.3   marker and bar outlines
#   LW_ERR  0.6   error-bar stems
#
# Ceiling rationale: with body text at 8 pt, lw/text = 1.0/8 = 0.125, at the top of
# the 0.10-0.12 journal range. The old 2.0 gave 0.25.
# ----------------------------------------------------------------------------
LW_AXIS = 0.6
LW_TICK = 0.6
LW_REF = 0.6
LW_DATA = 0.8
LW_EMPH = 1.0
LW_EDGE = 0.3
LW_ERR = 0.6
CAPSIZE = 2.0
HATCH_LW = 0.5

PAPER_WIDTH_IN = 6.69          # A4, 2 cm margins
DECK_WIDTH_IN = 10.8           # 1040 px on a 1280-px slide
DECK_DPI = 300                 # -> ~3240 px wide
PAPER_DPI = 300

# ----------------------------------------------------------------------------
# 1b. TYPOGRAPHIC GRID and PLOS placement arithmetic
#
#   LANDED pt = the size a reader sees after the PDF is scaled to \textwidth.
#   All four values below are landed pt, so they are only meaningful if the
#   figure is drawn at TEXTWIDTH_MM (see assert_design_width).
#
#   FS_PANEL 9.0  panel letters (a)(b)(c)
#   FS_TITLE 8.5  panel titles
#   FS_AXIS  8.0  axis labels AND tick labels
#   FS_NOTE  7.5  legends, annotations, in-panel notes
#
#   PLOS has no formal lower bound; 5 pt is the practical print floor. The
#   audit found 18/47 figures below it, 9 below 4 pt.
#
#   WHY THE GRID IS HIGHER THAN THE AUDIT'S FIRST SUGGESTION (8/7.5/7/6.5):
#   mathtext renders subscripts and superscripts at exactly 0.7x the base
#   size (measured 2026-11-11, .tmp/probe_mathtext_scale.py: 6.5->4.55,
#   7.0->4.90, 7.5->5.25, 8.0->5.60). A 6.5 pt annotation containing
#   $\kappa_{\mathrm{gap}}$ therefore prints its subscript at 4.55 pt and
#   still fails the 5 pt floor. FS_NOTE = 7.5 is the smallest grid step whose
#   script characters clear the floor (7.5 * 0.7 = 5.25).
# ----------------------------------------------------------------------------
TEXTWIDTH_MM = 170.0
TEXTWIDTH_PT = TEXTWIDTH_MM / 25.4 * 72.0        # 481.89 pt
TEXTWIDTH_IN = TEXTWIDTH_MM / 25.4               # 6.6929 in

MATH_SCRIPT_RATIO = 0.7     # measured matplotlib mathtext subscript scale

FS_PANEL = 9.0
FS_TITLE = 8.5
FS_AXIS = 8.0
FS_NOTE = 7.5
FS_LADDER = (FS_NOTE, FS_AXIS, FS_TITLE, FS_PANEL)
PLOS_MIN_PT = 5.0
TARGET_LANDED_PT = (6.5, 8.0)

# ----------------------------------------------------------------------------
# 1c. SEMANTIC ROLES -- the colour dictionary already existed; what was missing
#     was a written statement of what each colour is ALLOWED to mean. The audit
#     found #4C72B0 used for "neural cortex" in one panel, "home coverage" in a
#     second and "structural bound" in a third. Additions to this table require
#     an explicit decision, not a new hex literal in a script.
# ----------------------------------------------------------------------------
SEMANTIC_ROLES = {
    C_HEMA: "tissue:hematopoietic / encoder readout",
    C_NEURAL: "tissue:neural cortex",
    C_LUNG: "tissue:lung fibrosis / decoder readout",
    RED: "verdict:FAIL / threshold / critical line ONLY",
    GREEN: "verdict:PASS ONLY",
    PURPLE: "arm:prior ONLY",
    ARM_VANILLA: "arm:vanilla control ONLY",
    GRAY: "reference / guide lines (never a data series)",
    GREY: "secondary grey (never small text: 2.81 contrast on white)",
    INK: "body text",
    EDGE: "outlines",
    MUTED: "secondary annotation text",
}

# ----------------------------------------------------------------------------
# 2. NAMING REGISTRY
#
# One tissue name, one band name, one arm name -- everywhere, in every figure,
# in the same bracket style. A bracket carries ONE dimension only; verification
# status is shown by hatching, not by text (see the "excluded from evidence base"
# legend in fig2 panel c).
# ----------------------------------------------------------------------------
TISSUE_COLOR = {"hemo": C_HEMA, "neural": C_NEURAL, "lung": C_LUNG}

TISSUE_LABEL = {
    "hemo": "Hematopoietic",
    "neural": "Neural cortex",
    "lung": "Lung fibrosis",
}
TISSUE_CELLS = {"hemo": 29985, "neural": 21078, "lung": 19999}

# Two null bands, both "null": the sampling model differs, the object does not.
BAND_LABEL = {
    "family": "family-matched null (UNAGI)",
    "cross": "cross-family null (G-VAE)",
}

ARM_LABEL = {
    "vanilla": "vanilla",
    "prior": "prior",
    "vehicle": "vehicle (empty TF)",
    "wrongprior": "wrong-tissue prior",
    "ablation": "ablation",
}

# Verification status is a FILL/texture, never a parenthetical.
VERIFIED_FILL = "solid"
UNVERIFIED_HATCH = "///"


def tissue_color(key: str) -> str:
    return TISSUE_COLOR[key]


def tissue_label(key: str, n: int | None = None, cells: bool = False) -> str:
    """'Hematopoietic' | 'Hematopoietic (n=22)' | 'Hematopoietic (29,985 cells)'."""
    base = TISSUE_LABEL[key]
    if n is not None:
        return f"{base} (n={n})"
    if cells:
        return f"{base} ({TISSUE_CELLS[key]:,} cells)"
    return base


def arm_label(key: str, n: int | None = None) -> str:
    base = ARM_LABEL[key]
    return f"{base} (n={n})" if n is not None else base


def panel(tag: str) -> str:
    """Uniform panel lettering: (a) (b) (c) ... -- no bold, no trailing period."""
    return f"({tag})"


# ----------------------------------------------------------------------------
# Typographic baseline
# ----------------------------------------------------------------------------
# Declared as ordered pairs so a duplicate key is a hard error, not a silent
# override. The pre-2026-11-09 file declared "axes.linewidth" twice (0.6 then 0.8);
# the 0.6 silently never applied.
_RCPARAM_PAIRS = [
    ("font.family", "sans-serif"),
    ("font.sans-serif", ["Arial", "Helvetica", "DejaVu Sans"]),
    ("font.size", 8.0),
    ("axes.titlesize", 8.5),
    ("axes.labelsize", 8.0),
    ("axes.titleweight", "bold"),
    ("xtick.labelsize", 7.0),
    ("ytick.labelsize", 7.0),
    ("legend.fontsize", 6.5),
    ("axes.linewidth", LW_AXIS),
    ("xtick.major.size", 2.4),
    ("ytick.major.size", 2.4),
    ("xtick.major.width", LW_TICK),
    ("ytick.major.width", LW_TICK),
    ("xtick.minor.width", LW_TICK * 0.8),
    ("ytick.minor.width", LW_TICK * 0.8),
    ("xtick.direction", "out"),
    ("ytick.direction", "out"),
    ("axes.unicode_minus", False),
    ("lines.linewidth", LW_DATA),
    ("lines.markeredgewidth", LW_EDGE),
    ("lines.markersize", 3.2),
    ("patch.linewidth", LW_EDGE),
    ("hatch.linewidth", HATCH_LW),
    ("errorbar.capsize", CAPSIZE),
    ("figure.dpi", 200),
    ("savefig.dpi", PAPER_DPI),
    ("svg.fonttype", "none"),
    ("pdf.fonttype", 42),
    ("pdf.compression", True),
    ("legend.frameon", False),
    ("axes.spines.top", False),
    ("axes.spines.right", False),
    # One font family for body text AND math. Without this, mathtext falls back
    # to DejaVu Sans/Oblique (or CM) for kappa/lambda/sigma/R^2, which put two
    # different x-heights and italic angles on the same line (27/47 figures).
    # Arial is installed here and covers every symbol used in paper1.
    ("mathtext.fontset", "custom"),
    ("mathtext.rm", "Arial"),
    ("mathtext.it", "Arial:italic"),
    ("mathtext.bf", "Arial:bold"),
    ("mathtext.fallback", "cm"),
]

# A note on fallback: "cm" only fires when Arial lacks a glyph. If a future
# figure needs a symbol Arial cannot supply, switch the WHOLE document to stix
# rather than letting one glyph silently reintroduce a second typeface.


def _pairs_to_dict(pairs, where="rcParams"):
    keys = [k for k, _ in pairs]
    dup = {k for k in keys if keys.count(k) > 1}
    if dup:
        raise AssertionError(f"duplicate {where} keys: {sorted(dup)}")
    return dict(pairs)


_PAPER_RCPARAMS = _pairs_to_dict(_RCPARAM_PAIRS)


# ----------------------------------------------------------------------------
# 5. Executable placement arithmetic
# ----------------------------------------------------------------------------
def figsize(width_mm: float = TEXTWIDTH_MM, height_mm: float = 80.0):
    """Return (w_in, h_in) for a figure DRAWN at the width it will be PLACED at.

    height_mm is the only free parameter: 170 mm wide is fixed by \\textwidth,
    so a 4-panel figure must gain height (2x2) rather than lose width (1x4).
    """
    assert width_mm > 0 and height_mm > 0
    return (width_mm / 25.4, height_mm / 25.4)


def page_width_pt(fig) -> float:
    """Physical PDF page width in pt (before any tight bbox cropping)."""
    return float(fig.get_size_inches()[0]) * 72.0


def landed_pt(font_pt: float, pdf_width_pt: float) -> float:
    """Size the reader actually sees once the PDF is scaled to \\textwidth."""
    if pdf_width_pt <= 0:
        raise ValueError("pdf_width_pt must be positive")
    return font_pt * TEXTWIDTH_PT / pdf_width_pt


def assert_design_width(fig, tol_mm: float = 0.5) -> float:
    """Guard against the 13.6-in-drawn / 6.69-in-placed font halving."""
    w_mm = float(fig.get_size_inches()[0]) * 25.4
    if abs(w_mm - TEXTWIDTH_MM) > tol_mm:
        raise AssertionError(
            "figure width %.2f mm != \\textwidth %.2f mm: every font would be "
            "scaled by %.3f. Redraw at 170 mm or split the panels."
            % (w_mm, TEXTWIDTH_MM, TEXTWIDTH_PT / (w_mm / 25.4 * 72.0)))
    return w_mm


def _luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError("expected #RRGGBB, got %r" % hex_color)
    rgb = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
           for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast_ratio(fg: str, bg: str = "#FFFFFF") -> float:
    """WCAG contrast ratio. >= 4.5 for small text, >= 3.0 for graphics."""
    l1, l2 = sorted((_luminance(fg), _luminance(bg)), reverse=True)
    return (l1 + 0.05) / (l2 + 0.05)


def low_contrast_report(colors=None, threshold: float = 4.5):
    """Return [(hex, ratio)] for palette colours unsafe as small text on white.

    Reporting only: the existing low-contrast values are intentionally NOT
    changed here, because recolouring every figure is a separate decision.
    """
    colors = colors or sorted(set(SEMANTIC_ROLES))
    out = []
    for c in colors:
        r = contrast_ratio(c)
        if r < threshold:
            out.append((c, round(r, 2)))
    return out


def apply_paper_style() -> None:
    """Install the unified matplotlib rcParams. Call once after importing pyplot."""
    assert_style()
    plt.rcParams.update(_PAPER_RCPARAMS)


def apply_sample_style() -> None:
    """Paper style + the 4-step typographic grid, for restyled figures.

    Kept separate from apply_paper_style() on purpose: turning the grid on
    globally would silently change every existing figure the next time it is
    regenerated. Restyle one figure, get it approved, then migrate.
    """
    apply_paper_style()
    plt.rcParams.update({
        "font.size": FS_AXIS,
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_AXIS,
        "xtick.labelsize": FS_AXIS,
        "ytick.labelsize": FS_AXIS,
        "legend.fontsize": FS_NOTE,
    })


def assert_style() -> None:
    """Fail loudly if the ladder hierarchy or a naming key has regressed."""
    assert LW_REF <= LW_AXIS + 1e-9, "reference lines must not outrank the axes"
    assert LW_REF < LW_DATA < LW_EMPH + 1e-9, "stroke ladder order broken"
    assert LW_EMPH <= 1.0 + 1e-9, "emphasis stroke exceeds the 1.0 pt ceiling"
    assert list(FS_LADDER) == sorted(FS_LADDER), "type grid must be ascending"
    assert FS_NOTE >= PLOS_MIN_PT, "smallest grid size is below the 5 pt print floor"
    assert FS_NOTE * MATH_SCRIPT_RATIO >= PLOS_MIN_PT, (
        "mathtext subscripts at %.2f x %.2f = %.2f pt would fall below the "
        "%.1f pt floor; raise FS_NOTE" %
        (FS_NOTE, MATH_SCRIPT_RATIO, FS_NOTE * MATH_SCRIPT_RATIO, PLOS_MIN_PT))
    assert abs(PAPER_WIDTH_IN * 25.4 - TEXTWIDTH_MM) < 0.1, \
        "PAPER_WIDTH_IN and TEXTWIDTH_MM disagree"
    assert set(TISSUE_LABEL) == set(TISSUE_COLOR) == set(TISSUE_CELLS)
    for k, v in TISSUE_LABEL.items():
        assert "(" not in v, f"tissue label {k!r} must stay unqualified: {v!r}"
    for k, v in BAND_LABEL.items():
        assert "null" in v, f"band {k!r} must be named a null: {v!r}"


def style_ax(ax) -> None:
    """Despine top/right and use outward ticks (paper convention)."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(LW_AXIS)
    ax.tick_params(direction="out", length=2.4, width=LW_TICK)


def safe_legend(ax, loc="upper left", bbox_to_anchor=None, ncol=1,
                fontsize=6.5, frameon=False, handlelength=1.1, **kw):
    """Draw a legend that never occludes data."""
    if bbox_to_anchor is not None:
        ax.legend(loc=loc, bbox_to_anchor=bbox_to_anchor, ncol=ncol,
                  fontsize=fontsize, frameon=frameon, handlelength=handlelength, **kw)
    else:
        ax.legend(loc=loc, ncol=ncol, fontsize=fontsize,
                  frameon=frameon, handlelength=handlelength, **kw)


def legend_out_right(ax, fontsize=6.5, handlelength=1.1):
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=fontsize, frameon=False, handlelength=handlelength)


def legend_below(ax, ncol=3, fontsize=6.5, handlelength=1.1):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20),
              fontsize=fontsize, frameon=False, handlelength=handlelength)


# ----------------------------------------------------------------------------
# Dual export
# ----------------------------------------------------------------------------
def export(fig, stem: str, deck: bool = True, deck_dir: str | None = None):
    """Save the paper PDF/PNG and (optionally) a slide-resolution PNG.

    stem       path without extension, e.g. "figures/fig2_kc_phase_diagram"
    deck_dir   where the *_deck.png goes (default: <stem dir>/deck)
    """
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=PAPER_DPI)
    if deck:
        d = deck_dir or os.path.join(os.path.dirname(stem), "deck")
        os.makedirs(d, exist_ok=True)
        wide = fig.get_size_inches()[0]
        fig.savefig(os.path.join(d, os.path.basename(stem) + "_deck.png"),
                    dpi=max(200, int(round(DECK_DPI * DECK_WIDTH_IN / wide))))
    plt.close(fig)


def save_paper(fig, path_pdf, path_png=None, dpi=PAPER_DPI):
    """Legacy single-output save (kept for scripts not yet migrated)."""
    fig.savefig(path_pdf, dpi=dpi, bbox_inches="tight")
    if path_png is not None:
        fig.savefig(path_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
