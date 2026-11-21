# -*- coding: utf-8 -*-
"""Build the V39 Figure 3 from the complete Supporting Table S1 values."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "04_figure_sources" / "Code"
TABLE = ROOT / "01_manuscript" / "supplementary" / "table_s1_audit_body.tex"
OUT = ROOT / "v39_figures" / "Fig3"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CODE))
import figure_palette as fp  # noqa: E402

fp.FS_NOTE = 8.0
fp.apply_sample_style()


def load_rows(path: Path):
    text = path.read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        if "&" not in line or "\\" not in line:
            continue
        fields = [field.strip() for field in line.rstrip("\\").split("&")]
        if len(fields) < 7 or fields[1] not in {"H", "N", "L"}:
            continue
        config, tissue, _, kappa, _, compression, _ = fields[:7]
        rows.append(
            {
                "config": re.sub(r"\\_", "_", config),
                "tissue": {"H": "hemo", "N": "neural", "L": "lung"}[tissue],
                "kappa": float(kappa),
                "compression": float(compression),
            }
        )
    if len(rows) != 33:
        raise RuntimeError(f"Expected 33 S1 Table rows, found {len(rows)}")
    return rows


ROWS = load_rows(TABLE)
TISSUES = [
    ("hemo", "Hematopoietic", 2),
    ("neural", "Neural cortex", 1),
    ("lung", "Lung fibrosis", 0),
]
COLORS = {key: fp.tissue_color(key) for key, _, _ in TISSUES}
RNG = np.random.default_rng(20261120)


def jittered_points(rows, key, value_key):
    subset = [r for r in rows if r["tissue"] == key]
    base = next(y for tissue, _, y in TISSUES if tissue == key)
    offset = RNG.uniform(-0.12, 0.12, size=len(subset))
    return [r[value_key] for r in subset], base + offset


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fp.figsize(height_mm=135.0))
fig.subplots_adjust(left=0.22, right=0.96, top=0.87, bottom=0.17, hspace=0.56)

# Panel A: every audited configuration, with the observed range as a guide.
for key, label, ybase in TISSUES:
    subset = [r for r in ROWS if r["tissue"] == key]
    xs = [r["compression"] for r in subset]
    ys = ybase + RNG.uniform(-0.12, 0.12, size=len(xs))
    ax1.scatter(
        xs,
        ys,
        s=23,
        color=COLORS[key],
        edgecolors="white",
        linewidths=0.55,
        zorder=4,
    )
    ax1.plot(
        [min(xs), max(xs)],
        [ybase, ybase],
        color=COLORS[key],
        lw=fp.LW_EMPH,
        solid_capstyle="round",
        zorder=2,
    )
    text = f"{min(xs):+.1f} to {max(xs):+.1f}%"
    if min(xs) == max(xs):
        text = f"{min(xs):+.1f}%"
    ax1.text(
        16.6,
        ybase,
        text,
        ha="left",
        va="center",
        fontsize=fp.FS_NOTE,
        color=fp.INK,
    )

ax1.set_yticks([y for _, _, y in TISSUES])
ax1.set_yticklabels(
    [f"{label}  (n={sum(r['tissue'] == key for r in ROWS)})" for key, label, _ in TISSUES]
)
ax1.set_ylim(-0.55, 2.70)
ax1.set_xlim(-2.0, 24.0)
ax1.set_xlabel("Global compression, C (%)")
ax1.set_title("(a)  Individual configurations and tissue-level range", loc="left", pad=7)
ax1.axvline(0, color=fp.GRAY, lw=fp.LW_REF, zorder=0)
ax1.text(
    0.98,
    0.95,
    "Deep-compression threshold: C > 50% (off scale)",
    transform=ax1.transAxes,
    ha="right",
    va="top",
    fontsize=fp.FS_NOTE,
    color=fp.MUTED,
)
fp.style_ax(ax1)
ax1.tick_params(axis="y", length=0, pad=5)

# Panel B: both published reference bands and every configuration.
null_cross = (0.937, 1.064)
null_family = (0.915, 1.086)
ax2.axvspan(*null_family, color=fp.BAND_NULL, zorder=0)
ax2.axvspan(*null_cross, color=fp.BAND_CROSS, zorder=1)
ax2.axvline(0.5, color=fp.RED, lw=fp.LW_REF, ls="--", zorder=2)
for key, label, ybase in TISSUES:
    xs, ys = jittered_points(ROWS, key, "kappa")
    ax2.scatter(
        xs,
        ys,
        s=23,
        color=COLORS[key],
        edgecolors="white",
        linewidths=0.55,
        zorder=4,
    )

ax2.set_yticks([y for _, _, y in TISSUES])
ax2.set_yticklabels(
    [f"{label}  (n={sum(r['tissue'] == key for r in ROWS)})" for key, label, _ in TISSUES]
)
ax2.set_ylim(-0.55, 2.70)
ax2.set_xlim(0.43, 1.12)
ax2.set_xlabel(r"Aggregate predictability ratio, $\kappa$")
ax2.set_title("(b)  Configurations against both vanilla-equivalence bands", loc="left", pad=7)
ax2.text(
    0.505,
    -0.38,
    r"Deep compression: $\kappa<0.5$",
    ha="left",
    va="center",
    fontsize=fp.FS_NOTE,
    color=fp.RED,
)
ax2.legend(
    handles=[
        Patch(facecolor=fp.BAND_CROSS, edgecolor="none", label="Cross-family G-VAE null band"),
        Patch(facecolor=fp.BAND_NULL, edgecolor="none", label="Family-matched UNAGI band"),
        Line2D(
            [],
            [],
            color=fp.RED,
            ls="--",
            lw=fp.LW_REF,
            label=r"Deep-compression boundary ($\kappa=0.5$)",
        ),
    ],
    loc="upper right",
    bbox_to_anchor=(0.98, -0.34),
    ncol=3,
    frameon=False,
    fontsize=fp.FS_NOTE,
    handlelength=1.6,
    columnspacing=1.0,
)
fp.style_ax(ax2)
ax2.tick_params(axis="y", length=0, pad=5)

fig.suptitle(
    "No audited configuration reaches deep decoder-readout compression",
    x=0.22,
    y=0.982,
    ha="left",
    fontsize=fp.FS_TITLE,
    fontweight="bold",
)
fp.assert_design_width(fig)

stem = OUT / "Fig3_prior_injection_audit"
fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
fig.savefig(stem.with_suffix(".svg"), facecolor="white")
fig.savefig(OUT / f"{stem.name}_300dpi.png", dpi=300, facecolor="white")
tmp_tif = OUT / f".{stem.name}_600dpi_rgba.tif"
fig.savefig(
    tmp_tif,
    dpi=600,
    format="tiff",
    facecolor="white",
    transparent=False,
    pil_kwargs={"compression": "tiff_lzw"},
)
from PIL import Image

with Image.open(tmp_tif) as image:
    image.convert("RGB").save(
        OUT / f"{stem.name}_600dpi.tif",
        format="TIFF",
        compression="tiff_lzw",
        dpi=(600, 600),
    )
tmp_tif.unlink(missing_ok=True)
plt.close(fig)

caption = (
    "Fig 3. No audited configuration reaches deep decoder-readout compression. "
    "(a) Global compression, C = 100(1 - kappa), for all 33 audited configurations: "
    "22 hematopoietic, 10 neural-cortex and one lung-fibrosis configuration. "
    "Points are individual configurations; horizontal segments span the observed "
    "minimum-to-maximum range within each tissue. No configuration approaches the "
    "deep-compression threshold C > 50%, which lies outside the displayed scale. "
    "(b) Aggregate predictability ratio kappa for the same configurations against "
    "the cross-family G-VAE band (0.937-1.064) and the family-matched UNAGI band "
    "(0.915-1.086); the dashed line marks kappa = 0.5. Both calibrations preserve "
    "the 0/33 conclusion. Configuration-level values are reported in S1 Table."
)
(OUT / "Fig3_caption.txt").write_text(caption, encoding="utf-8")
print(f"Wrote Figure 3 outputs to {OUT}")
