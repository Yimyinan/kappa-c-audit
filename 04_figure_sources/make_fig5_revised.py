#!/usr/bin/env python3
"""Rebuild Fig. 5 from the supplied VEGA audit JSON package.

Panel A: pairwise decoder-readout ratios (nine VEGA/vanilla pairings and six
ordered vanilla/vanilla comparisons). Pairwise ratios reuse three trained
seeds and are not independent model replicates.

Panel B: held-out sum of gene-wise R^2 across 6,998 genes for five arms.
The degree-preserving condition contains 30 rewires per seed; these are first
averaged within seed, so the figure displays three seed-level means, not n=90
independent fits.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import figure_palette as fp

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "fig5_vega"
OUT = ROOT / "fig5_revised_output"
OUT.mkdir(parents=True, exist_ok=True)

fp.apply_sample_style()

# ---- Load canonical results and validate provenance -------------------------
summary = json.loads((DATA / "vega_audit_summary.json").read_text())
seeds = [1, 2, 3]
assert summary["seeds"] == seeds
assert summary["epochs"] == 150


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def check_meta(d: dict, path: Path) -> None:
    expected = {"epochs": 150, "n_cells": 16893, "n_genes": 6998, "n_pathways": 675}
    for key, val in expected.items():
        if d.get(key) != val:
            raise ValueError(f"Metadata mismatch in {path}: {key}={d.get(key)!r}, expected {val!r}")

# Panel A: preserve the ordered pairings from the canonical audit summary.
vanilla = {int(k): float(v) for k, v in summary["sum_r2"]["vanilla"].items()}
vega = {int(k): float(v) for k, v in summary["sum_r2"]["vega"].items()}
assert set(vanilla) == set(vega) == set(seeds)

cross = [(int(a), int(b), float(r)) for a, b, r in summary["cross"]]
null = [(int(a), int(b), float(r)) for a, b, r in summary["null"]]
assert len(cross) == 9 and len(null) == 6
for a, b, r in cross:
    calc = vega[a] / vanilla[b]
    if not math.isclose(calc, r, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"Cross ratio mismatch for VEGA seed {a}/vanilla seed {b}")
for a, b, r in null:
    calc = vanilla[a] / vanilla[b]
    if not math.isclose(calc, r, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"Vanilla null ratio mismatch for seed {a}/{b}")

# Check that the canonical single-rewire records in the summary are included
# among the 30-per-seed rewiring ensemble.
rewire_by_seed: dict[int, list[float]] = {s: [] for s in seeds}
rewire_records = []
for path in sorted((DATA / "rewiring").glob("*.json")):
    d = load_json(path)
    check_meta(d, path)
    s = int(d["seed"])
    if s not in rewire_by_seed:
        raise ValueError(f"Unexpected rewiring seed in {path}: {s}")
    v = float(d["sum_r2"])
    rewire_by_seed[s].append(v)
    rewire_records.append({"seed": s, "shuffle_seed": d.get("shuffle_seed"),
                           "sum_r2": v, "file": str(path.relative_to(ROOT))})
for s, vals in rewire_by_seed.items():
    if len(vals) != 30:
        raise ValueError(f"Expected 30 degree-preserving rewires for seed {s}; found {len(vals)}")
    canonical_base = load_json(DATA / "rewiring" / f"vega_vega_shuffled_seed{s}.json")
    if not any(math.isclose(float(canonical_base["sum_r2"]), v, abs_tol=1e-9) for v in vals):
        raise ValueError(f"Summary/base shuffle for seed {s} is not in the 30-rewire set")
    if not math.isclose(float(summary["sum_r2"]["vega_shuffled"][str(s)]),
                        float(canonical_base["sum_r2"]), rel_tol=1e-10):
        raise ValueError(f"Summary shuffled value does not match base JSON for seed {s}")

# Panel B: canonical 3-seed arms.
arm_files = {
    "Identity-reassigned": "vega_vega_rows_perm_seed{seed}.json",
    "Size-matched random": "vega_vega_random_rows_seed{seed}.json",
}
arm_values: dict[str, dict[int, float]] = {
    "Prior-free": vanilla,
    "Pathway mask": vega,
    "Identity-reassigned": {},
    "Degree-preserving rewiring": {},
    "Size-matched random": {},
}
for arm, pattern in arm_files.items():
    for s in seeds:
        path = DATA / "canonical" / pattern.format(seed=s)
        d = load_json(path)
        check_meta(d, path)
        arm_values[arm][s] = float(d["sum_r2"])
for s in seeds:
    arm_values["Degree-preserving rewiring"][s] = float(np.mean(rewire_by_seed[s]))

# Validate the arm labels and central manuscript-level numerical pattern.
means = {a: float(np.mean(list(v.values()))) for a, v in arm_values.items()}
sds = {a: float(np.std(list(v.values()), ddof=1)) for a, v in arm_values.items()}
assert math.isclose(means["Prior-free"], 49.2826296488444, rel_tol=1e-9)
assert math.isclose(means["Pathway mask"], 253.8742472330729, rel_tol=1e-9)
assert math.isclose(means["Identity-reassigned"], 243.65701802571616, rel_tol=1e-9)
assert math.isclose(means["Degree-preserving rewiring"], 233.10304141574437, rel_tol=1e-9)
assert math.isclose(means["Size-matched random"], 321.6192220052083, rel_tol=1e-9)

# Export machine-readable data tables used in this figure.
pd.DataFrame([
    {"comparison": "VEGA / vanilla", "vega_seed": a, "vanilla_seed": b,
     "pair_type": "same-seed" if a == b else "cross-seed", "ratio": r}
    for a, b, r in cross
] + [
    {"comparison": "vanilla / vanilla", "vega_seed": a, "vanilla_seed": b,
     "pair_type": "ordered vanilla-null", "ratio": r}
    for a, b, r in null
]).to_csv(OUT / "Fig5_panelA_pairwise_ratios.csv", index=False)

panel_b_rows = []
for arm, vals in arm_values.items():
    for s, value in vals.items():
        panel_b_rows.append({"arm": arm, "seed": s, "sum_gene_R2": value,
                             "within_seed_rewires": 30 if arm == "Degree-preserving rewiring" else 0})
pd.DataFrame(panel_b_rows).to_csv(OUT / "Fig5_panelB_seed_values.csv", index=False)
pd.DataFrame(rewire_records).to_csv(OUT / "Fig5_degree_preserving_all_90_rewirings.csv", index=False)

# ---- Plot -------------------------------------------------------------------
# A vertically stacked, full-width layout avoids forcing the five long arm
# labels into a narrow half-column at the final 170-mm print width.
fig = plt.figure(figsize=fp.figsize(height_mm=126.0))
ax_a = fig.add_axes([0.135, 0.605, 0.82, 0.205])
ax_b = fig.add_axes([0.305, 0.185, 0.65, 0.305])

# Harmonise font fallback for this environment while keeping an Arial-like
# sans-serif face and a single matching math family in exported vector files.
plt.rcParams["font.family"] = "Liberation Sans"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Liberation Sans"
plt.rcParams["mathtext.it"] = "Liberation Sans:italic"
plt.rcParams["mathtext.bf"] = "Liberation Sans:bold"

# Panel A: pairwise ratio strip plot on a log scale.
ax_a.axhspan(0.94, 1.06, color=fp.BAND_NULL, alpha=0.95, zorder=0)
ax_a.axhline(1.0, color=fp.EDGE, lw=fp.LW_REF, ls="-", zorder=1)
ax_a.axhline(0.94, color=fp.GRAY, lw=fp.LW_REF, ls=(0, (2, 2)), zorder=1)
ax_a.axhline(1.06, color=fp.GRAY, lw=fp.LW_REF, ls=(0, (2, 2)), zorder=1)

same = [(a, b, r) for a, b, r in cross if a == b]
diff = [(a, b, r) for a, b, r in cross if a != b]
x_same = np.array([-0.13, 0.00, 0.13])
x_diff = np.array([-0.22, -0.13, -0.045, 0.045, 0.13, 0.22])
x_null = np.array([-0.20, -0.12, -0.04, 0.04, 0.12, 0.20])
g1, g2 = 0.0, 1.9
ax_a.scatter(g1 + x_same, [r for _, _, r in same], s=33, marker="o",
             facecolor=fp.PURPLE, edgecolor="white", linewidth=0.6,
             zorder=4, label="same-seed pair (n=3)")
ax_a.scatter(g1 + x_diff, [r for _, _, r in diff], s=35, marker="^",
             facecolor=fp.ARM_PRIOR_ALT, edgecolor=fp.PURPLE, linewidth=0.6,
             zorder=4, label="cross-seed pair (n=6)")
ax_a.scatter(g2 + x_null, [r for _, _, r in null], s=30, marker="s",
             facecolor=fp.GRAY, edgecolor="white", linewidth=0.55,
             zorder=4, label="vanilla–vanilla (n=6)")

med_cross = statistics.median([r for _, _, r in cross])
med_null = statistics.median([r for _, _, r in null])
for x, med, col in [(g1, med_cross, fp.PURPLE), (g2, med_null, fp.INK)]:
    ax_a.plot([x - 0.27, x + 0.27], [med, med], color=col,
              lw=fp.LW_EMPH, solid_capstyle="round", zorder=5)
ax_a.text(g1, med_cross * 1.10, f"median {med_cross:.2f}", ha="center",
          va="bottom", fontsize=fp.FS_NOTE, color=fp.PURPLE, fontweight="bold")
ax_a.text(g2, 1.30,
          f"median {med_null:.2f}  ·  range {min(r for _,_,r in null):.2f}–{max(r for _,_,r in null):.2f}",
          ha="center", va="bottom", fontsize=fp.FS_NOTE, color=fp.MUTED)
ax_a.text(0.95, 0.80, "G-VAE cross-family null band: 0.94–1.06",
          ha="center", va="center", fontsize=fp.FS_NOTE, color=fp.MUTED)
ax_a.set_yscale("log")
ax_a.set_ylim(0.65, 7.3)
ax_a.set_xlim(-0.55, 2.45)
ax_a.set_xticks([g1, g2])
ax_a.set_xticklabels(["VEGA / vanilla", "vanilla / vanilla"])
ax_a.set_yticks([0.7, 1, 2, 3, 5, 7])
ax_a.set_yticklabels(["0.7", "1", "2", "3", "5", "7"])
ax_a.set_ylabel("Aggregate predictability ratio, $\\kappa$")
ax_a.set_title("(a) The readout detects conditioning delivered to the decoder",
               loc="left", fontsize=fp.FS_TITLE, fontweight="bold", pad=6)
ax_a.grid(axis="y", which="major", color="#E6E8EB", lw=0.45, zorder=0)
fp.style_ax(ax_a)

# Panel B: seed-level dot + mean ± between-seed SD. Rewires are nested in seed.
order = ["Size-matched random", "Degree-preserving rewiring",
         "Identity-reassigned", "Pathway mask", "Prior-free"]
colors = {
    "Prior-free": fp.ARM_VANILLA,
    "Pathway mask": fp.PURPLE,
    "Identity-reassigned": fp.ARM_PRIOR_ALT,
    "Degree-preserving rewiring": fp.GREY,
    "Size-matched random": fp.C_NEURAL,
}
markers = {1: "o", 2: "o", 3: "o"}
y_positions = np.arange(len(order))[::-1]
seed_offsets = {1: -0.075, 2: 0.0, 3: 0.075}
for y, arm in zip(y_positions, order):
    vals = arm_values[arm]
    for seed in seeds:
        ax_b.scatter(vals[seed], y + seed_offsets[seed], s=27,
                     marker=markers[seed], color=colors[arm],
                     edgecolor="white", linewidth=0.55, zorder=4)
    mean = means[arm]
    sd = sds[arm]
    ax_b.errorbar(mean, y, xerr=sd, fmt="D", color=fp.INK,
                  markerfacecolor="white", markeredgecolor=colors[arm],
                  markeredgewidth=1.0, markersize=5.0,
                  ecolor=colors[arm], elinewidth=fp.LW_EMPH,
                  capsize=2.5, capthick=fp.LW_EMPH, zorder=5)
    # Keep labels inside the plotting frame at the rightmost category.
    if mean > 300:
        ax_b.text(mean - sd - 7, y + 0.18, f"{mean:.0f} ± {sd:.0f}",
                  fontsize=fp.FS_NOTE, color=fp.MUTED, va="center", ha="right")
    else:
        ax_b.text(mean + sd + 7, y + 0.18, f"{mean:.0f} ± {sd:.0f}",
                  fontsize=fp.FS_NOTE, color=fp.MUTED, va="center", ha="left")

ax_b.set_yticks(y_positions)
ax_b.set_yticklabels(order)
ax_b.tick_params(axis="y", pad=7)
ax_b.set_xlim(0, 355)
ax_b.set_xticks([0, 50, 100, 150, 200, 250, 300, 350])
ax_b.set_ylim(-0.55, 4.58)
ax_b.set_xlabel(r"$\sum_g R_g^2$ (held-out; summed across 6,998 genes)", labelpad=5)
ax_b.set_title("(b) Seed-level readout across five mask conditions",
               loc="left", fontsize=fp.FS_TITLE, fontweight="bold", pad=6)
ax_b.grid(axis="x", color="#E6E8EB", lw=0.45, zorder=0)
fp.style_ax(ax_b)

# Compact figure-level heading, pair-type legend, and design note.
fig.text(0.135, 0.958,
         "VEGA masked-decoder positive control  ·  PBMC–IFN data  ·  150 epochs  ·  3 trained seeds",
         ha="left", va="center", fontsize=fp.FS_AXIS, color=fp.INK)
pair_handles = [
    Line2D([], [], linestyle="none", marker="o", markerfacecolor=fp.PURPLE,
           markeredgecolor="white", markersize=4.8, label="same-seed pair (n=3)"),
    Line2D([], [], linestyle="none", marker="^", markerfacecolor=fp.ARM_PRIOR_ALT,
           markeredgecolor=fp.PURPLE, markersize=5.0, label="cross-seed pair (n=6)"),
    Line2D([], [], linestyle="none", marker="s", markerfacecolor=fp.GRAY,
           markeredgecolor="white", markersize=4.5, label="vanilla–vanilla (n=6)"),
]
fig.legend(handles=pair_handles, loc="upper right", bbox_to_anchor=(0.95, 0.925),
           ncol=3, frameon=False, fontsize=fp.FS_NOTE, handletextpad=0.35,
           columnspacing=1.0, borderaxespad=0.0)
fig.text(0.135, 0.080,
         "(a) Nine pairwise ratios reuse three trained seeds; they are not nine independent fits.",
         ha="left", va="center", fontsize=fp.FS_NOTE, color=fp.MUTED)
fig.text(0.135, 0.045,
         "(b) Three seed-level points per arm; 30 rewires are averaged within each seed. Diamonds show mean ± SD.",
         ha="left", va="center", fontsize=fp.FS_NOTE, color=fp.MUTED)

fp.assert_design_width(fig)
for ext in ["pdf", "svg"]:
    fig.savefig(OUT / f"Fig5_VEGA_masked_decoder.{ext}")
fig.savefig(OUT / "Fig5_VEGA_masked_decoder_300dpi.png", dpi=300)
fig.savefig(OUT / "Fig5_VEGA_masked_decoder_600dpi.tif", dpi=600,
            pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)

# Machine-readable audit summary for the README and review.
summary_out = {
    "panel_a": {
        "vega_vanilla_pairings_n": len(cross),
        "vega_vanilla_median": med_cross,
        "vega_vanilla_range": [min(r for _, _, r in cross), max(r for _, _, r in cross)],
        "same_seed_n": len(same),
        "cross_seed_n": len(diff),
        "vanilla_vanilla_ordered_pairs_n": len(null),
        "vanilla_vanilla_median": med_null,
        "vanilla_vanilla_range": [min(r for _, _, r in null), max(r for _, _, r in null)],
        "cross_family_band": [0.94, 1.06],
    },
    "panel_b": {
        arm: {"seed_values": {str(s): arm_values[arm][s] for s in seeds},
              "mean": means[arm], "between_seed_sd": sds[arm],
              "n_seeds": 3,
              "nested_rewirings_per_seed": 30 if arm == "Degree-preserving rewiring" else 0}
        for arm in order[::-1]
    },
    "metadata": {"epochs": 150, "cells": 16893, "genes": 6998, "pathways": 675},
    "excluded_from_main_figure": {
        "uniform_random_folder": "Exploratory uniform-random variants; not the canonical size-matched random_rows arm cited in the V37 Figure 5 caption."
    }
}
(OUT / "Fig5_data_audit.json").write_text(json.dumps(summary_out, indent=2))
print(json.dumps(summary_out, indent=2))
print(f"Wrote outputs to {OUT}")
