#!/usr/bin/env python3
"""Build Supporting Figure S16 and Table S15 from the cross-architecture run."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parents[1]
sys.path.insert(0, str(PAPER / "paper1_SUBMISSION_PACKAGE_20261120_V42" / "04_figure_sources"))
import figure_palette as fp  # noqa: E402


def main() -> None:
    data = json.loads((ROOT / "crossarch_results.json").read_text(encoding="utf-8"))
    paired = data["paired"]
    seeds = [int(row["seed"]) for row in paired]
    output = ROOT / "figure"
    output.mkdir(parents=True, exist_ok=True)

    fp.apply_sample_style()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=fp.figsize(height_mm=74),
        gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]},
    )
    jitter = np.array([-0.08, 0.0, 0.08])

    # (A) 2 x 2 content design: each prior on its named and control genes.
    ax = axes[0]
    categories = ["Real prior", "Random prior"]
    on_target = np.array(
        [row["real_prior_on_real_targets"] for row in paired], dtype=float
    )
    random_on_target = np.array(
        [row["random_prior_on_random_targets"] for row in paired], dtype=float
    )
    off_target_real = np.array(
        [row["real_prior_on_random_targets"] for row in paired], dtype=float
    )
    off_target_random = np.array(
        [row["random_prior_on_real_targets"] for row in paired], dtype=float
    )
    means = [
        [float(on_target.mean()), float(off_target_real.mean())],
        [float(random_on_target.mean()), float(off_target_random.mean())],
    ]
    values = [
        [on_target, off_target_real],
        [random_on_target, off_target_random],
    ]
    width = 0.32
    for group, x in enumerate([0.0, 1.0]):
        for side, color in enumerate([fp.ARM_PRIOR, fp.GREY]):
            xs = x + (side - 0.5) * width
            ax.bar(
                xs,
                means[group][side],
                width=width * 0.86,
                color=color,
                edgecolor="white",
                linewidth=fp.LW_EDGE,
                zorder=2,
            )
            ax.errorbar(
                xs,
                means[group][side],
                yerr=values[group][side].std(ddof=1),
                fmt="none",
                color=fp.INK,
                elinewidth=fp.LW_ERR,
                capsize=fp.CAPSIZE,
                zorder=3,
            )
            ax.scatter(
                xs + jitter,
                values[group][side],
                s=9,
                facecolor="white",
                edgecolor=fp.EDGE,
                linewidth=fp.LW_EDGE,
                zorder=4,
            )
    ax.axhline(0, color=fp.INK, lw=fp.LW_REF, zorder=1)
    ax.set_xticks([0.0, 1.0], categories)
    ax.set_ylabel(r"$\Delta R^2$ vs no-prior")
    ax.set_title("(a)  Named versus control genes", loc="left", pad=7)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=fp.ARM_PRIOR, label="Named genes"),
            plt.Rectangle((0, 0), 1, 1, color=fp.GREY, label="Other prior's genes"),
        ],
        loc="upper left",
        fontsize=fp.FS_NOTE,
        frameon=False,
        handlelength=0.8,
    )

    # (B) Paired same-gene real-versus-random effect.
    ax = axes[1]
    direct = np.array(
        [row["real_vs_random_on_real_targets"] for row in paired], dtype=float
    )
    reverse = -np.array(
        [row["real_vs_random_on_random_targets"] for row in paired], dtype=float
    )
    for x, values_, color in [(0.0, direct, fp.ARM_PRIOR), (1.0, reverse, fp.ARM_PRIOR)]:
        ax.bar(
            x,
            float(values_.mean()),
            width=0.55,
            color=color,
            edgecolor="white",
            linewidth=fp.LW_EDGE,
            zorder=2,
        )
        ax.errorbar(
            x,
            float(values_.mean()),
            yerr=float(values_.std(ddof=1)),
            fmt="none",
            color=fp.INK,
            elinewidth=fp.LW_ERR,
            capsize=fp.CAPSIZE,
            zorder=3,
        )
        ax.scatter(
            x + jitter,
            values_,
            s=9,
            facecolor="white",
            edgecolor=fp.EDGE,
            linewidth=fp.LW_EDGE,
            zorder=4,
        )
    ax.axhline(0, color=fp.INK, lw=fp.LW_REF)
    ax.set_xticks(
        [0.0, 1.0],
        ["On real\ntargets", "On random\ntargets"],
    )
    ax.set_ylabel(r"Same-gene $\Delta R^2$")
    ax.set_title("(b)  Real prior minus random", loc="left", pad=7)

    # (C) Aggregate decoder ratio remains at unity.
    ax = axes[2]
    real = np.array([row["kappa_C_W_real"] for row in paired], dtype=float)
    random = np.array([row["kappa_C_W_random"] for row in paired], dtype=float)
    ax.axhspan(0.90, 1.10, color=fp.BAND_NULL, zorder=0)
    ax.axhline(1.0, color=fp.INK, lw=fp.LW_REF, zorder=1)
    for x, values_, color, label in [
        (0.0, real, fp.ARM_PRIOR, "Real"),
        (1.0, random, fp.GREY, "Random"),
    ]:
        ax.bar(
            x,
            float(values_.mean()),
            width=0.55,
            color=color,
            edgecolor="white",
            linewidth=fp.LW_EDGE,
            zorder=2,
        )
        ax.errorbar(
            x,
            float(values_.mean()),
            yerr=float(values_.std(ddof=1)),
            fmt="none",
            color=fp.INK,
            elinewidth=fp.LW_ERR,
            capsize=fp.CAPSIZE,
            zorder=3,
        )
        ax.scatter(
            x + jitter,
            values_,
            s=9,
            facecolor="white",
            edgecolor=fp.EDGE,
            linewidth=fp.LW_EDGE,
            zorder=4,
        )
    ax.set_xticks([0.0, 1.0], ["Real", "Random"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0.94, 1.07)
    ax.set_ylabel(r"Aggregate $\kappa$-$C(W)$")
    ax.set_title("(c)  Aggregate readout", loc="left", pad=7)
    ax.text(
        1.52,
        0.945,
        "deep-compression threshold\n"
        r"$\kappa<0.5$ is off scale",
        ha="right",
        va="bottom",
        fontsize=fp.FS_NOTE,
        color=fp.MUTED,
    )

    for ax in axes:
        fp.style_ax(ax)
        ax.tick_params(labelsize=fp.FS_AXIS)

    fig.text(
        0.5,
        0.01,
        "Plain MLP VAE; 12,000 cells; boost 10; n = 3 paired seeds; "
        "202 true and 202 random target genes.",
        ha="center",
        va="bottom",
        fontsize=fp.FS_NOTE,
        color=fp.MUTED,
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.22, top=0.88, wspace=0.42)
    fp.assert_design_width(fig)
    fp.export(fig, str(output / "FigS16_crossarch_delivered_prior"), deck=False)
    plt.close(fig)

    # Compact source table.
    table = output / "TableS15_crossarch_delivery.csv"
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "seed",
                "real_on_real",
                "real_on_random",
                "random_on_real",
                "random_on_random",
                "real_minus_random_real",
                "interaction",
                "kappa_real",
                "kappa_random",
                "real_gt_random_on_real_genes",
                "random_gt_real_on_random_genes",
            ]
        )
        for row in paired:
            writer.writerow(
                [
                    row["seed"],
                    row["real_prior_on_real_targets"],
                    row["real_prior_on_random_targets"],
                    row["random_prior_on_real_targets"],
                    row["random_prior_on_random_targets"],
                    row["real_vs_random_on_real_targets"],
                    row["interaction"],
                    row["kappa_C_W_real"],
                    row["kappa_C_W_random"],
                    row["real_gt_random_on_real_genes"],
                    row["random_gt_real_on_random_genes"],
                ]
            )


if __name__ == "__main__":
    main()
