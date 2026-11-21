#!/usr/bin/env python3
"""Build Supporting Figure S17 and Table S16 for the ZINB VAE."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parents[1]
sys.path.insert(0, str(PAPER / "paper1_SUBMISSION_PACKAGE_20261120_V43" / "04_figure_sources"))
import figure_palette as fp  # noqa: E402


def main() -> None:
    data = json.loads((ROOT / "crossarch_results.json").read_text(encoding="utf-8"))
    paired = data["paired"]
    out = ROOT / "figure"
    out.mkdir(parents=True, exist_ok=True)
    fp.apply_sample_style()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=fp.figsize(height_mm=74),
        gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]},
    )
    jitter = np.array([-0.08, 0.0, 0.08])

    ax = axes[0]
    real = np.array([r["real_prior_on_real_targets"] for r in paired], float)
    real_off = np.array([r["real_prior_on_random_targets"] for r in paired], float)
    random = np.array([r["random_prior_on_random_targets"] for r in paired], float)
    random_off = np.array([r["random_prior_on_real_targets"] for r in paired], float)
    width = 0.32
    for group, (on_values, off_values) in enumerate(
        [(real, real_off), (random, random_off)]
    ):
        for side, (values, color) in enumerate(
            [(on_values, fp.ARM_PRIOR), (off_values, fp.GREY)]
        ):
            x = group + (side - 0.5) * width
            mean = float(values.mean())
            ax.bar(
                x,
                mean,
                width=width * 0.86,
                color=color,
                edgecolor="white",
                linewidth=fp.LW_EDGE,
                zorder=2,
            )
            ax.errorbar(
                x,
                mean,
                yerr=float(values.std(ddof=1)),
                fmt="none",
                color=fp.INK,
                elinewidth=fp.LW_ERR,
                capsize=fp.CAPSIZE,
                zorder=3,
            )
            ax.scatter(
                x + jitter,
                values,
                s=9,
                facecolor="white",
                edgecolor=fp.EDGE,
                linewidth=fp.LW_EDGE,
                zorder=4,
            )
    ax.axhline(0, color=fp.INK, lw=fp.LW_REF)
    ax.set_xticks([0, 1], ["Real prior", "Random prior"])
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

    ax = axes[1]
    direct = np.array([r["real_vs_random_on_real_targets"] for r in paired], float)
    direct_cv = np.array(
        [r["cv_real_vs_random_on_real_targets"] for r in paired], float
    )
    reverse = -np.array(
        [r["real_vs_random_on_random_targets"] for r in paired], float
    )
    for x, values in [(0.0, direct), (1.0, reverse)]:
        mean = float(values.mean())
        ax.bar(
            x,
            mean,
            width=0.55,
            color=fp.ARM_PRIOR,
            edgecolor="white",
            linewidth=fp.LW_EDGE,
            zorder=2,
        )
        ax.errorbar(
            x,
            mean,
            yerr=float(values.std(ddof=1)),
            fmt="none",
            color=fp.INK,
            elinewidth=fp.LW_ERR,
            capsize=fp.CAPSIZE,
            zorder=3,
        )
        ax.scatter(
            x + jitter,
            values,
            s=9,
            facecolor="white",
            edgecolor=fp.EDGE,
            linewidth=fp.LW_EDGE,
            zorder=4,
        )
    ax.scatter(
        np.array([-0.08, 0.0, 0.08]),
        direct_cv,
        s=12,
        facecolor="none",
        edgecolor=fp.INK,
        marker="D",
        linewidth=fp.LW_EDGE,
        zorder=5,
        label="5-fold readout",
    )
    ax.axhline(0, color=fp.INK, lw=fp.LW_REF)
    ax.set_xticks([0, 1], ["On real\ntargets", "On random\ntargets"])
    ax.set_ylabel(r"Same-gene $\Delta R^2$")
    ax.set_title("(b)  Real minus random", loc="left", pad=7)
    ax.legend(loc="upper right", fontsize=fp.FS_NOTE, frameon=False)

    ax = axes[2]
    kappa_real = np.array([r["kappa_C_W_real"] for r in paired], float)
    kappa_random = np.array([r["kappa_C_W_random"] for r in paired], float)
    ax.axhspan(0.90, 1.10, color=fp.BAND_NULL, zorder=0)
    ax.axhline(1.0, color=fp.INK, lw=fp.LW_REF, zorder=1)
    for x, values, color in [
        (0.0, kappa_real, fp.ARM_PRIOR),
        (1.0, kappa_random, fp.GREY),
    ]:
        mean = float(values.mean())
        ax.bar(
            x,
            mean,
            width=0.55,
            color=color,
            edgecolor="white",
            linewidth=fp.LW_EDGE,
            zorder=2,
        )
        ax.errorbar(
            x,
            mean,
            yerr=float(values.std(ddof=1)),
            fmt="none",
            color=fp.INK,
            elinewidth=fp.LW_ERR,
            capsize=fp.CAPSIZE,
            zorder=3,
        )
        ax.scatter(
            x + jitter,
            values,
            s=9,
            facecolor="white",
            edgecolor=fp.EDGE,
            linewidth=fp.LW_EDGE,
            zorder=4,
        )
    ax.set_xticks([0, 1], ["Real", "Random"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0.94, 1.02)
    ax.set_ylabel(r"Aggregate $\kappa$-$C(W)$")
    ax.set_title("(c)  Aggregate readout", loc="left", pad=7)
    ax.text(
        1.52,
        0.944,
        "deep threshold\nκ<0.5 is off scale",
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
        "ZINB VAE; reconstructed count likelihood; 12,000 cells; "
        "boost 10; n = 3 paired seeds.",
        ha="center",
        va="bottom",
        fontsize=fp.FS_NOTE,
        color=fp.MUTED,
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.22, top=0.88, wspace=0.42)
    fp.assert_design_width(fig)
    fp.export(fig, str(out / "FigS17_zinb_prior_delivery"), deck=False)
    plt.close(fig)

    with (out / "TableS16_zinb_delivery.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
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
                "cv_real_minus_random_real",
                "cv_interaction",
                "kappa_real",
                "kappa_random",
                "real_gt_random_real_genes",
                "random_gt_real_random_genes",
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
                    row["cv_real_vs_random_on_real_targets"],
                    row["cv_interaction"],
                    row["kappa_C_W_real"],
                    row["kappa_C_W_random"],
                    row["real_gt_random_on_real_genes"],
                    row["random_gt_real_on_random_genes"],
                ]
            )


if __name__ == "__main__":
    main()
