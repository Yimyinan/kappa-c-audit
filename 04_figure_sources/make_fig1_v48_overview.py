# -*- coding: utf-8 -*-
"""Build the V48 graphical overview from measured latents and architecture results.

Panel (a) is a three-dimensional PCA projection of the real 64-dimensional ZINB
latents. Point clouds and cellwise links come from seed 42. The projection is
illustrative and is not used as evidence for any claim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from scipy.linalg import orthogonal_procrustes


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figure_palette as fp  # noqa: E402


PKG = HERE.parents[0]
FIG = PKG / "01_manuscript" / "figures"
SVG = PKG / "01_manuscript" / "figures_svg"
TIF = PKG / "01_manuscript" / "figures_tif"
ZINB = HERE / "crossarch_delivery_20261121_zinb"
GAUSS = HERE / "crossarch_delivery_20261120"
DOSE = HERE / "fig6_8_10_prior_delivery" / "dose_ladder.json"
PGEFFECT = (
    HERE
    / "fig6_8_10_prior_delivery"
    / "dose20_e45__pergene_vs_baseline.json"
)


def load_latents() -> tuple[dict[str, np.ndarray], np.ndarray]:
    base = ZINB / "runs" / "seed_42"
    latents = {
        arm: np.load(base / arm / "latent.npz")["Z"].astype("float64")
        for arm in ("empty", "random", "real")
    }
    reference = latents["empty"] - latents["empty"].mean(axis=0, keepdims=True)
    aligned = {"empty": reference}
    for arm in ("random", "real"):
        centered = latents[arm] - latents[arm].mean(axis=0, keepdims=True)
        rotation, _ = orthogonal_procrustes(centered, reference)
        aligned[arm] = centered @ rotation
    stacked = np.concatenate([aligned["empty"], aligned["random"], aligned["real"]], axis=0)
    mean = stacked.mean(axis=0)
    _, singular, vh = np.linalg.svd(stacked - mean, full_matrices=False)
    scores = {
        arm: (value - mean) @ vh[:3].T
        for arm, value in aligned.items()
    }
    explained = singular[:3] ** 2 / np.sum(singular**2)
    return scores, explained


def draw_ellipsoid(ax, points: np.ndarray, color: str) -> None:
    center = points.mean(axis=0)
    covariance = np.cov(points, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    u = np.linspace(0, 2 * np.pi, 26)
    v = np.linspace(0, np.pi, 16)
    sphere = np.stack(
        [
            np.outer(np.cos(u), np.sin(v)),
            np.outer(np.sin(u), np.sin(v)),
            np.outer(np.ones_like(u), np.cos(v)),
        ],
        axis=-1,
    )
    transform = vectors @ np.diag(np.sqrt(values) * 1.45)
    surface = sphere @ transform.T + center
    ax.plot_surface(
        surface[:, :, 0],
        surface[:, :, 1],
        surface[:, :, 2],
        color=color,
        alpha=0.075,
        linewidth=0,
        shade=False,
    )


def add_panel_tag(fig, x: float, y: float, tag: str) -> None:
    fig.text(x, y, f"({tag})", fontsize=fp.FS_PANEL, fontweight="bold", va="top")


def forest_panel(ax) -> None:
    gaussian = json.loads((GAUSS / "crossarch_results.json").read_text(encoding="utf-8"))
    zinb = json.loads((ZINB / "crossarch_results.json").read_text(encoding="utf-8"))
    pg = json.loads(PGEFFECT.read_text(encoding="utf-8"))
    rows = [
        (
            "PG-UNAGI",
            np.array(
                [
                    pg["per_seed"][seed]["real_vs_shuffled"]
                    for seed in ("42", "43", "44")
                ]
            ),
            fp.C_HEMA,
            "3/3 seeds",
        ),
        (
            "Gaussian VAE",
            np.array([p["real_vs_random_on_real_targets"] for p in gaussian["paired"]]),
            fp.C_NEURAL,
            "3/3 seeds",
        ),
        (
            "ZINB VAE",
            np.array([p["real_vs_random_on_real_targets"] for p in zinb["paired"]]),
            fp.ARM_PRIOR,
            "3/3 seeds",
        ),
    ]
    for y, (label, values, color, note) in enumerate(rows):
        if len(values) > 1:
            ax.plot(
                [values.mean() - values.std(ddof=1), values.mean() + values.std(ddof=1)],
                [y, y],
                color=color,
                lw=fp.LW_DATA,
                solid_capstyle="round",
            )
            ax.scatter(
                values,
                np.full(values.shape, y),
                s=10,
                color=color,
                alpha=0.48,
                edgecolors="none",
                zorder=3,
            )
        ax.scatter(
            [values.mean()],
            [y],
            s=52,
            marker="D",
            color=color,
            edgecolors="white",
            linewidths=0.6,
            zorder=4,
        )
        ax.text(
            values.mean(),
            y + 0.21,
            f"{values.mean():+.4f}",
            ha="center",
            va="bottom",
            fontsize=fp.FS_NOTE,
            color=fp.INK,
        )
        ax.text(
            values.mean(),
            y - 0.23,
            note,
            ha="center",
            va="top",
            fontsize=fp.FS_NOTE,
            color=fp.MUTED,
        )
    ax.axvline(0, color=fp.EDGE, lw=fp.LW_REF, zorder=0)
    ax.set_yticks(np.arange(len(rows)), [row[0] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(-0.002, 0.044)
    ax.set_xlabel(r"target-gene $\Delta R^2$: real minus matched random")
    ax.set_title("Content-specific target-gene effect", loc="left", pad=5)
    fp.style_ax(ax)


def ratio_panel(ax) -> None:
    gaussian = json.loads((GAUSS / "crossarch_results.json").read_text(encoding="utf-8"))["stats"]
    zinb = json.loads((ZINB / "crossarch_results.json").read_text(encoding="utf-8"))["stats"]
    dose = json.loads(DOSE.read_text(encoding="utf-8"))
    pg_values = np.concatenate(
        [np.asarray(dose["means"][level]["kcw"], dtype=float) for level in ("5.0", "20.0", "40.0")]
    )
    rows = [
        ("PG-UNAGI", pg_values, fp.C_HEMA),
        (
            "Gaussian VAE",
            np.asarray([gaussian["kappa_C_W_real"]["mean"], gaussian["kappa_C_W_random"]["mean"]]),
            fp.C_NEURAL,
        ),
        (
            "ZINB VAE",
            np.asarray([zinb["kappa_C_W_real"]["mean"], zinb["kappa_C_W_random"]["mean"]]),
            fp.ARM_PRIOR,
        ),
    ]
    ax.axhspan(0.90, 1.10, color=fp.BAND_CROSS, zorder=0)
    ax.axhline(1.0, color=fp.EDGE, lw=fp.LW_REF, zorder=1)
    ax.axhline(0.5, color=fp.FAIL, lw=fp.LW_DATA, ls="--", zorder=1)
    for x, (label, values, color) in enumerate(rows):
        if label == "PG-UNAGI":
            ax.vlines(x, values.min(), values.max(), color=color, lw=fp.LW_DATA)
            ax.scatter(np.full(values.shape, x), values, s=9, color=color, alpha=0.55)
            ax.scatter(x, np.median(values), s=42, marker="D", color=color,
                       edgecolors="white", linewidths=0.5, zorder=3)
        else:
            ax.vlines(x, values.min(), values.max(), color=color, lw=fp.LW_DATA)
            ax.scatter(x, values.mean(), s=42, marker="D", color=color,
                       edgecolors="white", linewidths=0.5, zorder=3)
    ax.text(2.10, 0.505, "deep compression", ha="right", va="bottom",
            fontsize=fp.FS_NOTE, color=fp.FAIL)
    ax.text(2.10, 1.012, "vanilla level", ha="right", va="bottom",
            fontsize=fp.FS_NOTE, color=fp.MUTED)
    ax.set_xticks(np.arange(len(rows)), [row[0] for row in rows])
    ax.set_ylim(0.40, 1.09)
    ax.set_ylabel(r"decoder ratio $\kappa$")
    ax.set_title("Decoder remains near unity", loc="left", pad=5)
    fp.style_ax(ax)


def summary_panel(ax) -> None:
    ax.set_axis_off()
    cards = [
        ("0 / 33", "no deep decoder\ncompression in the\nPG-UNAGI audit", fp.PASS),
        ("0", "functional readers of the\nper-gene prior in the\noriginal path", fp.FAIL),
        ("3 / 3", "positive target-gene\nseed tests in each\nadditional VAE", fp.ARM_PRIOR),
    ]
    for i, (value, label, color) in enumerate(cards):
        x = 0.012 + i * 0.327
        box = FancyBboxPatch(
            (x, 0.08),
            0.31,
            0.84,
            transform=ax.transAxes,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor="white",
            edgecolor=color,
            linewidth=fp.LW_DATA,
        )
        ax.add_patch(box)
        ax.text(
            x + 0.025,
            0.68,
            value,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color=color,
        )
        ax.text(
            x + 0.025,
            0.34,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.8,
            color=fp.INK,
            linespacing=1.3,
        )


def main() -> None:
    fp.apply_sample_style()
    FIG.mkdir(parents=True, exist_ok=True)
    SVG.mkdir(parents=True, exist_ok=True)
    TIF.mkdir(parents=True, exist_ok=True)

    scores, explained = load_latents()
    width, height = fp.figsize(height_mm=104)
    fig = plt.figure(figsize=(width, height))
    ax3 = fig.add_axes([0.010, 0.075, 0.565, 0.845], projection="3d")
    forest_ax = fig.add_axes([0.620, 0.585, 0.310, 0.285])
    ratio_ax = fig.add_axes([0.620, 0.175, 0.310, 0.255])

    colors = {"empty": fp.ARM_VANILLA, "random": fp.C_NEURAL, "real": fp.ARM_PRIOR}
    labels = {"empty": "no prior", "random": "matched random", "real": "real prior"}
    rng = np.random.default_rng(45)
    draw = rng.choice(scores["empty"].shape[0], size=950, replace=False)
    link = rng.choice(scores["empty"].shape[0], size=34, replace=False)

    for arm in ("empty", "random", "real"):
        draw_ellipsoid(ax3, scores[arm], colors[arm])
        ax3.scatter(
            scores[arm][draw, 0],
            scores[arm][draw, 1],
            scores[arm][draw, 2],
            s=2.2,
            color=colors[arm],
            alpha=0.24,
            depthshade=False,
            rasterized=False,
        )
    for arm in ("random", "real"):
        for idx in link:
            xs = [scores["empty"][idx, 0], scores[arm][idx, 0]]
            ys = [scores["empty"][idx, 1], scores[arm][idx, 1]]
            zs = [scores["empty"][idx, 2], scores[arm][idx, 2]]
            ax3.plot(xs, ys, zs, color=colors[arm], alpha=0.18, lw=0.45)

    # Tighten the data box around the central 98% of points. This reduces the
    # empty 3D volume and keeps the projected axis labels close to the cube.
    all_scores = np.concatenate([scores[arm] for arm in ("empty", "random", "real")], axis=0)
    axes = (ax3.set_xlim, ax3.set_ylim, ax3.set_zlim)
    for dimension, setter in enumerate(axes):
        low, high = np.percentile(all_scores[:, dimension], [1, 99])
        pad = max(0.08, 0.05 * (high - low))
        setter(low - pad, high + pad)
    ax3.set_box_aspect((1.0, 1.0, 0.68))
    ax3.set_proj_type("ortho")
    ax3.view_init(elev=22, azim=-58)
    ax3.set_xlabel("PC1", labelpad=1.5)
    ax3.set_ylabel("PC2", labelpad=3.0)
    # mplot3d projects a z-axis label toward the neighbouring panel after the
    # coordinate box is compressed. Keep PC3 in the stable explanatory note
    # rather than drawing a floating label over panel a or panel b.
    ax3.set_zlabel("")
    ax3.zaxis.set_label_text("")
    # Text2D handles the semantic axis labels; remove the projected mplot3d
    # z-label artist so it cannot leave a stray glyph over the explanatory note.
    if ax3.zaxis.label in ax3.texts:
        ax3.texts.remove(ax3.zaxis.label)
    ax3.zaxis.set_visible(False)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_zticks([])
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor(fp.EDGE)
        axis.pane.set_alpha(0.22)
    ax3.grid(True, color=fp.GREY, alpha=0.18, linewidth=0.45)
    ax3.set_title("Cellwise latent movement in a ZINB VAE", loc="left", pad=4)
    handles = [
        Line2D([], [], marker="o", ls="", color=colors[arm], markersize=4,
               label=labels[arm])
        for arm in ("empty", "random", "real")
    ]
    ax3.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.98),
               fontsize=fp.FS_NOTE, frameon=False, handlelength=0.8)
    ax3.text2D(
        0.02,
        0.02,
        "Seed 42; 950 cells per arm; 34 linked cells.\n"
        f"Explained variance: PC1 {explained[0]*100:.1f}%, "
        f"PC2 {explained[1]*100:.1f}%, PC3 {explained[2]*100:.1f}%.\n"
        "PCA view, illustrative only.",
        transform=ax3.transAxes,
        fontsize=fp.FS_NOTE,
        color=fp.MUTED,
        ha="left",
        va="bottom",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "pad": 0.3,
            "alpha": 1.0,
        },
        zorder=20,
    )

    forest_panel(forest_ax)
    ratio_panel(ratio_ax)
    add_panel_tag(fig, 0.012, 0.985, "a")
    add_panel_tag(fig, 0.600, 0.985, "b")
    add_panel_tag(fig, 0.600, 0.535, "c")

    fp.assert_design_width(fig)
    pdf = FIG / "Fig1_dual_spectrum_concept.pdf"
    png = FIG / "Fig1_dual_spectrum_concept.png"
    svg = SVG / "Fig1_dual_spectrum_concept.svg"
    tif = TIF / "Fig1.tif"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    fig.savefig(tif, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    with Image.open(tif) as image:
        image.convert("RGB").save(tif, dpi=(600, 600), compression="tiff_lzw")
    print(pdf)
    print(png)
    print(tif)


if __name__ == "__main__":
    main()
