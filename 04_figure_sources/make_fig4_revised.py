#!/usr/bin/env python3
"""Revised, reproducible Figure 4 for the single-cell VAE manuscript.

Main figure (a-d):
  (a-c) pooled latent covariance participation-ratio trajectories by tissue.
  (d) run/configuration-level log change from iteration 0 to iteration 4.

Supplementary candidate (e-f):
  (e) descriptive hematopoietic specificity comparison (correctly labels the
      empty-TF vehicle arm; does not imply a monotonic specificity trend).
  (f) descriptive endpoint values for the neural 2^3 design. No inferential
      p-values or confidence intervals are shown because six of eight cells
      have one configuration only.

Inputs expected alongside this script:
  z_spectrum_all.csv
  figure_palette.py

Run:
  python make_fig4_revised.py --csv z_spectrum_all.csv --outdir figures
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import figure_palette as fp


def _apply_fig_style() -> None:
    """Use the shared paper palette with an Arial-metric-compatible font fallback."""
    fp.apply_sample_style()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Arimo", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arimo",
        "mathtext.it": "Arimo:italic",
        "mathtext.bf": "Arimo:bold",
    })


def _save_all(fig: plt.Figure, outdir: Path, stem: str) -> dict[str, Path]:
    """Save vector and raster copies at fixed, full-width design size."""
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pdf": outdir / f"{stem}.pdf",
        "svg": outdir / f"{stem}.svg",
        "png": outdir / f"{stem}_300dpi.png",
        "tif": outdir / f"{stem}_600dpi.tif",
    }
    fig.savefig(paths["pdf"], facecolor="white")
    fig.savefig(paths["svg"], facecolor="white")
    fig.savefig(paths["png"], dpi=300, facecolor="white")
    fig.savefig(paths["tif"], dpi=600, pil_kwargs={"compression": "tiff_lzw"}, facecolor="white")
    plt.close(fig)
    return paths


def _style_ax(ax: plt.Axes) -> None:
    fp.style_ax(ax)
    ax.tick_params(labelsize=7.2)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.10, f"({label})", transform=ax.transAxes,
            fontsize=9, fontweight="bold", ha="left", va="bottom")


def _complete_run_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run, tissue, role), g in df.groupby(["run", "tissue", "role"], dropna=False):
        g = g.sort_values("iteration")
        if g["iteration"].nunique() < 5 or g["iteration"].min() != 0 or g["iteration"].max() != 4:
            continue
        z0 = g.loc[g.iteration.eq(0), "pr"]
        z4 = g.loc[g.iteration.eq(4), "pr"]
        if len(z0) != 1 or len(z4) != 1 or z0.iloc[0] <= 0 or z4.iloc[0] <= 0:
            continue
        rows.append({
            "run": run, "tissue": tissue, "role": role,
            "dlog": float(np.log(z4.iloc[0] / z0.iloc[0])),
        })
    return pd.DataFrame(rows)


def _run_trajectory_panel(ax: plt.Axes, sub: pd.DataFrame, tissue: str,
                          show_ylabel: bool) -> None:
    tissue_col = fp.tissue_color(tissue)
    # Draw prior/ablation first, then controls, then vanilla for visual priority.
    prior = sub[sub.role.isin(["prior", "ablation"])]
    controls = sub[sub.role.isin(["vehicle", "wrongprior"])]
    vanilla = sub[sub.role.eq("vanilla")]

    for _, g in prior.groupby("run", sort=True):
        g = g.sort_values("iteration")
        ax.plot(g.iteration, g.pr, color=tissue_col, lw=0.62, alpha=0.46,
                marker="o", markersize=1.8, markeredgewidth=0, zorder=2)
    for _, g in controls.groupby("run", sort=True):
        g = g.sort_values("iteration")
        ax.plot(g.iteration, g.pr, color=fp.GREY, lw=0.72, alpha=0.9,
                linestyle=(0, (2.2, 1.6)), marker="D", markersize=1.8,
                markeredgewidth=0, zorder=3)
    for _, g in vanilla.groupby("run", sort=True):
        g = g.sort_values("iteration")
        ax.plot(g.iteration, g.pr, color=fp.VANILLA, lw=1.05, alpha=0.95,
                marker="s", markersize=2.4, markeredgewidth=0.25,
                markeredgecolor="white", zorder=5)

    ax.set_title(fp.tissue_label(tissue), fontsize=8.5, fontweight="bold", pad=5)
    ax.set_xlim(-0.12, 4.12)
    ax.set_xticks(range(5))
    ax.set_ylim(0, 10)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel("IDREM iteration", fontsize=7.7, labelpad=3)
    if show_ylabel:
        ax.set_ylabel("Latent effective rank\n(participation ratio)", fontsize=7.7, labelpad=4)
    else:
        ax.tick_params(labelleft=False)
    _style_ax(ax)


def make_main_figure(df: pd.DataFrame, outdir: Path) -> tuple[dict, dict]:
    tissue_order = ["hemo", "neural", "lung"]
    tissue_counts = df.groupby("tissue")["run"].nunique().to_dict()
    complete = _complete_run_table(df)
    complete_counts = complete.groupby("tissue").size().to_dict()

    _apply_fig_style()
    fig = plt.figure(figsize=fp.figsize(height_mm=121))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.92],
                          left=0.17, right=0.985, top=0.885, bottom=0.135,
                          wspace=0.34, hspace=0.56)

    # A-C: small-multiple tissue trajectories, common y scale.
    for i, tissue in enumerate(tissue_order):
        ax = fig.add_subplot(gs[0, i])
        sub = df[df.tissue.eq(tissue)]
        _run_trajectory_panel(ax, sub, tissue, show_ylabel=(i == 0))
        _panel_label(ax, "abc"[i])

    top_handles = [
        Line2D([], [], color=fp.VANILLA, lw=1.05, marker="s", markersize=3,
               label="Vanilla control"),
        Line2D([], [], color=fp.C_HEMA, lw=0.8, marker="o", markersize=2.5,
               label="Prior / ablation (tissue-coloured)"),
        Line2D([], [], color=fp.GREY, lw=0.8, linestyle=(0, (2.2, 1.6)),
               marker="D", markersize=2.3, label="Vehicle / wrong-tissue prior"),
    ]
    fig.legend(handles=top_handles, loc="upper center", bbox_to_anchor=(0.54, 0.975),
               ncol=3, frameon=False, fontsize=7.0, handlelength=1.8,
               columnspacing=1.4, handletextpad=0.55)

    # D: all configurations with a complete iteration 0 -> 4 trajectory.
    ax = fig.add_subplot(gs[1, :])
    y_centres = {"hemo": 2.0, "neural": 1.0, "lung": 0.0}
    y_labels = {
        "hemo": f"Hematopoietic ({complete_counts.get('hemo', 0)} configs)",
        "neural": f"Neural cortex ({complete_counts.get('neural', 0)} configs)",
        "lung": f"Lung fibrosis ({complete_counts.get('lung', 0)} configs)",
    }
    for tissue in tissue_order:
        s = complete[complete.tissue.eq(tissue)].sort_values("dlog")
        if s.empty:
            continue
        offsets = np.linspace(-0.23, 0.23, len(s)) if len(s) > 1 else np.array([0.0])
        for (_, r), off in zip(s.iterrows(), offsets):
            y = y_centres[tissue] + off
            if r.role == "vanilla":
                marker, color, size = "s", fp.VANILLA, 31
            elif r.role in ("vehicle", "wrongprior"):
                marker, color, size = "D", fp.GREY, 25
            else:
                marker, color, size = "o", fp.tissue_color(tissue), 25
            ax.scatter(r.dlog, y, marker=marker, s=size, facecolor=color,
                       edgecolor="white", linewidth=0.42, zorder=4)
    ax.axvline(0, color=fp.GRAY, lw=fp.LW_REF, linestyle=(0, (2.5, 2)), zorder=1)
    ax.set_yticks([])
    ax.set_xlim(-1.88, 0.75)
    ax.set_ylim(-0.45, 2.45)
    for tissue in tissue_order:
        label = {
            "hemo": f"Hematopoietic\n(n={complete_counts.get('hemo', 0)})",
            "neural": f"Neural cortex\n(n={complete_counts.get('neural', 0)})",
            "lung": f"Lung fibrosis\n(n={complete_counts.get('lung', 0)})",
        }[tissue]
        ax.text(-1.84, y_centres[tissue], label, ha="left", va="center",
                fontsize=7.0, color=fp.MUTED, clip_on=True)
    ax.set_xlabel(r"Change in log participation ratio, $\log(PR_{iter4}/PR_{iter0})$   (negative = contraction)",
                  fontsize=7.8, labelpad=5)
    ax.set_title("Change from iteration 0 to 4", fontsize=8.5,
                 fontweight="bold", pad=5)
    _panel_label(ax, "d")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.45, zorder=0)
    _style_ax(ax)
    paths = _save_all(fig, outdir, "Fig4_latent_spectrum_main")
    metadata = {
        "available_runs_by_tissue": tissue_counts,
        "complete_runs_by_tissue": complete_counts,
        "unique_runs_with_data": int(df.run.nunique()),
        "complete_runs": int(len(complete)),
        "incomplete_runs": sorted(set(df.run) - set(complete.run)),
        "hemo_prior_complete": int(((complete.tissue == "hemo") & (complete.role == "prior")).sum()),
        "hemo_prior_contracted": int(((complete.tissue == "hemo") & (complete.role == "prior") & (complete.dlog < 0)).sum()),
    }
    return paths, metadata


def make_supplement_figure(df: pd.DataFrame, outdir: Path) -> tuple[dict, dict]:
    _apply_fig_style()
    fig = plt.figure(figsize=fp.figsize(height_mm=82))
    gs = fig.add_gridspec(1, 2, left=0.085, right=0.985, top=0.86, bottom=0.20,
                          wspace=0.48, width_ratios=[1.0, 1.35])

    # E: four seed-42/single-vanilla observations; no connecting line, because
    # the category sequence is not a quantitative continuous axis.
    ax = fig.add_subplot(gs[0, 0])
    endpoint = df[(df.tissue == "hemo") & (df.iteration == 4)].copy()
    ladder = [
        ("R14_hemato_vanilla", "No prior", fp.VANILLA, "s", (5, -2), "left"),
        ("R14_hemato_seed42", "Empty-TF vehicle", fp.GREY, "D", (6, 7), "left"),
        ("R14_wrongprior_seed42", "Wrong-tissue prior", fp.ARM_PRIOR_ALT, "D", (5, 7), "left"),
        ("R13_hemato_trueprior_seed42", "Curated prior", fp.C_HEMA, "o", (-5, 7), "right"),
    ]
    e_rows = []
    for run, label, color, marker, offset, align in ladder:
        s = endpoint[endpoint.run.eq(run)]
        if s.empty:
            continue
        r = s.iloc[0]
        ax.scatter(r.tr, r.pr, marker=marker, s=38, facecolor=color,
                   edgecolor="white", linewidth=0.55, zorder=4)
        ax.annotate(label, (r.tr, r.pr), xytext=offset, textcoords="offset points",
                    fontsize=7.0, color=fp.INK, ha=align, va="center",
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.45,
                                    shrinkA=1, shrinkB=3))
        e_rows.append({"run": run, "label": label, "total_variance": float(r.tr),
                       "participation_ratio": float(r.pr)})
    ax.set_xlabel(r"Total latent variance, $\mathrm{tr}(\mathrm{Cov}(Z))$", fontsize=7.6)
    ax.set_ylabel("Latent effective rank (PR)", fontsize=7.6)
    ax.set_title("Single-seed specificity comparison", fontsize=8.5, fontweight="bold", pad=5)
    _panel_label(ax, "a")
    _style_ax(ax)

    # F: display the eight treatment combinations and the observed endpoint PR.
    # Six cells are singletons; do not add inferential CIs or p-values.
    ax = fig.add_subplot(gs[0, 1])
    neural = df[(df.tissue == "neural") & (df.iteration == 4)].copy()
    neural = neural[neural.run.str.startswith(tuple(f"R{i}_" for i in range(8)))].copy()
    conditions = [(0,0,0), (0,0,1), (0,1,0), (0,1,1),
                  (1,0,0), (1,0,1), (1,1,0), (1,1,1)]
    f_rows = []
    for x, cond in enumerate(conditions):
        s = neural[(neural["C"] == cond[0]) & (neural["M"] == cond[1]) & (neural["T"] == cond[2])]
        vals = s.pr.to_numpy(float)
        offsets = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
        for val, off in zip(vals, offsets):
            ax.scatter(x + off, val, s=31, facecolor=fp.C_NEURAL,
                       edgecolor="white", linewidth=0.5, zorder=4)
        if len(vals) > 1:
            ax.plot([x - 0.17, x + 0.17], [np.mean(vals), np.mean(vals)],
                    color=fp.INK, lw=0.85, zorder=3)
        f_rows.append({"condition_CMT": "".join(map(str, cond)), "n": int(len(vals)),
                       "mean_pr": float(np.mean(vals)) if len(vals) else np.nan,
                       "values": ";".join(f"{v:.6f}" for v in vals)})
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(["".join(map(str, c)) for c in conditions], fontsize=7.1)
    ax.set_xlim(-0.4, 7.4)
    vals_all = neural["pr"].to_numpy(float)
    ax.set_ylim(max(0.0, float(vals_all.min()) - 0.35), float(vals_all.max()) + 0.55)
    ax.set_xlabel("C M T indicator (0 = off; 1 = on)", fontsize=7.6)
    ax.set_ylabel("Latent effective rank (PR), iteration 4", fontsize=7.6)
    ax.set_title("Neural 2³ design: observed endpoints", fontsize=8.5,
                 fontweight="bold", pad=5)
    ax.text(0.99, 0.97, "n = 2 for 000 and 111; n = 1 otherwise\nDescriptive; no inferential tests",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
            color=fp.MUTED)
    _panel_label(ax, "b")
    _style_ax(ax)

    paths = _save_all(fig, outdir, "Fig4_SI_candidate_panels_EF")
    metadata = {"specificity_ladder": e_rows, "neural_factorial_endpoints": f_rows}
    return paths, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path(__file__).resolve().parent / "z_spectrum_all.csv")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent / "fig4_revised_output")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.csv}")
    df = pd.read_csv(args.csv)
    required = {"run", "tissue", "role", "iteration", "pr", "tr", "C", "M", "T"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    if not set(df.tissue.dropna().unique()).issubset({"hemo", "neural", "lung"}):
        raise ValueError("Unexpected tissue labels in CSV")

    main_paths, main_meta = make_main_figure(df, args.outdir)
    supp_paths, supp_meta = make_supplement_figure(df, args.outdir)

    # Export a concise, auditable data summary alongside the figures.
    summary = pd.DataFrame([
        {"figure": "main", "format": k, "path": str(v)} for k, v in main_paths.items()
    ] + [
        {"figure": "supplement_candidate_EF", "format": k, "path": str(v)} for k, v in supp_paths.items()
    ])
    summary.to_csv(args.outdir / "export_manifest.csv", index=False)
    import json
    (args.outdir / "figure4_data_audit.json").write_text(
        json.dumps({"main": main_meta, "supplement_candidate": supp_meta}, indent=2),
        encoding="utf-8")

    print("MAIN FIGURE")
    print(json.dumps(main_meta, indent=2))
    for k, v in main_paths.items():
        print(f"  {k}: {v}")
    print("SUPPLEMENTARY CANDIDATE E-F")
    for k, v in supp_paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
