# -*- coding: utf-8 -*-
"""
Figure: latent-spectrum contraction replicates in hematopoietic data,
is null in neural data, and is untestable in lung (n=1 pair).

(a) hemo   eff_rank trajectories: vanilla vs prior runs
(b) neural eff_rank trajectories: vanilla vs prior/ablation runs
(c) lung   eff_rank trajectories: vanilla vs prior (n=1)
(d) forest: self-normalised change Dlog(eff_rank), iter_0 -> iter_4
(e) hemo specificity ladder, all seed 42, 29,985 cells
(f) neural 2^3 factorial main effects (all null)

NAMING (2026-11-09 audit)
-------------------------
* Tissue names come from figure_palette.tissue_label(): "Lung fibrosis", never
  "Lung IPF"/"hemo"/"Hema.".
* The trajectory legend now names every arm. Previously ``label=`` sat inside the
  per-run loop, so a tissue with two vanilla runs produced two identical
  "vanilla (n=10)" keys while the 18 prior runs had no key at all.
* Panel lettering is the manuscript-wide "(a)" form.

STYLE
-----
Canvas width == placement width (6.69 in), strokes from the palette ladder:
vanilla mean at LW_EMPH (was lw=2.0), individual runs at LW_DATA, references at
LW_REF.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(r"C:/Users/12901/WorkBuddy/2026-09-03-18-47-34")
OUT = ROOT / "outputs" / "positive_result"
CSV = OUT / "z_spectrum_all.csv"
FIGDIR = Path(__file__).resolve().parent.parent / "figures"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_palette as fp  # noqa: E402
from figure_palette import (apply_paper_style, C_HEMA, C_NEURAL, GREY,  # noqa: E402
                            LW_DATA, LW_EDGE, LW_EMPH, LW_REF, MUTED, VANILLA,
                            tissue_color, tissue_label)

apply_paper_style()
TISSUES = ["hemo", "neural", "lung"]


def style_ax(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction="out", length=2.4, width=fp.LW_TICK)


def traj_panel(ax, sub, tag, with_legend=False, with_ylabel=True):
    """eff_rank trajectories; vanilla drawn last, heavier and dark."""
    van = sub[sub.role == "vanilla"]
    pri = sub[sub.role.isin(["prior", "ablation"])]
    veh = sub[sub.role.isin(["vehicle", "wrongprior"])]
    col = tissue_color(sub.tissue.iloc[0])

    for _, g in pri.groupby("run"):
        g = g.sort_values("iteration")
        ax.plot(g.iteration, g.pr, "-", color=col, lw=LW_DATA, alpha=0.55)
    for _, g in veh.groupby("run"):
        g = g.sort_values("iteration")
        ax.plot(g.iteration, g.pr, ":", color=GREY, lw=LW_DATA, alpha=0.9)
    for _, g in van.groupby("run"):
        g = g.sort_values("iteration")
        ax.plot(g.iteration, g.pr, "-", color=VANILLA, lw=LW_EMPH,
                marker="o", ms=3.4, zorder=5)

    ax.set_xlabel("IDREM iteration")
    if with_ylabel:
        ax.set_ylabel("effective spectral rank of Z\n(participation ratio)",
                      fontsize=7.2)
    ax.set_xticks(range(5))
    ax.set_title(f"{fp.panel(tag)} {tissue_label(sub.tissue.iloc[0], cells=True)}",
                 fontweight="bold", fontsize=8.0)

    if with_legend:
        # one handle per arm, built from proxies so the legend does not depend on
        # how many runs a group happens to have (that was the duplicate-key bug).
        # Counts stay out of the keys: n differs per tissue and belongs in the
        # caption, not in a legend that is drawn once for panel (a).
        handles = [Line2D([], [], color=VANILLA, lw=LW_EMPH, marker="o", ms=3.4,
                          label=fp.ARM_LABEL["vanilla"])]
        if len(pri):
            handles.append(Line2D([], [], color=col, lw=LW_DATA, alpha=0.55,
                                  label=fp.ARM_LABEL["prior"]))
        handles.append(Line2D([], [], color=GREY, lw=LW_DATA, ls=":",
                              label="vehicle / wrong prior"))
        ax.legend(handles=handles, loc="lower left", frameon=False,
                  fontsize=6.0, handlelength=1.2, borderpad=0.1,
                  labelspacing=0.26)
    style_ax(ax)


def main() -> None:
    df = pd.read_csv(CSV)

    fig, axes = plt.subplots(2, 3, figsize=(fp.PAPER_WIDTH_IN, 4.35))
    fig.subplots_adjust(left=0.075, right=0.975, top=0.92, bottom=0.105,
                        wspace=0.40, hspace=0.50)

    # ---------------- (a)(b)(c) trajectories ----------------
    for i, (ax, tis) in enumerate(zip(axes[0], TISSUES)):
        sub = df[df.tissue == tis]
        traj_panel(ax, sub, "abc"[i], with_legend=(i == 2), with_ylabel=(i == 0))

    # ---------------- (d) forest: self-normalised change ----------------
    ax = axes[1, 0]
    rows = []
    for run, g in df.groupby("run"):
        g = g.sort_values("iteration")
        if len(g) < 5 or g.iteration.min() != 0:
            continue
        rows.append(dict(run=run, tissue=g.tissue.iloc[0], role=g.role.iloc[0],
                         dlog=float(np.log(g.pr.iloc[-1] / g.pr.iloc[0]))))
    ch = pd.DataFrame(rows)

    ypos, ytick = [], []
    for i, tis in enumerate(TISSUES):
        s = ch[ch.tissue == tis]
        base = i * 1.0
        for j, r in enumerate(s.sort_values("dlog").itertuples()):
            y = base + j * 0.055
            is_van = r.role == "vanilla"
            col = VANILLA if is_van else (
                GREY if r.role in ("vehicle", "wrongprior") else tissue_color(tis))
            ax.plot(r.dlog, y, "o" if not is_van else "s",
                    color=col, ms=2.6 if not is_van else 4.0,
                    markeredgewidth=LW_EDGE, zorder=4 if is_van else 3)
            ypos.append(y)
        vv = s[s.role == "vanilla"].dlog
        if len(vv):
            ax.axvline(vv.mean(), color=VANILLA, ls="--", lw=LW_REF, alpha=0.7)
        ytick.append((base + 0.14, tis))
    ax.axvline(0, color=GREY, lw=LW_REF, zorder=0)
    ax.set_yticks([t[0] for t in ytick])
    ax.set_yticklabels([tissue_label(t[1]) for t in ytick], fontsize=6.2)
    ax.set_ylim(-0.06, 3.05)
    ax.set_xlabel(r"$\Delta\log$(effective rank), iter$_0 \rightarrow$ iter$_4$"
                  "\n(negative = contraction)")
    ax.set_title(f"{fp.panel('d')} Self-normalised change",
                 fontweight="bold", fontsize=8.0)
    style_ax(ax)
    ax.legend(handles=[Line2D([], [], color=VANILLA, marker="s", ls="none",
                              ms=3.6, label="vanilla"),
                       Line2D([], [], color=VANILLA, ls="--", lw=LW_REF,
                              label="vanilla mean")],
              loc="upper left", frameon=False, fontsize=6.0,
              handlelength=1.4, borderpad=0.05, labelspacing=0.25)
    # the three tissue names sit outside the axes; shift this panel right so the
    # longest one ("Hematopoietic") is not clipped by the canvas edge
    _pos = ax.get_position()
    ax.set_position([_pos.x0 + 0.055, _pos.y0, _pos.width - 0.055, _pos.height])

    # ---------------- (e) hemo specificity ladder, seed 42 ----------------
    ax = axes[1, 1]
    sub = df[(df.tissue == "hemo") & (df.iteration == 4)]
    # annotation offsets are explicit per rung: at 6 pt the old "always to the right"
    # rule pushed the right-most label ("curated prior") past the panel edge and pushed
    # the top-most one ("legacy blood prior") into the panel title.  The two rung
    # descriptions ("(52 neural TFs)", "(11 TFs, 325 edges)") now live only in the
    # caption: kept in the panel they forced a second line that the leader line ran through.
    ladder = [("R14_hemato_vanilla", "no prior", VANILLA, (6, -1, "left")),
              ("R14_hemato_seed42", "legacy blood prior", GREY, (6, -3, "left")),
              ("R14_wrongprior_seed42", "wrong-tissue prior",
               fp.ARM_PRIOR_ALT, (8, 8, "left")),
              ("R13_hemato_trueprior_seed42", "curated prior",
               C_HEMA, (-8, 4, "right"))]
    xs, ys, cs = [], [], []
    for run, lab, col, (dx, dy, ha) in ladder:
        r = sub[sub.run == run]
        if r.empty:
            continue
        xs.append(r.tr.iloc[0])
        ys.append(r.pr.iloc[0])
        cs.append(col)
        ax.annotate(lab, (r.tr.iloc[0], r.pr.iloc[0]),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=6.0, color=col, ha=ha)
    ax.plot(xs, ys, "-", color=GREY, lw=LW_DATA, zorder=1)
    ax.scatter(xs, ys, c=cs, s=24, zorder=3, edgecolors="white", linewidth=LW_EDGE)
    ax.set_xlabel(r"total latent variance  tr(Cov $\mathbf{Z}$)")
    ax.set_ylabel("effective spectral rank")
    ax.set_title(f"{fp.panel('e')} Specificity ladder",
                 fontweight="bold", fontsize=8.0)
    ax.set_xlim(right=625)
    style_ax(ax)

    # ---------------- (f) neural 2^3 factorial main effects ----------------
    ax = axes[1, 2]
    neu = df[(df.tissue == "neural") & (df.iteration == 4)]
    fac = neu[neu.run.str.startswith(("R0_", "R1_", "R2_", "R3_", "R4_",
                                      "R5_", "R6_", "R7_"))]
    y = fac["pr"].to_numpy(float)
    X = np.column_stack([np.ones(len(fac)), fac["C"], fac["M"], fac["T"]])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = len(fac) - 4
    se = np.sqrt(np.sum(res ** 2) / dof * np.diag(np.linalg.pinv(X.T @ X)))
    names = ["curriculum\n(C)", "mixture\n(M)", "decoder audit\n(T)"]
    ax.axhline(0, color=GREY, lw=LW_REF)
    for i in range(3):
        b, s_ = beta[i + 1], se[i + 1]
        ax.errorbar(i, b, yerr=1.96 * s_, fmt="o", color=C_NEURAL, ms=4.0,
                    ecolor=C_NEURAL, capsize=fp.CAPSIZE, elinewidth=LW_EDGE + 0.3)
    ax.set_xticks(range(3))
    ax.set_xticklabels(names, fontsize=6.2)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylabel(r"$\beta$ (95% CI)", fontsize=7.2)
    ax.set_title(f"{fp.panel('f')} Neural $2^3$ factorial",
                 fontweight="bold", fontsize=8.0)
    style_ax(ax)
    ax.text(0.5, 0.04, "no component effect\n(all $|t|<0.3$, $p>0.39$)",
            transform=ax.transAxes, fontsize=6.0, color=fp.MUTED,
            ha="center", va="bottom")

    stem = FIGDIR / "fig2_z_spectrum_3tissue"
    fp.export(fig, str(stem))
    print("wrote", stem.with_suffix(".pdf"))
    print("wrote", stem.with_suffix(".png"))
    print("wrote", FIGDIR / "deck" / "fig2_z_spectrum_3tissue_deck.png")


if __name__ == "__main__":
    main()
