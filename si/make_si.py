# -*- coding: utf-8 -*-
"""Build the generated tables of the paper3 supplementary material.

Writes, under paper3/si/:
  table_arms.tex     the arm tree by family (directories, distinct artefacts, what it spans)
  table_copies.tex   the three shared no-prior baselines and the directories that copy them
  table_ladder.tex   the per-seed dose ladder (matched d and kappa-C(W))
  SI_NUMBERS.md      every value emitted above, with its source artefact

Nothing here is typed by hand: the counts come from walking prior_delivery_runs/ and
hashing latent.npz, the ladder from dose_ladder.json. Re-run after adding arms:

    python make_si.py
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"F:\UNAGI_Project\prior_delivery_runs"
SEEDS = (42, 43, 44)      # dose_summary.py iterates --seeds [42, 43, 44]
SI = os.path.join(HERE, "si")
os.makedirs(SI, exist_ok=True)


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def load_tree():
    """Every arm directory that carries a model artefact, with its metadata and hash."""
    rows = []
    for dp, dn, fn in os.walk(ROOT):
        if "latent.npz" not in fn:
            continue
        rel = os.path.relpath(dp, ROOT).replace(os.sep, "/")
        meta = {}
        mp = os.path.join(dp, "arm_meta.json")
        if os.path.exists(mp):
            meta = json.load(open(mp, encoding="utf-8"))
        h = hashlib.sha256(open(os.path.join(dp, "latent.npz"), "rb").read()).hexdigest()
        rows.append({"rel": rel, "family": rel.split("/")[0], "hash": h, "meta": meta})
    return rows


def family_table(rows):
    fam = defaultdict(list)
    for r in rows:
        fam[r["family"]].append(r)

    lines = [
        r"\begin{tabular}{@{}lrcclll@{}}",
        r"\toprule",
        r"Family & Dirs & Distinct & Seeds & Dose & Weighted epochs & Support \\",
        r"\midrule",
    ]
    ledger = []
    for f in sorted(fam):
        rs = fam[f]
        seeds = sorted({r["meta"].get("seed") for r in rs if r["meta"].get("seed") is not None})
        # dose: only arms that actually carry a boost identify the level (a no-prior or
        # permuted arm records the trainer default, which is not a dose of this family)
        doses = sorted(
            {r["meta"].get("prior_level") for r in rs
             if r["meta"].get("prior_level") is not None
             and (r["meta"].get("prior") or {}).get("n_boosted", 0) > 0}
        )
        eps = sorted({r["meta"].get("epoch_iter") for r in rs if r["meta"].get("epoch_iter")})
        if f == "calib":            # the ladder is 2/5/10/20/45 total epochs, not a weighted phase
            eps = []
        sup = sorted({r["meta"].get("support_k") for r in rs if r["meta"].get("support_k")})
        distinct = len({r["hash"] for r in rs})

        def rng(vals, suffix=""):
            if not vals:
                return "--"
            v = [f"{x:g}" for x in vals]
            if len(v) == 1:
                return v[0] + suffix
            return f"{v[0]}--{v[-1]}{suffix}"

        lines.append(
            f"\\texttt{{{tex_escape(f)}}} & {len(rs)} & {distinct} & {rng(seeds)} & "
            f"{rng(doses, r'$\times$')} & {rng(eps)} & {rng(sup)} \\\\"
        )
        ledger.append((f, len(rs), distinct, seeds, doses, eps, sup))

    lines += [
        r"\midrule",
        f"\\textbf{{total}} & \\textbf{{{len(rows)}}} & \\textbf{{{len({r['hash'] for r in rows})}}} & & & & \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    open(os.path.join(SI, "table_arms.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return ledger


def copies_table(rows):
    """The no-prior baselines that several families share, one table row per baseline."""
    by_hash = defaultdict(list)
    for r in rows:
        if r["meta"].get("arm") == "empty" or r["rel"].endswith("empty") or "/empty" in r["rel"]:
            by_hash[r["hash"]].append(r["rel"])
    groups = {h: v for h, v in by_hash.items() if len(v) > 1}

    lines = [
        r"\begin{tabular}{@{}p{1.9cm}p{11.4cm}@{}}",
        r"\toprule",
        r"Baseline & Directories holding a byte-identical copy \\",
        r"\midrule",
    ]
    order = sorted(groups.values(), key=lambda v: sorted(v)[0])
    for i, dirs in enumerate(order, start=1):
        dirs = sorted(dirs)
        seed = next((r["meta"].get("seed") for r in rows if r["rel"] == dirs[0]), None)
        listing = ", ".join(r"\texttt{" + tex_escape(d) + "}" for d in dirs)
        lines.append(f"no-prior, seed {seed} & {listing} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(SI, "table_copies.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return {i + 1: sorted(v) for i, v in enumerate(order)}


def ladder_table():
    d = json.load(open(os.path.join(ROOT, "dose_ladder.json"), encoding="utf-8"))
    levels = [float(x) for x in d["levels"]]
    lines = [
        r"\begin{tabular}{@{}rrrr@{}}",
        r"\toprule",
        r"Dose & Seed & Matched $d$ & $\kCW$ \\",
        r"\midrule",
    ]
    ledger = []
    for lvl in levels:
        m = d["means"][str(lvl)]
        for seed, dd, kk in zip(SEEDS, m["matched"], m["kcw"]):
            lines.append(f"{lvl:g}$\\times$ & {seed} & ${dd:+.5f}$ & ${kk:.5f}$ \\\\")
            ledger.append((lvl, seed, dd, kk))
        lines.append(
            f" & mean & ${sum(m['matched']) / len(m['matched']):+.5f}$ & "
            f"${sum(m['kcw']) / len(m['kcw']):.5f}$ \\\\"
        )
        ledger.append((lvl, "mean", sum(m["matched"]) / len(m["matched"]), sum(m["kcw"]) / len(m["kcw"])))
        if lvl != levels[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(SI, "table_ladder.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return ledger


def basis_table():
    """The basis audit on both datasets, straight from the two JSON artefacts."""
    haem = json.load(open(os.path.join(ROOT, "b1", "basis_variance_audit.json"), encoding="utf-8"))
    mel = json.load(open(os.path.join(ROOT, "b1", "basis_audit_frangieh.json"), encoding="utf-8"))

    def shares(d):
        out = {}
        for r in d["rows"]:
            out[r["basis"]] = (r["block_share"], r.get("n_covered"))
        return out

    lines = [
        r"\begin{tabular}{@{}lrrrrrrrr@{}}",
        r"\toprule",
        r"Dataset & Cells & Genes & Covered & Spectral & Indicator & Permuted & PCA-32 & NMF-32 \\",
        r"\midrule",
    ]
    ledger = []
    for label, d in (("haematopoiesis", haem), ("Perturb-CITE-seq", mel)):
        s = shares(d)
        cells, genes = d["n_cells"], d["n_genes"]
        cov = s["grn_spectral"][1]
        lines.append(
            f"{label} & {cells:,} & {genes:,} & {cov:,} & "
            f"{s['grn_spectral'][0] * 100:.2f} & {s['grn_indicator'][0] * 100:.2f} & "
            f"{s['permuted'][0] * 100:.2f} & {s['pca32'][0] * 100:.1f} & {s['nmf32'][0] * 100:.1f} \\\\"
        )
        ledger.append((label, cells, genes, cov, {k: round(v[0] * 100, 2) for k, v in s.items()}))
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(SI, "table_basis.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return ledger, mel["build"]


def baseline_sums():
    """Sum R2 of the three 75-epoch no-prior baselines, and their mean."""
    import numpy as np

    sums = []
    for s in ("s42", "s43", "s44"):
        p = os.path.join(ROOT, "dose20_e45", s, "empty_e45", "pergene_readout_r2.npy")
        sums.append(float(np.load(p).sum()))
    return sums


def write_ledger(fam, copies, ladder, rows, basis, build, sums):
    out = [
        "# SI_NUMBERS -- values emitted into the paper3 supplementary tables",
        "",
        "Generated by `make_si.py` on every run; do not edit by hand. The gate",
        "(`verify_numbers.py`) accepts a number in `supplementary.tex` or `si/*.tex` if it",
        "appears either here or in `NUMBERS.md`. Every row names the artefact it came from, so",
        "the two ledgers together cover the main text and the supplement.",
        "",
        "## The arm tree (`si/table_arms.tex`, `si/table_copies.tex`)",
        "",
        "| quantity | value | source |",
        "|---|---|---|",
        f"| arm directories holding `latent.npz` | {len(rows)} | `walk prior_delivery_runs/` |",
        f"| distinct model artefacts (sha256) | {len({r['hash'] for r in rows})} | sha256 of each `latent.npz` |",
        f"| byte-identical copies | {len(rows) - len({r['hash'] for r in rows})} | {len(rows)} - {len({r['hash'] for r in rows})} |",
        f"| baselines copied across families | {len(copies)} groups | `si/table_copies.tex` |",
        "",
        "| family | directories | distinct | seeds | dose | weighted epochs | support |",
        "|---|---|---|---|---|---|---|",
    ]
    for f, n, d_, seeds, doses, eps, sup in fam:
        out.append(
            f"| `{f}` | {n} | {d_} | {seeds} | {doses} | {eps} | {sup} |"
        )
    out += [
        "",
        "## Per-seed dose ladder (`si/table_ladder.tex`)",
        "",
        "Source: `prior_delivery_runs/dose_ladder.json` (`means[dose]['matched']`, `['kcw']`).",
        "`d` is the matched per-gene readout change (real minus weight-matched permutation, against a",
        "matched no-prior baseline); `kcw` is $\\kappa$-$C(W)$ for the same arm.",
        "",
        "| dose | seed | matched d | kappa-C(W) |",
        "|---|---|---|---|",
    ]
    for lvl, seed, d_, k in ladder:
        out.append(f"| {lvl:g} | {seed} | {d_:+.5f} | {k:.5f} |")

    out += [
        "",
        "## The two basis-audit datasets (`si/table_basis.tex`)",
        "",
        "Sources: `prior_delivery_runs/b1/basis_variance_audit.json` (haematopoiesis, 3,000-cell audit",
        "draw) and `b1/basis_audit_frangieh.json` (Perturb-CITE-seq; the deposited file is",
        "`data/public/frangieh_2021.h5ad`, whose `uns/doi` is 10.1038/s41588-021-00779-1).",
        "",
        "| dataset | cells | genes | covered | spectral | indicator | permuted | PCA-32 | NMF-32 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label, cells, genes, cov, sh in basis:
        out.append(
            f"| {label} | {cells:,} | {genes:,} | {cov:,} | {sh['grn_spectral']} | "
            f"{sh['grn_indicator']} | {sh['permuted']} | {sh['pca32']} | {sh['nmf32']} |"
        )
    out += [
        "",
        f"| TFs in the second dataset's GRN | {build.get('n_tf', '')} | `b1/basis_audit_frangieh.json:build` |",
        f"| edges in the second dataset's GRN | {build.get('n_edges', '')} | same |",
        "",
        "## Sum R2 of the 75-epoch no-prior baselines",
        "",
        "Each is `dose20_e45/s*/empty_e45/pergene_readout_r2.npy`, summed over 20,527 genes. The main",
        "text quotes the mean of the three; the two-convention table of the supplement quotes seed 42,",
        "which is why the two figures differ in the third digit.",
        "",
        "| seed | sum R2 |",
        "|---|---|",
    ]
    for s, v in zip(("42", "43", "44"), sums):
        out.append(f"| {s} | {v:.1f} |")
    out.append(f"| mean | {sum(sums) / len(sums):.1f} |")
    out.append("")
    open(os.path.join(SI, "SI_NUMBERS.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")


def main():
    rows = load_tree()
    fam = family_table(rows)
    copies = copies_table(rows)
    ladder = ladder_table()
    basis, build = basis_table()
    sums = baseline_sums()
    write_ledger(fam, copies, ladder, rows, basis, build, sums)
    print(f"arm directories: {len(rows)}  distinct: {len({r['hash'] for r in rows})}  "
          f"copies: {len(rows) - len({r['hash'] for r in rows})}")
    print(f"families: {len(fam)}  shared baselines: {len(copies)}")
    print("wrote si/table_arms.tex, si/table_copies.tex, si/table_ladder.tex, si/SI_NUMBERS.md")


if __name__ == "__main__":
    main()
