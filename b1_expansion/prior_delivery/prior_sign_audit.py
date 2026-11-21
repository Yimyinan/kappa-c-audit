#!/usr/bin/env python3
"""prior_delivery.prior_sign_audit -- does the prior carry direction at all?

Every number in Parts 1-9 measures *amplitude*: a per-gene multiplicative factor on
the reconstruction likelihood. TRRUST provides a Mode column (Activation / Repression
/ Unknown) and the prior builder never reads it, so the delivered object is
sign-blind by construction. This script quantifies what that costs, and which part of
the loss of information is due to the database rather than to the operator.

    py -3.13 -u prior_sign_audit.py
"""
from __future__ import annotations

import collections
import json
import os

import numpy as np

TRRUST = r"F:\UNAGI_Project\unagi_custom_data\trrust_raw_human.txt"
SUPPORT = r"F:\UNAGI_Project\prior_delivery_runs\dose20_e45\s42\real\support.npz"
OUT = r"F:\UNAGI_Project\prior_delivery_runs\prior_sign_audit.json"


def load_edges():
    rows = []
    with open(TRRUST, encoding="utf-8") as f:
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) >= 3:
                rows.append((p[0].upper(), p[1].upper(), p[2]))
    return rows


def main():
    edges = load_edges()
    mode = collections.Counter(m for _, _, m in edges)
    n = len(edges)

    sup = np.load(SUPPORT, allow_pickle=True)
    names = list(sup["gene_names"])          # gene_names is the full var_names vector
    idx = np.asarray(sup["idx"], dtype=int)
    genes = [str(names[i]).upper() for i in idx]
    gset = set(genes)

    # edges that touch the support, and their direction composition
    touch = [(tf, tg, m) for tf, tg, m in edges if tg in gset or tf in gset]
    touch_mode = collections.Counter(m for _, _, m in touch)

    # a multiplicative weight cannot express sign, so a gene hit by both an
    # activation and a repression edge is collapsed into one scalar
    by_gene = collections.defaultdict(set)
    for tf, tg, m in edges:
        if tg in gset:
            by_gene[tg].add(m)
    mixed = [g for g, ms in by_gene.items() if {"Activation", "Repression"} <= ms]
    signed = [g for g, ms in by_gene.items() if ms & {"Activation", "Repression"}]
    unknown_only = [g for g, ms in by_gene.items() if ms == {"Unknown"}]

    # distinct TFs per support gene: how much structure one scalar has to absorb
    tfs_per_gene = collections.Counter()
    for tf, tg, m in edges:
        if tg in gset:
            tfs_per_gene[tg] += 1
    multi = sum(1 for g, c in tfs_per_gene.items() if c > 1)

    rep = dict(
        edges_total=n,
        edge_mode_counts=dict(mode),
        edge_signed_fraction=round((mode["Activation"] + mode["Repression"]) / n, 4),
        support_genes=len(genes),
        edges_touching_support=len(touch),
        support_edge_mode_counts=dict(touch_mode),
        support_genes_with_known_direction=len(signed),
        support_genes_with_conflicting_direction=len(mixed),
        support_genes_unknown_only=len(unknown_only),
        support_genes_under_multiple_edges=multi,
        note=("A weight applied multiplicatively to the reconstruction likelihood "
              "cannot carry sign: the operator is the constraint, not the database. "
              "54% of TRRUST edges do have a direction, and it is discarded."),
    )

    print(json.dumps(rep, indent=1, ensure_ascii=False))
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
