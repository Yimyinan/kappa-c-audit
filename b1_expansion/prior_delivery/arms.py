#!/usr/bin/env python3
"""prior_delivery.arms -- the four arms of the prior-delivery experiment.

Why these four
--------------
The prior does not reach UNAGI's loss as a number; it reaches it through
``transfer_to_ranking_score`` (utils/attribute_utils.py:49), which converts the
per-gene weight of each cell into ``1 + 1/sqrt(rank)``. Only the *ordering* of
the weights survives. Two consequences:

1. Monotone rescalings of the prior (sqrt, log1p, min-max) produce byte-identical
   training. The "raw vs sqrt" comparison I proposed earlier is vacuous, and the
   concern about 1.5**9 = 38.4 blowing up the loss is misplaced -- 38.4 and 1.5
   differ only in rank, and only for genes that carry different TF in-degrees.
2. The things that *can* change the loss are: whether a gene is in the prior at
   all, and what order the prior assigns within it. That is exactly the content
   question paper1 asks, now testable in a delivered setting.

So the arms are:

``empty``    no prior (wrapper's disable_tf_prior). geneWeight stays all-zero,
             ranking is uniform, the loss is unweighted. Baseline.
``graded``   the real TRRUST-derived prior, weight 1.5**(TF in-degree).
``binary``   same support (same 203 genes), flat weight -- grading removed.
``shuffle``  the real multiset of boost values, reassigned to random genes.
             Preserves the number of boosted genes and the exact distribution of
             in-degrees; destroys only which genes carry them. This is paper1's
             degree-preserving content control.
"""
from __future__ import annotations

import numpy as np


def _gene_index(gene_names):
    return {g.upper(): i for i, g in enumerate(gene_names)}


def build_graded(gene_names, tf_database, boost_factor: float = 1.5):
    """Reimplementation of pg_unagi.tf_database.build_prior_matrix on bare names.

    Same arithmetic (multiply 1.5 per TF that targets the gene), but it takes the
    gene list instead of an AnnData so the arms can be built without loading data.
    """
    w = np.ones(len(gene_names), dtype=np.float64)
    idx = _gene_index(gene_names)
    for _tf, targets in tf_database.items():
        for t in targets:
            j = idx.get(str(t).upper())
            if j is not None:
                w[j] *= boost_factor
    return w


def arm_binary(graded, level: float = 2.0):
    """Same support, flat weight: removes the in-degree grading, keeps the gene set."""
    w = np.ones_like(graded)
    w[graded > 1.0] = level
    return w


def arm_shuffle(graded, seed: int = 0):
    """Degree-preserving content shuffle.

    Takes the multiset of boost values the real prior assigned (one per boosted
    gene) and hands them to the same number of uniformly chosen genes. The number
    of boosted genes is unchanged, and so is the distribution of boost values --
    which is the distribution of TF in-degrees. Only the gene identities move.
    """
    rng = np.random.default_rng(seed)
    w = np.ones_like(graded)
    vals = graded[graded > 1.0].copy()
    rng.shuffle(vals)
    j = rng.choice(w.size, vals.size, replace=False)
    w[j] = vals
    return w


def arm_shuffle_flat(graded, level: float = 5.0, seed: int = 0):
    """Content control matched to a *flat* dose: same support size, same value,
    different genes.

    ``arm_shuffle`` is the right control for the rank form, where the graded boost
    values are what the trainer sees. In direct mode a graded shuffle would hand the
    raw in-degree multipliers (1.5--38.4) to random genes, which is a different and
    much stronger intervention than a flat 5x. This variant keeps everything the
    flat arm has -- number of genes and exact weight -- and moves only which genes
    carry it, so a per-gene comparison between the two arms is paired on gene
    identity with nothing else varying.
    """
    rng = np.random.default_rng(seed)
    w = np.ones_like(graded)
    n = int((graded > 1.0).sum())
    j = rng.choice(w.size, n, replace=False)
    w[j] = level
    return w


def ranking_profile(prior, n_show: int = 3):
    """What the trainer actually sees: 1 + 1/sqrt(rank), per gene.

    Mirrors transfer_to_ranking_score so the report can state the effective
    weight range instead of the raw boost factors (which never reach the loss).
    """
    from scipy.stats import rankdata

    prior = np.asarray(prior, dtype=np.float64).ravel()
    od = prior.size - rankdata(prior) + 1
    score = 1.0 + 1.0 / np.sqrt(od)
    boosted = prior > 1.0
    return dict(
        n_boosted=int(boosted.sum()),
        raw_min=float(prior[boosted].min()) if boosted.any() else 1.0,
        raw_max=float(prior[boosted].max()) if boosted.any() else 1.0,
        eff_top=float(score[boosted].max()) if boosted.any() else float(score.max()),
        eff_bottom_boosted=float(score[boosted].min()) if boosted.any() else float(score.min()),
        eff_unboosted=float(score[~boosted].mean()) if (~boosted).any() else float(score.min()),
    )


def describe(graded, binary, shuffled):
    rows = []
    for name, w in (("graded", graded), ("binary", binary), ("shuffle", shuffled)):
        p = ranking_profile(w)
        rows.append((name, p))
    return rows


# --------------------------------------------------------------------------
# support ladder
# --------------------------------------------------------------------------

def trrust_degree_ranking(gene_names, trrust_path):
    """Target genes present in the data, ordered by how many distinct TFs hit them.

    Returns ``(ranked_names, n_available)``. Used to widen the prior beyond the 11
    curated hematopoietic TFs: TRRUST supplies 735 TFs, but only ~2,000 of the
    20,527 genes in this dataset are targets of any of them, which is the natural
    upper bound on a real GRN support here.
    """
    import pandas as pd

    tr = pd.read_csv(trrust_path, sep="\t", header=None, comment="#",
                     names=["TF", "target", "dir", "pmid"])
    present = {str(g).upper() for g in gene_names}
    tr["t"] = tr["target"].astype(str).str.upper()
    tr["f"] = tr["TF"].astype(str).str.upper()
    tr = tr[tr["t"].isin(present)]
    deg = tr.groupby("t")["f"].nunique().sort_values(ascending=False)
    return list(deg.index), int(len(deg))


def expand_support(base_prior, gene_names, ranked, k, level=5.0):
    """Widen a flat prior to ``k`` genes: keep the curated core, add top-degree genes.

    The core is whatever ``base_prior`` already boosts, so the ladder's narrow end is
    the same 203-gene prior used in Parts 1 and 2; wider rungs add the highest
    in-degree TRRUST targets not already included. Every rung is therefore a real
    GRN prior, not a real prior padded with random genes.
    """
    w = np.ones(len(base_prior), dtype=np.float64)
    sel = set(np.where(np.asarray(base_prior) > 1.0)[0].tolist())
    idx = {str(g).upper(): i for i, g in enumerate(gene_names)}
    for g in ranked:
        if len(sel) >= k:
            break
        j = idx.get(str(g).upper())
        if j is not None:
            sel.add(j)
    for j in sel:
        w[j] = level
    return w, len(sel)
