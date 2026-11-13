"""The in-loss half of the four-check protocol: turn a spectral claim into a
verifiable one.

A high compression ``C`` (low ``kappa``) is routinely read as "the injected prior
was used". It is not: ``C`` is a trace-collapse diagnostic, and deep compression
can arise from the data alone. The three checks below are the computational forms
used in the manuscript to separate the two. They test whether an injected
quantity is active *within* the loss; the protocol's fourth check -- the
injection-path check -- asks whether the quantity reached the loss at all, and
is a source inspection rather than a computation, so it has no function here
(see ``README.md``). Run that fourth check *first*: if the prior never reached
the objective, these three can only return false negatives. A flat scan or a
contained collapse is uninformative when the target response is structurally
infeasible on the data manifold.

  1. ``gradient_check``       -- does the spectral response track the prior's
                                 gradient contribution?
  2. ``prior_strength_scan``  -- does ``kappa`` respond to the prior strength
                                 ``lambda``?  A flat scan is evidence the collapse
                                 is not prior-driven.
  3. ``structural_bound``     -- is the claimed compression even feasible given
                                 the vanilla trace and spectral gap?
"""

from __future__ import annotations

import numpy as np

__all__ = ["gradient_check", "prior_strength_scan", "structural_bound"]


def gradient_check(spectral_response, prior_gradient_norm):
    """Check 1. Correlate the per-configuration spectral response with the prior's
    gradient contribution.

    Parameters
    ----------
    spectral_response : array-like
        Per-configuration spectral quantity (e.g. ``C`` in percent, or ``kappa``).
    prior_gradient_norm : array-like
        Per-configuration magnitude of the prior's gradient contribution.

    Returns
    -------
    dict with ``pearson``, ``spearman``, ``n`` and ``prior_driven`` (True when the
    correlation is materially non-zero, ``|spearman| >= 0.5`` and ``n >= 4``).
    """
    from scipy.stats import pearsonr, spearmanr

    a = np.asarray(spectral_response, float).ravel()
    b = np.asarray(prior_gradient_norm, float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return dict(pearson=float("nan"), spearman=float("nan"), n=len(a),
                    prior_driven=False, note="degenerate input")
    pr = float(pearsonr(a, b)[0])
    sr = float(spearmanr(a, b).correlation)
    return dict(pearson=pr, spearman=sr, n=len(a),
                prior_driven=bool(abs(sr) >= 0.5 and len(a) >= 4))


def prior_strength_scan(kappa_by_lambda):
    """Check 2. Scan ``kappa`` across prior strengths ``lambda``.

    Parameters
    ----------
    kappa_by_lambda : mapping or sequence of (lambda, kappa) pairs

    Returns
    -------
    dict with ``slope`` (linear fit of kappa vs lambda), ``range`` (max-min),
    ``monotone`` and ``responds`` (True when the range exceeds 0.05, i.e. the scan
    is not flat).
    """
    if isinstance(kappa_by_lambda, dict):
        items = sorted(kappa_by_lambda.items(), key=lambda kv: float(kv[0]))
        lam = np.array([float(k) for k, _ in items])
        kap = np.array([float(v) for _, v in items])
    else:
        arr = np.asarray(kappa_by_lambda, float)
        lam, kap = arr[:, 0], arr[:, 1]
    if len(lam) < 2:
        return dict(slope=float("nan"), range=0.0, monotone=False, responds=False)
    slope = float(np.polyfit(lam, kap, 1)[0]) if lam.std() > 1e-12 else 0.0
    rng = float(kap.max() - kap.min())
    diffs = np.diff(kap)
    mono = bool(np.all(diffs >= 0) or np.all(diffs <= 0))
    return dict(slope=slope, range=rng, monotone=mono, responds=bool(rng > 0.05))


def structural_bound(C_percent, trH_vanilla, gap_vanilla):
    """Check 3. Upper bound on the attainable gap ratio at compression ``C``.

    Implements the manuscript bound

        kappa_gap <= (1 - C) * tr(H_V) / gap_V,

    where ``C_percent`` is the global compression in percent, ``trH_vanilla`` the
    vanilla Hessian trace and ``gap_vanilla`` the vanilla spectral gap. If the
    training target exceeds this bound, the target is structurally infeasible on
    that dataset.
    """
    c = float(C_percent) / 100.0
    return float((1.0 - c) * float(trH_vanilla) / float(gap_vanilla))
