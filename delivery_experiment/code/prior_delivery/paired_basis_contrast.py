#!/usr/bin/env python3
"""prior_delivery.paired_basis_contrast -- amplifing the measurement, not the effect.

The between-run estimate of the content effect is dA = +0.0088 with a replicate gap of
0.0029: four sigma, but only 1.1% of A, because two separately trained decoders differ in
every weight before the basis is even considered.

There is a statistic that removes that entire variance term. Every trained decoder can be
scored against BOTH bases. The contrast

    dA_within(arm) = A_C(J_arm) - A_P(J_arm)

uses one Jacobian, so no training noise enters. If the penalty shaped the decoder toward
its own basis, the arm trained on C must show dA_within > 0 and the arm trained on P must
show dA_within < 0. The content effect is then the difference of those two within-decoder
contrasts, which is still free of training variance because each contrast is conditioned
on its own decoder.

PRE-REGISTERED BEFORE LOOKING (2026-10-28 21:10):
  H1  dA_within(real) > 0, in both real replicates
  H2  dA_within(perm) < 0
  H3  interaction = dA_within(real) - dA_within(perm) > 0
  Decision: report the effect size as a fraction of A; if |interaction| < 2 x the spread
  between the two real replicates, call it unresolved rather than positive.

Secondary, also free, and more sensitive to *where* the geometry sits than to its overall
magnitude: restrict to the top-r left singular directions of J (r = 32), and ask what
fraction of that dominant subspace each basis captures.

    py -3.13 -u paired_basis_contrast.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = r"F:\UNAGI_Project\prior_delivery_runs\b1\modules_p32.npz"
ROOTS = {
    "ganoff": r"F:\UNAGI_Project\prior_delivery_runs\b1\ganoff\s42",
    "gan_on": r"F:\UNAGI_Project\prior_delivery_runs\b1\contrast\s42",
}
OUT = r"F:\UNAGI_Project\prior_delivery_runs\b1\paired_basis_contrast.json"


def load_bases(path=None):
    d = np.load(path or MODULES, allow_pickle=True)
    C, P = d["C"], d["P"]
    cov = np.where(np.abs(C).sum(1) > 0)[0]
    return C[cov], P[cov], cov


def orth(B):
    Q, _ = np.linalg.qr(B)
    return Q


def capture(J, Q):
    """Fraction of ||J||_F^2 that lies in span(Q)."""
    PJ = Q @ (Q.T @ J)
    return float((PJ ** 2).sum() / max((J ** 2).sum(), 1e-30))


def captured_top(J, Q, r=32):
    """Fraction of the top-r left-singular subspace of J captured by span(Q)."""
    U, S, _ = np.linalg.svd(J, full_matrices=False)
    V = U[:, :r]                       # dominant directions in gene space
    PV = Q @ (Q.T @ V)
    return float((PV ** 2).sum() / r)


def jacobian(arm_dir, cov):
    pths = [q for q in sorted(glob.glob(os.path.join(arm_dir, "model_save", "*_1.pth")))
            if "_dis_" not in os.path.basename(q)]        # never the discriminator
    if not pths:
        pths = [q for q in sorted(glob.glob(os.path.join(arm_dir, "model_save", "*.pth")))
                if "_dis_" not in os.path.basename(q)]
    if not pths:
        return None, None
    sd = torch.load(pths[-1], map_location="cpu", weights_only=False)
    sd = {str(k): v for k, v in sd.items()}             # older arms may lack the prefix
    def pick(suffix):
        for k in sd:
            if k.endswith(suffix):
                return sd[k]
        raise KeyError(suffix)
    W3 = pick("fc3.weight").double().numpy()             # (hidden, latent)
    b3 = pick("fc3.bias").double().numpy()
    W4 = pick("fc4.weight").double().numpy()             # (genes, hidden)
    z = np.load(os.path.join(arm_dir, "latent.npz"), allow_pickle=True)["Z"]
    z = np.asarray(z, dtype=np.float64)
    s = 1.0 / (1.0 + np.exp(-(z @ W3.T + b3)))          # (cells, hidden)
    W4c = W4[cov]                                       # (covered, hidden)
    return (W4c, W3, s), os.path.basename(pths[-1])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", default=MODULES)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--roots", nargs="+", default=list(ROOTS))
    ap.add_argument("--basis-dir",
                    help="score one arm directory (or several) against this basis")
    ap.add_argument("--arm-dirs", nargs="*", default=[])
    a = ap.parse_args()

    C, P, cov = load_bases(a.modules)
    QC, QP = orth(C), orth(P)
    print(f"covered genes {cov.size}, basis rank {C.shape[1]}")
    # how different are the two constraints at all?
    sv = np.linalg.svd(QC.T @ QP, compute_uv=False)
    print(f"principal cosines between span(C) and span(P): max {sv.max():.3f}, "
          f"mean {sv.mean():.3f}")

    rep = {"covered": int(cov.size),
           "basis_max_cos": float(sv.max()), "basis_mean_cos": float(sv.mean()),
           "arms": {}}

    jobs = []
    for root_name in a.roots:
        root = ROOTS[root_name]
        for arm in sorted(os.listdir(root)):
            d = os.path.join(root, arm)
            if os.path.isdir(d):
                jobs.append((f"{root_name}/{arm}", d))
    for d in a.arm_dirs:
        jobs.append((os.path.basename(os.path.dirname(d)) + "/" + os.path.basename(d), d))

    if True:
        for arm, d in jobs:
            pack, ckpt = jacobian(d, cov)
            if pack is None:
                continue
            W4c, W3, S = pack
            n = S.shape[0]
            acs, aps, tcs, tps = [], [], [], []
            for c in range(n):
                # J for one cell: (covered, latent). Never materialise the stack.
                J = (W4c * S[c][None, :]) @ W3
                acs.append(capture(J, QC)); aps.append(capture(J, QP))
                if c % 5 == 0:                     # SVD statistic on a 1-in-5 subsample
                    tcs.append(captured_top(J, QC)); tps.append(captured_top(J, QP))
            ac, ap = float(np.mean(acs)), float(np.mean(aps))
            tc, tp = float(np.mean(tcs)), float(np.mean(tps))
            rep["arms"][arm] = dict(
                ckpt=ckpt, A_C=float(ac), A_P=float(ap), dA_within=float(ac - ap),
                topC=float(tc), topP=float(tp), dtop_within=float(tc - tp))
            print(f"{arm:22s} A_C {ac:.4f} A_P {ap:.4f} "
                  f"dA_within {ac-ap:+.5f} | top32: {tc:.4f} vs {tp:.4f} "
                  f"({tc-tp:+.5f})", flush=True)

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {OUT}")

    arms = rep["arms"]
    key = lambda p: [v for k, v in arms.items() if k.startswith(p)]
    real = key("ganoff/real")
    perm = [v for k, v in arms.items() if k == "ganoff/perm"]
    if real and perm:
        r = np.mean([v["dA_within"] for v in real])
        spread = abs(real[0]["dA_within"] - real[-1]["dA_within"]) if len(real) > 1 else float("nan")
        inter = r - perm[0]["dA_within"]
        print(f"\nGAN-off: dA_within(real) {r:+.5f} (replicate spread {spread:.5f}), "
              f"dA_within(perm) {perm[0]['dA_within']:+.5f}")
        print(f"interaction {inter:+.5f}  = {100*inter/np.mean([v['A_C'] for v in real]):+.2f}% of A")
        rt = np.mean([v["dtop_within"] for v in real])
        inter_t = rt - perm[0]["dtop_within"]
        print(f"top-32 variant: real {rt:+.5f}, perm {perm[0]['dtop_within']:+.5f}, "
              f"interaction {inter_t:+.5f}")


if __name__ == "__main__":
    main()
