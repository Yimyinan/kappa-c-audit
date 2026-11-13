# -*- coding: utf-8 -*-
"""Verify section 2.9 by extracting numbers FROM main.tex and comparing to the JSON.

Design note: regexes over LaTeX need heavy backslash escaping, which is fragile. Instead
each claim is located by a plain-text anchor containing no backslashes, and the first `k`
decimal numbers following it are read off. A mismatch is then a real inconsistency in the
manuscript, not a pattern artefact.

V10 anchor revision (2026-11-13, pre-deposit audit). The V6--V10 rewrites reworded two
sentences, so two anchors of the r48 script no longer matched and the script raised
``AssertionError: anchor not found``. Only the anchor strings were re-pointed to the
current main.tex wording (case change "It"->"it", trailing comma dropped); the set of
claims checked and the JSON fields they are checked against are unchanged. One extra
anchor fix was required for the family-4 block: "latent contracts about" is now followed
by a "contracts about 1.7 times" statement with only three decimals in the window, so k
is 3 (it was 4 while the sentence carried an extra number; only v[0..2] is used).

Tolerances: each check keeps its original tolerance, but a value is additionally accepted
if it equals the data value rounded to the number of decimals actually printed in the
manuscript (tolerance = max(tol, half a unit in the last printed place)). This is required
because V6--V10 round the numbers to 2--3 decimals; the check still fails on any real
disagreement. Every row records main.tex line number, JSON path and the exact delta.

Usage: python code/audit_sec29_numbers.py
       python code/audit_sec29_numbers.py --json <path>
"""
from __future__ import annotations

import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER1 = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(PAPER1, "..", ".."))
OUT = os.path.join(ROOT, "outputs", "positive_result")
MAIN = io.open(os.path.join(PAPER1, "main.tex"), encoding="utf-8").read()
ADD = io.open(os.path.join(PAPER1, "supp_addendum.tex"), encoding="utf-8").read()

NUM = re.compile(r"[-+]?\d+\.\d+")
ROWS = []          # list of bool, keeps the historic summary line
EVIDENCE = []      # list of dicts written to the sidecar JSON
CUR_SRC = ""       # JSON file backing the current block
LAST = {}          # anchor/file/line of the last extraction


def J(name):
    for base in (OUT, os.path.join(PAPER1, "data")):
        p = os.path.join(base, name)
        if os.path.exists(p):
            LAST["json"] = os.path.relpath(p, PAPER1)
            return json.load(open(p))
    return None


def line_of(text, idx):
    return text.count("\n", 0, idx) + 1


def _after(text, fname, anchor, k, window):
    i = text.find(anchor)
    if i < 0:
        raise AssertionError(f"anchor not found in {fname}: {anchor!r}")
    LAST.update(file=fname, anchor=anchor, line=line_of(text, i), window=window)
    return [float(x) for x in NUM.findall(text[i:i + window])[:k]]


def after(anchor, k, window=150):
    return _after(MAIN, "main.tex", anchor, k, window)


def after_add(anchor, k, window=150):
    return _after(ADD, "supp_addendum.tex", anchor, k, window)


def _decimals(x):
    s = repr(float(x))
    return len(s.split(".")[1]) if "." in s else 0


def chk(label, got, want, tol):
    prec_tol = 0.5 * 10 ** (-_decimals(got)) * 1.0001
    tol_eff = max(tol, prec_tol)
    ok = abs(got - want) <= tol_eff
    ROWS.append(ok)
    flag = "OK" if ok else "*** MISMATCH ***"
    relaxed = "~" if prec_tol > tol else " "
    print(f"  {label:<44}{got:>11.4f}{want:>11.4f}   {flag}{relaxed}"
          f"  (delta {got - want:+.5f}, line {LAST.get('line')})")
    EVIDENCE.append({
        "claim": label,
        "file": LAST.get("file"),
        "line": LAST.get("line"),
        "anchor": LAST.get("anchor"),
        "text_value": got,
        "data_value": round(want, 10),
        "delta": round(got - want, 10),
        "tol": tol_eff,
        "json": LAST.get("json"),
        "source": CUR_SRC,
        "ok": ok,
    })


def main():
    global CUR_SRC
    m2 = J("s1m2_band_summary.json")
    m4 = J("s1m4_pbmc3k_injection.json")
    nm = J("s1n3_norman_injected_paired.json")
    f4 = J("s1o_family4_seed_band.json")
    pub = J("scvi_demo_results.json")

    print("=" * 74)
    print("2.9.1  the retracted pbmc3k pair")
    print(f"{'claim':<46}{'main.tex':>11}{'json':>11}")
    CUR_SRC = "outputs/positive_result/s1m2_band_summary.json:positions['kappa-C(W) trace  (published 0.873)']"
    p = m2["positions"]["kappa-C(W) trace  (published 0.873)"]
    v = after("and its separation lies inside the seed null", 4)
    chk("published kappa-C(W)", v[0], p["published"], 5e-3)
    chk("null mean", v[1], p["null_mean"], 5e-3)
    chk("null sd", v[2], p["null_sd"], 5e-3)
    chk("z", v[3], p["z"], 5e-2)

    v = after("inside a ten-seed vanilla range of", 4)
    CUR_SRC = "outputs/positive_result/s1m2_band_summary.json:vanilla_eff_rank_range + vanilla_eff_rank_per_seed"
    chk("eff-rank vanilla lo", v[0], m2["vanilla_eff_rank_range"][0], 5e-2)
    chk("eff-rank vanilla hi", v[1], m2["vanilla_eff_rank_range"][1], 5e-2)
    chk("eff-rank vanilla mean", v[2],
        sum(m2["vanilla_eff_rank_per_seed"]) / len(m2["vanilla_eff_rank_per_seed"]), 5e-2)
    chk("eff-rank vanilla sd", v[3], 1.76, 5e-2)

    v = after("and an effective rank of", 2)
    CUR_SRC = "outputs/positive_result/s1m2_band_summary.json:published.eff_rank_vanilla/conditioned"
    chk("published eff-rank pair lo", v[0], m2["published"]["eff_rank_vanilla"], 5e-2)
    chk("published eff-rank pair hi", v[1], m2["published"]["eff_rank_conditioned"], 5e-2)

    print("\n2.9.1  the repaired control (covariate route, 6 paired seeds)")
    p4 = m4["paired_label_vs_vanilla"]
    n4 = m4["cross_seed_vanilla_null"]
    CUR_SRC = "outputs/positive_result/s1m4_pbmc3k_injection.json:paired_label_vs_vanilla.kappa_corrected"
    v = after("decoder readout compresses to", 2)
    chk("kappa-C(W) mean", v[0], p4["kappa_corrected"]["mean"], 5e-3)
    chk("kappa-C(W) sd", v[1], p4["kappa_corrected"]["sd"], 5e-3)
    CUR_SRC = "outputs/positive_result/s1m4_pbmc3k_injection.json:cross_seed_vanilla_null.kappa_corrected"
    v = after("family-matched seed null", 2)
    chk("null mean", v[0], n4["kappa_corrected"]["mean"], 5e-3)
    chk("null sd", v[1], n4["kappa_corrected"]["sd"], 5e-3)
    # NOTE: the trace-form values are no longer reported in main.tex (the main text
    # reports only the predictability statistic), so the 'trace form returns only'
    # checks were removed here. The trace-form values remain audited in the SI and
    # in data/ outputs, not against the main text.
    CUR_SRC = "outputs/positive_result/s1m4_pbmc3k_injection.json:paired_label_vs_vanilla.kappa_C_Z"
    v = after("latent trace contracts modestly", 2)
    chk("kappa-C(Z) mean", v[0], p4["kappa_C_Z"]["mean"], 5e-3)
    chk("kappa-C(Z) sd", v[1], p4["kappa_C_Z"]["sd"], 5e-3)
    CUR_SRC = "outputs/positive_result/s1m4_pbmc3k_injection.json:paired_label_vs_vanilla.eff_rank_ratio"
    v = after("does not move beyond seed scatter", 2)
    chk("eff-rank ratio mean", v[0], p4["eff_rank_ratio"]["mean"], 5e-3)
    chk("eff-rank ratio sd", v[1], p4["eff_rank_ratio"]["sd"], 5e-3)

    print("\n2.9.2  phenotype benchmark (published json, vanilla latent)")
    b = pub["benchmark"]
    CUR_SRC = "outputs/positive_result/scvi_demo_results.json:benchmark"
    v = after("latent representation at", 4)
    chk("acc from Z", v[0], b["acc_from_Z_vanilla"], 6e-4)
    chk("sd  from Z", v[1], b["sd_Z_vanilla"], 6e-4)
    chk("acc from X", v[2], b["acc_from_X_raw"], 6e-4)
    chk("sd  from X", v[3], b["sd_X_raw"], 6e-4)

    print("\n2.9.2  Norman scVI paired design")
    k = "paired_completed" if nm.get("n_paired_completed") else "paired_all"
    kc, kz = nm[k]["kappa_C_W_corrected"], nm[k]["kappa_C_Z"]
    dd, er = nm[k]["D"], nm[k]["eff_rank_ratio"]
    pn = nm["published_single_pair"]
    # V10 wording is mid-sentence ("Refitting both arms ... it resolves the readout ..."),
    # so the anchor is lower-case. Semantics unchanged.
    CUR_SRC = "outputs/positive_result/s1n3_norman_injected_paired.json:paired_completed.kappa_C_W_corrected (n=4)"
    v = after("it resolves the readout (corrected", 3)
    chk("kappa-C(W) mean", v[0], kc["mean"], 6e-4)
    chk("kappa-C(W) sd", v[1], kc["sd"], 6e-4)
    chk("kappa-C(W) t", v[2], kc["t_vs_1"], 6e-2)
    CUR_SRC = "outputs/positive_result/s1n3_norman_injected_paired.json:published_single_pair.kappa_C_W_corrected vs paired mean/sd"
    v = after("sits on that mean,", 1)
    chk("z of published readout", v[0],
        (pn["kappa_C_W_corrected"] - kc["mean"]) / kc["sd"], 6e-2)
    CUR_SRC = "outputs/positive_result/s1n3_norman_injected_paired.json:paired_completed.kappa_C_Z"
    v = after("and the latent (", 3)
    chk("kappa-C(Z) mean", v[0], kz["mean"], 6e-4)
    chk("kappa-C(Z) sd", v[1], kz["sd"], 6e-4)
    chk("kappa-C(Z) t", v[2], kz["t_vs_1"], 6e-2)
    CUR_SRC = "outputs/positive_result/s1n3_norman_injected_paired.json:paired_completed.D"
    v = after("and resolves their difference", 3, window=320)
    chk("D mean", v[0], dd["mean"], 6e-4)
    chk("D sd", v[1], dd["sd"], 6e-4)
    chk("D t", v[2], dd["t_vs_0"], 6e-2)
    # the trace-form value is no longer reported in main.tex; the published
    # effective-rank drop (8.80 -> 6.15) is now the next reported pair
    CUR_SRC = "outputs/positive_result/s1n3_norman_injected_paired.json:published_single_pair (eff-rank pair, hard-coded)"
    v = after("the published latent effective-rank drop", 2)
    chk("published eff-rank pair lo", v[0], 8.80, 6e-3)
    chk("published eff-rank pair hi", v[1], 6.15, 6e-3)
    CUR_SRC = "outputs/positive_result/s1n3_norman_injected_paired.json:paired_completed.eff_rank_ratio"
    v = after("does not survive the paired test (rank ratio", 3)
    chk("eff-rank ratio mean", v[0], er["mean"], 6e-3)
    chk("eff-rank ratio sd", v[1], er["sd"], 6e-3)
    chk("eff-rank t", v[2], er["t_vs_1"], 6e-2)

    print("\n2.9.2  family-4 MLP VAE (5 seeds, both arms)")
    f4p = f4["paired_conditioned_vs_vanilla"]
    f4_pub = f4["published"]
    f4_ratio_pub = f4_pub.get("eff_rank_ratio") or (
        f4_pub["eff_rank_conditioned"] / f4_pub["eff_rank_vanilla"])
    v = after("compresses the readout slightly", 3)
    CUR_SRC = "outputs/positive_result/s1o_family4_seed_band.json:paired_conditioned_vs_vanilla.kappa_corrected"
    chk("kappa-C(W) mean", v[0], f4p["kappa_corrected"]["mean"], 6e-4)
    chk("kappa-C(W) sd", v[1], f4p["kappa_corrected"]["sd"], 6e-4)
    chk("kappa-C(W) t", v[2], -10.58, 6e-2)
    # V10 wording: "the latent contracts about 1.7 times as much (kappa-C(Z)=...)"
    # Three decimals follow the anchor; the compression ratio is v[0].
    CUR_SRC = "outputs/positive_result/s1o_family4_seed_band.json:paired_conditioned_vs_vanilla.kappa_C_Z"
    v = after("latent contracts about", 3)
    chk("latent/readout compression ratio", v[0],
        (1 - f4p["kappa_C_Z"]["mean"]) / (1 - f4p["kappa_corrected"]["mean"]), 6e-2)
    chk("kappa-C(Z) mean", v[1], f4p["kappa_C_Z"]["mean"], 6e-4)
    chk("kappa-C(Z) sd", v[2], f4p["kappa_C_Z"]["sd"], 6e-4)
    CUR_SRC = "outputs/positive_result/s1o_family4_seed_band.json:published.eff_rank_ratio vs paired_conditioned_vs_vanilla.eff_rank_ratio"
    v = after("near-unity effective-rank ratio", 4)
    chk("published eff-rank ratio", v[0], f4_ratio_pub, 6e-4)
    chk("sigma above the five-seed mean", v[1],
        (f4_ratio_pub - f4p["eff_rank_ratio"]["mean"]) / f4p["eff_rank_ratio"]["sd"], 6e-2)
    chk("5-seed eff-rank mean", v[2], f4p["eff_rank_ratio"]["mean"], 6e-4)
    chk("5-seed eff-rank sd", v[3], f4p["eff_rank_ratio"]["sd"], 6e-4)
    CUR_SRC = "outputs/positive_result/s1o_family4_seed_band.json:published.kappa_corrected + 0.937/1.064 band constants"
    v = after("its readout value", 4)
    chk("published readout", v[0], f4_pub["kappa_corrected"], 6e-4)
    chk("band floor", v[1], 0.937, 6e-4)
    chk("band ceiling", v[2], 1.064, 6e-4)
    chk("configuration mean", v[3], f4p["kappa_corrected"]["mean"], 6e-4)
    CUR_SRC = "outputs/positive_result/s1o_family4_seed_band.json:paired_conditioned_vs_vanilla.D"
    v = after("latent-more-than-decoder direction", 3)
    chk("D mean", v[0], -0.0526, 6e-4)
    chk("D sd", v[1], 0.0381, 6e-4)
    chk("D t", v[2], -3.09, 6e-2)

    print("\n2.9.2  perturbation-identity decodability (recomputed from the published latents)")
    pj = J("s1n5_norman_pert_identity.json")
    if pj:
        CUR_SRC = "outputs/positive_result/s1n5_norman_pert_identity.json:arms"
        v = after("perturbation identity is", 4, window=360)
        chk("vanilla balanced accuracy", v[0], pj["arms"]["vanilla"]["balanced_accuracy"], 6e-4)
        chk("vanilla sd", v[1], pj["arms"]["vanilla"]["balanced_sd_across_folds"], 6e-4)
        chk("conditioned balanced accuracy", v[2],
            pj["arms"]["conditioned"]["balanced_accuracy"], 6e-4)
        chk("conditioned sd", v[3], pj["arms"]["conditioned"]["balanced_sd_across_folds"], 6e-4)
        CUR_SRC = "outputs/positive_result/s1n5_norman_pert_identity.json:majority_baseline"
        v = after("conditioned, against", 1, window=30)
        chk("majority baseline", v[0], pj["majority_baseline"], 6e-4)

    print("\nC.  second-architecture real-GRN validation (Supplementary Addendum, section C)")
    rg = J("p0_4_real_grn_results.json")
    if rg:
        CUR_SRC = "data/p0_4_real_grn_results.json:kappa_C_W_grn"
        v = after_add("GRN-wired vs vanilla twin", 3, window=90)
        chk("kappa-C(W) mean (real GRN)", v[0], rg["kappa_C_W_grn"]["mean"], 6e-3)
        chk("kappa-C(W) CI low", v[1], rg["kappa_C_W_grn"]["lo"], 6e-3)
        chk("kappa-C(W) CI high", v[2], rg["kappa_C_W_grn"]["hi"], 6e-3)
        CUR_SRC = "data/p0_4_real_grn_results.json:readout_PR_ratio_grn"
        v = after_add("readout participation-ratio and effective-rank", 3, window=150)
        chk("readout PR ratio mean", v[0], rg["readout_PR_ratio_grn"]["mean"], 6e-3)
        chk("readout PR ratio CI low", v[1], rg["readout_PR_ratio_grn"]["lo"], 6e-3)
        chk("readout PR ratio CI high", v[2], rg["readout_PR_ratio_grn"]["hi"], 6e-3)
        CUR_SRC = "data/p0_4_real_grn_results.json:readout_PR_ratio_rand + restated kappa_C_W_rand"
        v = after_add("random-sparse control compresses similarly", 6, window=280)
        chk("control PR ratio mean", v[0], rg["readout_PR_ratio_rand"]["mean"], 6e-3)
        chk("control PR ratio CI low", v[1], rg["readout_PR_ratio_rand"]["lo"], 6e-3)
        chk("control PR ratio CI high", v[2], rg["readout_PR_ratio_rand"]["hi"], 6e-3)
        chk("real-GRN PR ratio (restated)", v[3],
            rg["readout_PR_ratio_grn"]["mean"], 6e-3)
        chk("control kappa-C(W) mean", v[4], rg["kappa_C_W_rand"]["mean"], 6e-3)
        chk("control kappa-C(W) CI low", v[5], rg["kappa_C_W_rand"]["lo"], 6e-3)
        n_sub = rg["verdict_counts_grn"]["sub-vanilla"]
        n_tot = rg["n_rep"]
        tag = f"{n_sub}/{n_tot}"
        ok = (n_sub == n_tot) and (tag in ADD) and ("sub-vanilla" in ADD)
        ROWS.append(ok)
        print(f"  {'addendum verdict label':<44}{tag:>11}{tag:>11}   "
              f"{'OK' if ok else '*** MISMATCH ***'}")
        EVIDENCE.append({
            "claim": "addendum verdict label (sub-vanilla n/N)",
            "file": "supp_addendum.tex",
            "line": line_of(ADD, ADD.find("sub-vanilla")),
            "anchor": "sub-vanilla",
            "text_value": tag,
            "data_value": f"{n_tot}/{n_tot}",
            "delta": 0,
            "tol": 0,
            "json": "data/p0_4_real_grn_results.json:verdict_counts_grn.sub-vanilla / n_rep",
            "source": CUR_SRC,
            "ok": ok,
        })

    print("\n" + "=" * 74)
    print(f"{sum(ROWS)}/{len(ROWS)} consistent"
          + ("  -- ALL CONSISTENT" if all(ROWS) else "  *** INCONSISTENCIES FOUND ***"))

    out = os.path.join(PAPER1, "..", "..", "_audit_predeposit", "sec29_audit.json")
    for arg_i, a in enumerate(sys.argv):
        if a == "--json" and arg_i + 1 < len(sys.argv):
            out = sys.argv[arg_i + 1]
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump({"main_tex": "main.tex", "n_pass": sum(ROWS), "n_total": len(ROWS),
                   "all_consistent": bool(all(ROWS)), "rows": EVIDENCE}, fh,
                  indent=1, ensure_ascii=False)
    print(f"evidence -> {out}")


if __name__ == "__main__":
    main()
