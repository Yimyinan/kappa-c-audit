# -*- coding: utf-8 -*-
"""null_protocol_preflight.py -- guard the cross-arm null-hypothesis defect class.

Root cause of the S11 Panel A bug: the published pair was re-run with
`labels_key` (the INERT arm), so its seed null must come from the inert arm
(`Q2` in s1m5_pbmc3k_verdict.json). The categorical-covariate arm (Panel B)
must use the categorical null (`Q3`). A null mismatched to its arm's training
protocol is a silent mis-attribution -- the table still "looks" fine.

Reads the RENDERED table (supplementary/table_s11_body.tex) and the canonical
verdict JSONs, then asserts:
  * Panel A null == inert (Q2) values, and Panel A published == Q2 published
  * Panel B trace null == categorical (Q3) trace values
  * Panel B corrected z == Q3 z_vs_null_corrected
  * CROSS-ARM GUARD: Panel A null is NOT the categorical null, and Panel B null
    is NOT the inert null (detects the original swap)
  * optional: Panel B corrected null mean/sd == s1m4 cross_seed_vanilla_null

Exit code 1 if any assertion fails.
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.abspath(os.path.join(ROOT, "..", "..", "outputs", "positive_result"))
DATA = os.path.join(ROOT, "data")
S11 = os.path.join(ROOT, "supplementary", "table_s11_body.tex")

VERDICT = "s1m5_pbmc3k_verdict.json"
INJECT = "s1m4_pbmc3k_injection.json"

TRACE_PM = re.compile(r"trace form & \$([-\d.]+)\\pm([-\d.]+)\$ & "
                     r"\$([-\d.]+)\\pm([-\d.]+)\$, \$z=([-\d.]+)\$")
TRACE_SIMPLE = re.compile(r"trace form & \$([-\d.]+)\$ & "
                          r"\$([-\d.]+)\\pm([-\d.]+)\$, \$z=([-\d.]+)\$")
CORRECTED = re.compile(r"corrected & \$([-\d.]+)\\pm([-\d.]+)\$ & "
                       r"\$([-\d.]+)\\pm([-\d.]+)\$, \$z=([-\d.]+)\$")


def load(name):
    for base in (OUT, DATA):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return json.load(open(p))
    return None


def jget(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def main() -> int:
    v = load(VERDICT)
    inj = load(INJECT)
    if v is None:
        print(f"ERROR: {VERDICT} not found", file=sys.stderr)
        return 2
    text = open(S11, encoding="utf-8", errors="ignore").read()

    q2 = jget(v, "Q2_published_pair")
    q3 = jget(v, "Q3_genuine_injection")
    q2_null = (q2["null_mean"], q2["null_sd"], q2["z"])
    q2_pub = q2["published_kappa_C_W_trace"]
    q3_trace = jget(q3, "kappa_C_W_trace")
    q3_trace_null = (q3_trace["null_mean"], q3_trace["null_sd"])
    q3_trace_pub = q3_trace["mean"]
    q3_corr_z = q3["z_vs_null_corrected"]

    # parse the rendered table -- each row yields 5 numbers (corrected/trace rows)
    # or 4 (the simple Panel A trace row). Identified by keyword, not fragile anchors.
    a_pub = a_nm = a_ns = a_z = None
    b_pub = b_nm = b_ns = b_z = None
    bc_pub = bc_nm = bc_ns = bc_z = None
    parsed = {"A": False, "Btrace": False, "Bcorr": False}
    for line in text.splitlines():
        if "trace form" in line and TRACE_SIMPLE.search(line):
            m = TRACE_SIMPLE.search(line)
            a_pub, a_nm, a_ns, a_z = (float(m.group(i)) for i in range(1, 5))
            parsed["A"] = True
        elif "trace form" in line and TRACE_PM.search(line):
            m = TRACE_PM.search(line)
            b_pub, _b_pubsd, b_nm, b_ns, b_z = (float(m.group(i)) for i in range(1, 6))
            parsed["Btrace"] = True
        elif "corrected" in line and CORRECTED.search(line):
            m = CORRECTED.search(line)
            bc_pub, _bc_pubsd, bc_nm, bc_ns, bc_z = (float(m.group(i)) for i in range(1, 6))
            parsed["Bcorr"] = True

    if not all(parsed.values()):
        print(f"  [WARN] S11 rows not all parsed: {parsed}", file=sys.stderr)

    checks = []

    def chk(label, got, want, tol):
        if got is None or want is None:
            checks.append(False)
            print(f"  [FAIL] {label:<52} got={got} want={want} (missing value)")
            return
        ok = abs(got - want) <= tol
        checks.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<52} got={got:.4f} want={want:.4f} tol={tol}")

    print("== null-protocol preflight (S11 cross-arm guard) ==")
    print("-- Panel A must use the INERT (labels_key) null --")
    chk("A null mean == Q2(inert) mean", a_nm, q2_null[0], 1e-2)
    chk("A null sd   == Q2(inert) sd", a_ns, q2_null[1], 1e-2)
    chk("A null z    == Q2(inert) z", a_z, q2_null[2], 2e-2)
    chk("A published == Q2 published", a_pub, q2_pub, 1e-2)

    print("-- Panel B must use the CATEGORICAL null --")
    chk("B trace null mean == Q3 trace null mean", b_nm, q3_trace_null[0], 1e-2)
    chk("B trace null sd   == Q3 trace null sd", b_ns, q3_trace_null[1], 1e-2)
    chk("B trace z        == Q3 trace z", b_z, q3["z_vs_null_trace"], 2e-2)
    chk("B published      == Q3 trace mean", b_pub, q3_trace_pub, 1e-2)
    chk("B corrected z    == Q3 z_vs_null_corrected", bc_z, q3_corr_z, 1e-1)

    print("-- CROSS-ARM GUARD (the original defect) --")
    guard_a = (a_nm is not None) and (q3_trace_null[0] is not None) and abs(a_nm - q3_trace_null[0]) > 5e-2
    checks.append(guard_a)
    print(f"  [{'PASS' if guard_a else 'FAIL'}] Panel A null ({a_nm}) != categorical "
          f"null ({q3_trace_null[0]})  (swap would make them ~equal)")
    guard_b = (b_nm is not None) and (q2_null[0] is not None) and abs(b_nm - q2_null[0]) > 5e-2
    checks.append(guard_b)
    print(f"  [{'PASS' if guard_b else 'FAIL'}] Panel B null ({b_nm}) != inert "
          f"null ({q2_null[0]})  (swap would make them ~equal)")

    if inj is not None:
        cnull = jget(inj, "cross_seed_vanilla_null", "kappa_corrected")
        if cnull:
            chk("B corrected null mean == s1m4 vanilla null mean", bc_nm, cnull["mean"], 2e-2)
            chk("B corrected null sd   == s1m4 vanilla null sd", bc_ns, cnull["sd"], 2e-2)

    print("\nVERDICT:", "ALL PASS" if all(checks) else "FAILURES FOUND")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
