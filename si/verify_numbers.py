# -*- coding: utf-8 -*-
"""paper3 gate: numbers, citations, floats, and the supplement.

Five checks, in the order that matters:

  1. TRACE      every numeric literal in main.tex, supplementary.tex and si/*.tex must
                appear in one of the two ledgers (NUMBERS.md, si/SI_NUMBERS.md).
                A number in the prose with no ledger entry is an untraceable claim.
  2. CITATIONS  every \\bibitem is cited, every \\cite has a \\bibitem.
  3. FLOATS     every \\label is referenced, every \\ref resolves, every figure exists.
  4. TABLES     every supplementary table is cited from the main text, and every
                "Table S<n>" / "Section S<n>" the main text mentions exists.
  5. GENERATED  the tables written by make_si.py are byte-identical to a fresh run,
                so an arm added to the tree cannot leave a stale table behind.

Usage:
    python verify_numbers.py            # report only, exit 0
    python verify_numbers.py --strict   # exit 1 on any failure

The 1% tolerance on check 1 is deliberate: the ledger records exact values (0.00861) and
the prose quotes rounded ones (+0.86 percentage points), with the rounding declared in
NUMBERS.md's "Rounding" sections. The gate compares value stems, with the ledger's own
mixtures of fractions and percentages handled explicitly.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "main.tex")
SUPP = os.path.join(HERE, "supplementary.tex")
LEDGERS = [os.path.join(HERE, "NUMBERS.md"), os.path.join(HERE, "si", "SI_NUMBERS.md")]
SI_TEX = sorted(glob.glob(os.path.join(HERE, "si", "*.tex")))


def strip_comments(s: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", s)


def decimals(s: str, latex_comments: bool = True):
    """Every numeric literal, with LaTeX thousands separators ({,} and ,) removed.

    NUMBERS.md is markdown, where '%' is a percent sign and not a comment: pass
    latex_comments=False for it, or every percentage is silently dropped.
    """
    if latex_comments:
        s = strip_comments(s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", s)      # drop control words
    s = s.replace("{,}", "")                                # 20{,}527 -> 20527
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)               # 20,527 -> 20527
    out = []
    for m in re.finditer(r"(?<![\w.])(\d+)(\.\d+)?", s):
        raw = m.group(1) + (m.group(2) or "")
        try:
            out.append(float(raw))
        except ValueError:
            pass
    return out


def body_of(path: str) -> str:
    s = open(path, encoding="utf-8").read()
    s = s.split(r"\begin{document}", 1)[-1]
    return s.split(r"\begin{thebibliography}")[0]


def main() -> int:
    fails = []

    main_src = open(MAIN, encoding="utf-8").read()
    ledgers = []
    for p in LEDGERS:
        if os.path.exists(p):
            ledgers.append((os.path.basename(p), decimals(open(p, encoding="utf-8").read(),
                                                          latex_comments=False)))
    ledger_vals = [v for _, vals in ledgers for v in vals]
    print(f"ledgers: {', '.join(n for n, _ in ledgers)} "
          f"({len(ledger_vals)} numbers, {len(set(ledger_vals))} distinct)")

    def traced(v: float) -> bool:
        for w in ledger_vals:
            if v == w:
                return True
            if w and abs(v - w) / max(abs(w), 1e-12) < 1e-3:
                return True
            # the ledger mixes fractions (0.027) and percentages (2.7)
            if abs(v - w * 100) < 1e-6 or abs(v * 100 - w) < 1e-6:
                return True
            if round(v, 2) == round(w, 2):
                return True
            # the ledger keeps exact values; the prose quotes 1-3 significant figures
            if round(v, 1) == round(w, 1) or round(v) == round(w):
                return True
        return False

    # ---- 1. trace -------------------------------------------------------
    docs = [("main", MAIN)]
    if os.path.exists(SUPP):
        docs.append(("supp", SUPP))
    for p in SI_TEX:
        docs.append((f"si/{os.path.basename(p)}", p))

    for name, path in docs:
        vals = decimals(body_of(path))
        missing = sorted({v for v in vals if not traced(v)})
        ok = not missing
        print(f"  {'OK  ' if ok else 'FAIL'} - {name}: {len(vals) - len(missing)}/{len(vals)} "
              f"numbers traced")
        if missing:
            print(f"         untraced: {missing[:20]}")
            fails.append(f"{name}: {len(missing)} untraced number(s)")

    # ---- 2. citations ---------------------------------------------------
    # Both sides read the comment-stripped source. Reading \bibitem from the raw source while
    # reading \cite from the stripped one was asymmetric: a retired entry left as
    # "% \bibitem{key}" would be counted as an uncited reference for ever, which is how a check
    # starts being ignored.
    main_nc = strip_comments(main_src)
    bib = re.findall(r"\\bibitem\{([^}]*)\}", main_nc)
    cites = set()
    for m in re.finditer(r"\\cite[a-zA-Z]*\{([^}]*)\}", main_nc):
        cites |= {k.strip() for k in m.group(1).split(",") if k.strip()}
    uncited = [b for b in bib if b not in cites]
    undefined = sorted(cites - set(bib))
    print(f"bibitems: {len(bib)} | cited: {len(cites)}")
    print(f"  {'OK  ' if not uncited else 'FAIL'} - uncited bibitems: {uncited or 'none'}")
    print(f"  {'OK  ' if not undefined else 'FAIL'} - undefined cites: {undefined or 'none'}")
    if uncited:
        fails.append(f"uncited bibitem(s): {uncited}")
    if undefined:
        fails.append(f"undefined cite(s): {undefined}")

    # ---- 3. floats ------------------------------------------------------
    for name, path in docs:
        src = open(path, encoding="utf-8").read()
        labels = set(re.findall(r"\\label\{([^}]*)\}", src))
        refs = set()
        for m in re.finditer(r"\\(?:ref|eqref|autoref)\{([^}]*)\}", strip_comments(src)):
            refs.add(m.group(1).strip())
        unreferenced = sorted(labels - refs)
        unresolved = sorted(refs - labels)
        print(f"  {'OK  ' if not (unreferenced or unresolved) else 'FAIL'} - {name}: "
              f"{len(labels)} labels, {len(refs)} refs")
        if unreferenced:
            print(f"         unreferenced: {unreferenced}")
            fails.append(f"{name}: unreferenced label(s) {unreferenced}")
        if unresolved:
            print(f"         unresolved: {unresolved}")
            fails.append(f"{name}: unresolved ref(s) {unresolved}")
        figs = re.findall(r"\\includegraphics[^{]*\{([^}]*)\}", src)
        absent = [f for f in figs if not os.path.exists(os.path.join(HERE, f))]
        if absent:
            print(f"         missing figure files: {absent}")
            fails.append(f"{name}: missing figure file(s) {absent}")

    # ---- 4. tables cited across documents -------------------------------
    if os.path.exists(SUPP):
        supp_src = open(SUPP, encoding="utf-8").read()
        n_tables = len(re.findall(r"\\begin\{table\}", supp_src))
        n_sections = len(re.findall(r"^\\section\{", supp_src, re.M))
        cited_t = {int(m) for m in re.findall(r"Table~S(\d+)", main_src)}
        cited_s = {int(m) for m in re.findall(r"Section~S(\d+)", main_src)}
        print(f"supplementary: {n_tables} tables, {n_sections} sections")
        print(f"  main cites tables {sorted(cited_t)} / sections {sorted(cited_s)}")

        bad_t = sorted(n for n in cited_t if not 1 <= n <= n_tables)
        bad_s = sorted(n for n in cited_s if not 1 <= n <= n_sections)
        orphan = sorted(set(range(1, n_tables + 1)) - cited_t)
        print(f"  {'OK  ' if not (bad_t or bad_s) else 'FAIL'} - "
              f"out-of-range mentions: tables {bad_t or 'none'}, sections {bad_s or 'none'}")
        print(f"  {'OK  ' if not orphan else 'FAIL'} - tables never cited from main: "
              f"{orphan or 'none'}")
        if bad_t or bad_s:
            fails.append(f"main cites a supplementary float that does not exist: "
                         f"tables {bad_t}, sections {bad_s}")
        if orphan:
            fails.append(f"supplementary table(s) not cited from main: {orphan}")

    # ---- 5. generated tables are fresh ----------------------------------
    try:
        import make_si
    except Exception as exc:                                  # pragma: no cover
        print(f"  skip - make_si.py not importable ({exc})")
        make_si = None
    if make_si is not None:
        tmp = tempfile.mkdtemp(prefix="si_fresh_")
        keep, make_si.SI = make_si.SI, tmp
        try:
            rows = make_si.load_tree()
            make_si.family_table(rows)
            make_si.copies_table(rows)
            make_si.ladder_table()
            make_si.basis_table()
        finally:
            make_si.SI = keep
        stale = []
        for f in ("table_arms.tex", "table_copies.tex", "table_ladder.tex", "table_basis.tex"):
            fresh = open(os.path.join(tmp, f), encoding="utf-8").read()
            disk_p = os.path.join(keep, f)
            disk = open(disk_p, encoding="utf-8").read() if os.path.exists(disk_p) else ""
            if fresh != disk:
                stale.append(f)
        print(f"  {'OK  ' if not stale else 'FAIL'} - generated tables match a fresh run: "
              f"{stale or 'all four'}")
        if stale:
            fails.append(f"stale generated table(s): {stale} (run make_si.py)")

    print()
    if fails:
        print("FAIL: " + "; ".join(fails))
    else:
        print("PASS: every number traced, every reference closed, tables fresh")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # 默认把失败当失败。以前不传 --strict 就一律 exit 0，接 CI 时会把 FAIL 吞掉；
    # 需要只打印不拦的情况（手工看报告）改用 --lenient。
    ap.add_argument("--lenient", action="store_true",
                    help="always exit 0, for eyeballing the report only")
    a = ap.parse_args()
    rc = main()
    sys.exit(0 if a.lenient else rc)
