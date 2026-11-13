# -*- coding: utf-8 -*-
"""Freeze the current manuscript as a reproducible release snapshot.

Creates ``releases/<tag>/`` containing
    source/           the LaTeX sources actually consumed by the build, mirroring
                      the directory layout so the snapshot compiles standalone
    MANIFEST.sha256   sha256 + size of every source AND every data artifact the
                      build depends on (data artifacts are referenced, not copied)
    README.md         what the release is, what changed, how to rebuild, and the
                      headline numbers so the snapshot is self-describing

Usage: python code/freeze_release.py --tag 2026-09-17_s1-corrected

DEPRECATED (2026-11-08)
-----------------------
Prefer ``code/release_freezer.py`` for new snapshots: it derives the README from
the current working tree and is the tool used for r28 onward.  This legacy script
is kept because its MANIFEST also fingerprints data artifacts.  Its README
template used to hard-code the 2026-09-19 results (Fig. S26, "pbmc3k re-run
pending", a rebuild command naming a script that no longer exists); that text
was removed.  The template now describes the snapshot without duplicating any
result, so it cannot silently contradict the manuscript.  A warning is printed
at runtime to make it hard to cut a release with this tool by accident.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER1 = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(PAPER1, "..", ".."))

TOP_TEX = ["main.tex", "supplementary.tex", "star_methods.tex",
           "latent_residence_conclusion.tex", "submission_metadata.tex",
           "supp_addendum.tex"]

SOURCES_FIXED = [
    "kappa_c_audit/kappa_c_audit/core.py", "kappa_c_audit/kappa_c_audit/audit.py",
    "kappa_c_audit/kappa_c_audit/__init__.py", "kappa_c_audit/README.md",
    "kappa_c_audit/tests/test_core.py", "kappa_c_audit/examples/quickstart.py",
]


def discover_scripts():
    """Every analysis/figure/patch script, found by walking the tree rather than by name.

    A hardcoded list has silently dropped newly added scripts three times in this project
    (the reference closure, the archived figures, the manifest), so this is deliberately
    discovery-based: an extra file in a provenance record is harmless, a missing one is not.
    """
    out = []
    for sub in ("code", "figures"):
        d = os.path.join(PAPER1, sub)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".py") and not f.startswith("_"):
                out.append(f"{sub}/{f}")
    return out


SOURCES = discover_scripts() + SOURCES_FIXED

DATA = [
    "outputs/positive_result/s1f_0of33_recheck.json",
    "outputs/positive_result/s1g_corrected_summaries.json",
    "outputs/positive_result/s1h_null_bands.json",
    "outputs/positive_result/s1i_crossarch_recheck.json",
    "outputs/positive_result/s1j_holm_test1.json",
    "outputs/positive_result/s1k_gvae_band.json",
    "outputs/positive_result/s1l_pbmc3k_scvi.json",
    "outputs/positive_result/s1_live_path/s1e_reanalysis.json",
]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pdf_pages(p):
    """Page count from the PDF object tree.

    NB: counting form feeds in ``pdftotext`` output is NOT a page count -- layout
    mode emits extra form feeds (it reported 54 for a 37-page document).
    """
    import re
    try:
        with open(p, "rb") as f:
            data = f.read()
    except OSError:
        return None
    return len(re.findall(rb"/Type\s*/Page[^s]", data)) or None


def _targets(tex_text, cmd):
    """Resolve \\includegraphics / \\includepdf / \\input targets from .tex text."""
    import re
    out = []
    for m in re.finditer(re.escape(cmd) + r"\[[^\]]*\]\{([^}]+)\}", tex_text):
        out.append(m.group(1))
    for m in re.finditer(re.escape(cmd) + r"\{([^}]+)\}", tex_text):
        out.append(m.group(1))
    return out


def collect_inputs():
    """The true build closure: every .tex consumed plus every file it includes.

    Derived from the sources rather than assumed from the directory layout -- an
    earlier version listed directories by hand and silently missed
    ``supp_addendum.pdf`` at the project root, which only surfaced when the
    snapshot failed to compile standalone.
    """
    tex_files = []
    for t in TOP_TEX:
        p = os.path.join(PAPER1, t)
        if os.path.exists(p):
            tex_files.append(p)
    sub = os.path.join(PAPER1, "supplementary")
    if os.path.isdir(sub):
        for f in sorted(os.listdir(sub)):
            if f.endswith(".tex"):
                tex_files.append(os.path.join(sub, f))

    inputs = set(tex_files)
    for p in tex_files:
        text = open(p, encoding="utf-8", errors="ignore").read()
        for target in (_targets(text, r"\includegraphics")
                       + _targets(text, r"\includepdf")
                       + _targets(text, r"\input")):
            cand = os.path.normpath(os.path.join(PAPER1, target))
            if not os.path.exists(cand) and not cand.endswith(".tex"):
                cand_tex = cand + ".tex"
                if os.path.exists(cand_tex):
                    cand = cand_tex
            if os.path.exists(cand) and os.path.isfile(cand):
                inputs.add(cand)
    return sorted(inputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    print("[legacy freezer] DEPRECATED: use code/release_freezer.py for new "
          "releases (its README is generated from the current working tree).")

    rel = os.path.join(PAPER1, "releases", args.tag)
    if os.path.exists(rel):
        print(f"refusing: {rel} already exists")
        return 1
    src_dir = os.path.join(rel, "source")
    os.makedirs(src_dir)

    inputs = collect_inputs()
    copied = 0
    for p in inputs:
        rp = os.path.relpath(p, PAPER1)
        if rp.startswith("releases" + os.sep):
            continue
        dst = os.path.join(src_dir, rp)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(p, dst)
        copied += 1

    # the built PDFs belong in the snapshot too
    for name in ("main.pdf", "supplementary.pdf"):
        p = os.path.join(PAPER1, "build", name)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(rel, name))

    lines = [f"# release {args.tag}  frozen {datetime.now().isoformat(timespec='seconds')}", ""]
    lines.append("## manuscript sources (copied into source/)")
    for p in sorted(inputs):
        rp = os.path.relpath(p, PAPER1)
        if rp.startswith("releases" + os.sep):
            continue
        lines.append(f"{sha(p)}  {os.path.getsize(p):>10}  {rp}")
    lines.append("")
    lines.append("## analysis + figure + patch scripts (referenced)")
    for rp in SOURCES:
        p = os.path.join(PAPER1, rp) if not rp.startswith(("outputs", "kappa")) \
            else os.path.join(PAPER1, rp)
        if os.path.exists(p):
            lines.append(f"{sha(p)}  {os.path.getsize(p):>10}  {rp}")
        else:
            lines.append(f"{'MISSING':<64}  {rp}")
    lines.append("")
    lines.append("## data artifacts (referenced)")
    for rp in DATA:
        p = os.path.join(ROOT, rp)
        if os.path.exists(p):
            lines.append(f"{sha(p)}  {os.path.getsize(p):>10}  {rp}")
        else:
            lines.append(f"{'MISSING':<64}  {rp}")
    with open(os.path.join(rel, "MANIFEST.sha256"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    mp = os.path.join(rel, "main.pdf")
    sp = os.path.join(rel, "supplementary.pdf")
    head = {}
    for rp in ("outputs/positive_result/s1f_0of33_recheck.json",
               "outputs/positive_result/s1h_null_bands.json"):
        p = os.path.join(ROOT, rp)
        if os.path.exists(p):
            head[os.path.basename(rp)] = json.load(open(p, encoding="utf-8-sig"))

    readme = f"""# Release {args.tag}

Frozen snapshot of the κ-C diagnostic manuscript with the **corrected readout
statistic** fully applied. Self-contained: `source/` compiles standalone with the
same toolchain, and every data artifact used by the figures and tables is
identified by sha256 in `MANIFEST.sha256`.

* `main.pdf` — {pdf_pages(mp) or '?'} pages
* `supplementary.pdf` — {pdf_pages(sp) or '?'} pages
* sources copied: {copied} files
* frozen: {datetime.now().isoformat(timespec='seconds')}

## What this release contains

A self-contained LaTeX source closure for the κ-C diagnostic manuscript.  The
audited quantity and every headline number are those printed in the frozen
`main.pdf` / `supplementary.pdf` and their sources in `source/`.  No result is
duplicated in this README, so it cannot drift away from the manuscript it
describes.  Data artifacts consumed by figures and tables are identified by
sha256 in `MANIFEST.sha256`.

> Cut with the legacy `code/freeze_release.py`.  For releases from r28 onward,
> `code/release_freezer.py` is the supported tool and generates this file from
> the current working tree.

## Rebuild

```bash
cd source
# All generated figure/table bodies are already in the snapshot; a plain
# recompile needs no regeneration step.
for i in 1 2 3; do /d/MiKTeX/miktex/bin/x64/pdflatex.exe -interaction=nonstopmode main.tex; done
for i in 1 2 3; do /d/MiKTeX/miktex/bin/x64/pdflatex.exe -interaction=nonstopmode supplementary.tex; done
```

The regeneration scripts (`code/make_tabS1.py`, `code/make_tabS6.py`,
`code/make_tabS8.py`, `figures/make_fig2_kc_phase.py`,
`figures/make_figS5_sensitivity.py`, `figures/make_figS8_band_calibration.py`)
live in the working tree, not in the snapshot.  Run them there only when the
sources change, then cut a new dated release -- frozen snapshots are immutable.
"""
    with open(os.path.join(rel, "README.md"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write(readme)

    print(f"[frozen] {rel}")
    print(f"  sources copied : {copied}")
    print(f"  manifest lines : {len(lines)}")
    print(f"  pages          : main {pdf_pages(mp)}, supplementary {pdf_pages(sp)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
