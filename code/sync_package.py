# -*- coding: utf-8 -*-
"""Sync paper1's submission package to a frozen release closure, then rebuild the
package MANIFEST. Idempotent; safe to re-run after every freeze.

Usage:  <system python> code/sync_package.py [release-dir-name]
        default release dir name: the newest *-r<NUM> under releases/ (this
        project renamed release dirs mid-way, so the r-number is the only
        stable key -- globbing a name prefix silently picked a stale release)

Why a script and not a one-off: this has been needed after every narrative revision
(r17 -> r18 -> r20 -> r21). It also *verifies* the copy (missing / extra / hash
mismatch), which a plain copy would not.
"""
import io, os, re, shutil, hashlib, sys, glob

PAPER1 = r"C:/Users/12901/WorkBuddy/2026-09-03-18-47-34/paper1"
ROOT = PAPER1 + "/paper1_kc_diagnostic"
# The 2026-09-19 package was archived on 2026-11-10 (renamed with an _archive_
# prefix) because its manuscript predates the PLOS rewrite. The live package is
# now the 2026-11-10 one, re-cut from the newest frozen release after every
# narrative revision (r39 anti-defensive pass, r40 Chinese sync + SI caption).
# Never re-point this at an _archive_
# directory: a sync would overwrite a frozen record while looking like a routine
# package refresh. The guard below makes that mistake fail loudly instead.
PKG = PAPER1 + "/paper1_SUBMISSION_PACKAGE_20261110"
DST = PKG + "/01_manuscript"

if os.path.basename(PKG.rstrip("/")).startswith("_archive_"):
    sys.exit("refusing to sync into an archived package: %s\n"
             "No live submission package exists as of 2026-11-10. Cut a new one\n"
             "from releases/ and re-point PKG in this script before re-running."
             % os.path.basename(PKG))

INTERMEDIATE = (".aux", ".log", ".out", ".toc", ".bbl", ".blg", ".spl", ".fls", ".fdb_latexmk")

def newest_release():
    # releases/ changed naming mid-project: ..._s1-corrected-r32 (through r32),
    # then ..._plos-format-r33 and later. A glob on the old prefix therefore
    # picked r32 while the working tree was at r37 -- a silent two-revision
    # regression rather than an error. Match on the r-number alone and refuse
    # to guess when nothing matches.
    dirs = [d for d in glob.glob(ROOT + "/releases/*") if os.path.isdir(d)]
    def rnum(d):
        m = re.search(r"-r(\d+)$", os.path.basename(d))
        return int(m.group(1)) if m else -1
    dirs = [d for d in dirs if rnum(d) >= 0]
    if not dirs:
        sys.exit("no release directories matching *-r<NUM> under releases/")
    return sorted(dirs, key=rnum)[-1]

rel = sys.argv[1] if len(sys.argv) > 1 else newest_release()
SRC = os.path.join(rel, "source") if os.path.isdir(os.path.join(rel, "source")) else rel
print("release :", os.path.basename(rel))
print("source  :", SRC)
assert os.path.isdir(SRC), "no source dir"
assert os.path.isfile(os.path.join(SRC, "main.tex")), "no main.tex in source"

def dig(p):
    return hashlib.sha256(io.open(p, "rb").read()).hexdigest()

# Sync WITHOUT a bulk directory delete: `shutil.rmtree` on a 70-file dir trips
# [safe-delete] SAFE_DELETE_BULK_CONFIRM_REQUIRED (>50 files). Overwrite in place
# and remove only the few stale files individually.
#
# Intermediates are skipped on the way IN as well as deleted on the way OUT.  The
# frozen snapshots carry their own build leftovers (main.aux/.log/.out,
# supplementary.toc, ...), so copying them and then deleting them made this script
# report a constant "7 copied, 7 removed" on every single run -- which reads like
# the sync is not converging when in fact it is.
copied, removed, skipped_int = 0, 0, 0
for dp, _d, fs in os.walk(SRC):
    rel_dir = os.path.relpath(dp, SRC)
    dst_dir = DST if rel_dir == "." else os.path.join(DST, rel_dir)
    os.makedirs(dst_dir, exist_ok=True)
    for f in fs:
        if f.endswith(INTERMEDIATE):
            skipped_int += 1
            continue
        src_f, dst_f = os.path.join(dp, f), os.path.join(dst_dir, f)
        if os.path.isfile(dst_f) and dig(dst_f) == dig(src_f):
            continue
        shutil.copy2(src_f, dst_f)
        copied += 1

# Files that live in 01_manuscript/ but are NOT part of the release source: the
# TIFF renders of the main figures (tools/figures_to_tiff.py), which the journal
# takes as separate files. Without this list the stale-file pass below deletes
# them on every sync -- the package would silently lose its upload figures while
# reporting a clean closure.
KEEP_IN_DST = ("figures_tif/",)

# Remove DST files the frozen snapshot does not have (a revision can drop a
# figure or a section file; leaving it behind would silently keep it alive).
for dp, _d, fs in os.walk(DST):
    for f in fs:
        rel = os.path.relpath(os.path.join(dp, f), DST)
        keep = rel.replace("\\", "/").startswith(KEEP_IN_DST)
        if not keep and not os.path.isfile(os.path.join(SRC, rel)):
            os.remove(os.path.join(dp, f))
            removed += 1
print("synced: %d copied, %d stale removed, %d intermediates not copied"
      % (copied, removed, skipped_int))

removed = 0
for dp, _d, fs in os.walk(DST):
    for f in fs:
        if f.endswith(INTERMEDIATE):
            os.remove(os.path.join(dp, f))
            removed += 1
print("intermediates removed:", removed)

def index(root):
    out = {}
    for dp, _d, fs in os.walk(root):
        for f in fs:
            if f.endswith(INTERMEDIATE):
                continue
            p = os.path.join(dp, f)
            out[os.path.relpath(p, root).replace("\\", "/")] = dig(p)
    return out

a, b = index(SRC), index(DST)
missing, extra = sorted(set(a) - set(b)), sorted(set(b) - set(a))
# The TIFF renders are expected extras: they are generated from the source PDFs
# and are not part of the release. Count them separately so a real extra (a stale
# section file surviving a revision) is still reported as one.
tif = [k for k in extra if k.startswith(KEEP_IN_DST)]
extra = [k for k in extra if k not in tif]
mismatch = [k for k in sorted(set(a) & set(b)) if a[k] != b[k]]
print("closure files src=%d dst=%d | missing=%s extra=%s mismatch=%s"
      % (len(a), len(b), missing or "0", extra or "0", mismatch or "0"))
if tif:
    print("upload figures (expected extras, not release source): %d" % len(tif))

# Caches must never enter the integrity record: their contents change on every
# test run, so the MANIFEST becomes non-reproducible, and a `.pytest_cache/` in
# the delivered package is simply wrong.  Skipping them *here* -- rather than
# only cleaning the tree by hand -- is what stops this from coming back.
CACHE_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git"}
CACHE_SUFFIX = (".pyc", ".pyo")

entries, skipped = [], []
for dp, dirs, fs in os.walk(PKG):
    skipped += [os.path.join(dp, d) for d in dirs if d in CACHE_DIRS]
    dirs[:] = [d for d in dirs if d not in CACHE_DIRS]
    for f in sorted(fs):
        if f == "MANIFEST.sha256" or f.endswith(CACHE_SUFFIX):
            continue
        p = os.path.join(dp, f)
        entries.append((dig(p), os.path.getsize(p),
                        os.path.relpath(p, PKG).replace("\\", "/")))
entries.sort(key=lambda t: t[2])
if skipped:
    rel_skip = sorted(os.path.relpath(s, PKG).replace("\\", "/") for s in skipped)
    print("cache dirs skipped (not indexed): %d | %s%s"
          % (len(rel_skip), ", ".join(rel_skip[:5]),
             " ..." if len(rel_skip) > 5 else ""))
io.open(PKG + "/MANIFEST.sha256", "w", encoding="utf-8", newline="\n").write(
    "\n".join("%s  %10d  %s" % t for t in entries) + "\n")

# page counts straight from the copied PDFs
for n in ("main", "supplementary"):
    fp = os.path.join(DST, n + ".pdf")
    if os.path.isfile(fp):
        d = io.open(fp, "rb").read()
        print("  package %-14s %d pages" % (n, d.count(b"/Type /Page") - d.count(b"/Type /Pages")))

print("MANIFEST.sha256 rebuilt: %d entries | %.1f MB"
      % (len(entries), sum(s for _h, s, _r in entries) / 1024 / 1024))
