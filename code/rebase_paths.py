# -*- coding: utf-8 -*-
"""rebase_paths.py -- re-point the analysis scripts at your own checkout.

Why this exists
---------------
71 of the 136 scripts under ``code/`` hard-code the author's local tree, e.g.
``C:/Users/12901/WorkBuddy/2026-09-03-18-47-34/outputs/positive_result``. Those
literals are **recorded provenance**: they are the paths at which the reported
numbers were produced. We keep them in the release rather than silently
rewriting history, but a public reader cannot execute the scripts as shipped,
and the path carries a user name.

This tool replaces only the *prefix* of those literals -- the machine-specific
session directory -- with a root you choose. It never touches numbers, figure
names, LaTeX, or any logic. The default is a dry run: it prints what would
change and writes nothing.

Usage
-----
    python code/rebase_paths.py                          # dry run, shows hits
    python code/rebase_paths.py --root /path/to/repo    # still a dry run
    python code/rebase_paths.py --root /path/to/repo --apply
    KAPPA_ROOT=/path/to/repo python code/rebase_paths.py --apply

With ``--apply``, every modified file is first copied into a backup directory
(``code/_rebase_backup/`` by default), so the operation is fully reversible.
Run ``python code/rebase_paths.py --verify`` afterwards to confirm no literal
remains.

Scope / honest limits
---------------------
Rebasing the prefix makes the scripts *addressable* on your machine; it does
not recreate the inputs. The expression matrices are third-party datasets and
are not bundled (see *Reproducibility notes* in the README). ``data/`` contains
processed JSON outputs, not raw inputs. The standalone ``kappa_c_audit`` package
has no such dependency and runs on its own.
"""
import argparse
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# This file is expected at <release>/code/rebase_paths.py; the scripts it scans
# live beside it (including subdirectories).
CODE_DIR = HERE
PKG_ROOT = os.path.dirname(HERE)

# Matches both forward-slash and backslash forms, and both the 2026-09-03 and
# 2026-08-30 author session directories. Deliberately narrow: only the machine
# root is matched, never a filename or a number inside a path.
OLD_ROOT = re.compile(
    r"(?:[A-Za-z]:|/c)[\\/]Users[\\/]12901[\\/]WorkBuddy[\\/]"
    r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}"
)


def scan(root):
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in ("__pycache__", "_rebase_backup", ".git")]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            with open(p, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            n = len(OLD_ROOT.findall(text))
            if n:
                hits.append((p, text, n))
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("KAPPA_ROOT"),
                    help="local root that replaces the author's session directory "
                         "(default: $KAPPA_ROOT, else the package root)")
    ap.add_argument("--apply", action="store_true", help="actually write files (default: dry run)")
    ap.add_argument("--backup-dir", default=os.path.join(CODE_DIR, "_rebase_backup"),
                    help="where originals are copied before writing")
    ap.add_argument("--verify", action="store_true", help="only check for remaining literals")
    args = ap.parse_args()

    hits = scan(CODE_DIR)

    if args.verify:
        print("scripts still carrying an author path: %d" % len(hits))
        for p, _t, n in hits[:10]:
            print("  %-70s %d" % (os.path.relpath(p, PKG_ROOT), n))
        return 1 if hits else 0

    if not hits:
        print("No author-local path literals found. Nothing to do.")
        return 0

    new_root = (args.root or PKG_ROOT).replace("\\", "/").rstrip("/")
    total = sum(n for _p, _t, n in hits)
    print("scripts with author-local paths : %d" % len(hits))
    print("literals to rebase             : %d" % total)
    print("replacement root               : %s" % new_root)
    for p, _t, n in hits[:8]:
        print("  %-70s %d" % (os.path.relpath(p, PKG_ROOT), n))
    if len(hits) > 8:
        print("  ... and %d more" % (len(hits) - 8))

    if not args.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply to rebase.")
        return 0

    os.makedirs(args.backup_dir, exist_ok=True)
    for p, text, _n in hits:
        rel = os.path.relpath(p, CODE_DIR)
        bak = os.path.join(args.backup_dir, rel)
        os.makedirs(os.path.dirname(bak), exist_ok=True)
        shutil.copy2(p, bak)
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(OLD_ROOT.sub(new_root, text))
    print("\nrebased %d files; originals copied to %s" % (len(hits), args.backup_dir))
    print("run with --verify to confirm (expect 0).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
