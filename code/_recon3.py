# -*- coding: utf-8 -*-
"""Recon for the three follow-ups:
  (2) where does main.tex's family-matched band [0.971, 1.029] come from?
  (3) which scripts carry the superseded [0.78, 1.23] band, and what kind of script is each?
Also locates the band inside the RELEASED toolkit, which is a published artefact.
"""
import io, os, re, glob

W = r"C:/Users/12901/WorkBuddy/2026-09-03-18-47-34"
P = os.path.join(W, "paper1", "paper1_kc_diagnostic")

print("=== (2) provenance hunt for 0.971 / 1.029 ===")
pat = re.compile(r"0\.971|1\.029")
for p in glob.glob(os.path.join(W, "**", "*"), recursive=True):
    if not os.path.isfile(p) or "/.venvs/" in p or "/node_modules/" in p:
        continue
    if os.path.splitext(p)[1].lower() not in (".json", ".py", ".md", ".tex", ".txt"):
        continue
    try:
        t = io.open(p, encoding="utf-8", errors="ignore").read()
    except Exception:
        continue
    if pat.search(t):
        rel = os.path.relpath(p, W)
        print("   %s" % rel)

print()
print("=== (3) scripts carrying the superseded band ===")
CLASS = {"generator": ("make_fig", "make_tab", "generate_", "build_"),
         "toolkit":   ("positive_control",),
         "recalc":    ("recalc", "recompute_")}
for p in sorted(glob.glob(os.path.join(P, "**", "*.py"), recursive=True)):
    if any(x in p.replace("\\", "/") for x in ("/releases/", "/snapshots/", "_archive", "/archive/", "/build/")):
        continue
    try:
        t = io.open(p, encoding="utf-8", errors="ignore").read()
    except Exception:
        continue
    if not re.search(r"0\.78|1\.23", t):
        continue
    base = os.path.basename(p)
    kind = next((k for k, pre in CLASS.items() if base.startswith(pre)), "other")
    hits = len(re.findall(r"0\.78|1\.23", t))
    print("   %-8s %-46s %d hits" % (kind, os.path.relpath(p, P), hits))

print()
print("=== does the RELEASED toolkit hardcode the band? ===")
for p in glob.glob(os.path.join(P, "kappa_c_audit", "**", "*.py"), recursive=True):
    t = io.open(p, encoding="utf-8", errors="ignore").read()
    m = re.findall(r"0\.78|1\.23|0\.937|1\.064", t)
    if m:
        print("   %s -> %s" % (os.path.relpath(p, P), sorted(set(m))))
print()
print("=== where is main.tex's family-matched sentence? ===")
mt = io.open(os.path.join(P, "main.tex"), encoding="utf-8-sig").read()
for m in re.finditer(r"[^.]{0,200}0\.971[^.]{0,200}\.", mt):
    print("   ...%s..." % re.sub(r"\s+", " ", m.group(0)))
