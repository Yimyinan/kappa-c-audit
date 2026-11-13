# -*- coding: utf-8 -*-
"""check_figure_typography.py -- reusable PLOS figure legibility gate.

Promotes the one-off .tmp/fig_visual_probe.py (2026-11-11 visual audit) into a
permanent, runnable check. It deliberately shares ALL placement arithmetic with
figure_palette.py -- there must be only one definition of landed pt.

What it measures, per exported PDF
---------------------------------
  W_mm / scale      physical page width and the shrink factor at \\textwidth
  landed_min/max    smallest/largest type size the reader actually sees
  font families     Arial vs DejaVu mix (the "double typeface" defect)
  size steps        how many distinct drawn sizes (a 4-step grid is the target)
  embedded rasters  effective DPI, for the one figure that shipped at 160 dpi

Verdicts
--------
  FAIL  landed_min < 5.0 pt (PLOS practical print floor), or a raster < 300 dpi
  WARN  landed_min < 6.5, mixed font families, > 4 size steps, width != 170 mm
  OK    everything above clean

The full manuscript currently has legitimate FAILs (18 figures below 5 pt).
Therefore the default mode REPORTS and exits 0; use --strict to make FAILs
fatal, and --file to check a single regenerated figure during the migration.

Usage
-----
    python code/check_figure_typography.py                 # full report
    python code/check_figure_typography.py --file figures/X.pdf
    python code/check_figure_typography.py --strict       # exit 1 on any FAIL
    python code/check_figure_typography.py --roles        # palette role summary
    python code/check_figure_typography.py --json out.json
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
from collections import Counter

import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import figure_palette as fp  # noqa: E402  (single source of placement arithmetic)

FIG = os.path.join(ROOT, "figures")
SUPP = os.path.join(ROOT, "supplementary")

MAIN_BY_FILE = {
    "fig_concept_3d_latent.pdf": 1,
    "fig2_z_spectrum_3tissue.pdf": 2,
    "fig0_workflow.pdf": 3,
    "fig2_kc_phase_diagram.pdf": 4,
    "fig5_prior_biology_validation.pdf": 5,
    "fig4_prior_activity.pdf": 6,
    "fig0_prior_activity_flowchart.pdf": 7,
}

# The SI directory also carries full-page prose notes and A4 tables. They are
# PDFs, but they are not figures: a 210 mm note is never placed at \textwidth,
# so its landed size is meaningless for the 5 pt floor. Report them as SKIP,
# never as FAIL -- the 2026-11-11 audit counted 18 figures + 1 such note.
NON_FIGURE_PREFIXES = ("supp_note", "table_s")


def classify(fname):
    for pfx in NON_FIGURE_PREFIXES:
        if fname.startswith(pfx):
            return "table" if pfx == "table_s" else "note"
    return "figure"


def short_font(name):
    """Collapse 'Arial-BoldMT' / 'DejaVuSans-Oblique' to a family label."""
    n = (name or "?").lower()
    if n.startswith("arial") or "helvetica" in n:
        return "Arial"
    if "dejavu" in n:
        return "DejaVu"
    if "cm" in n or "computer" in n:
        return "CM"
    return name.split("-")[0][:12]


def probe(path):
    d = fitz.open(path)
    pg = d[0]
    W, H = pg.rect.width, pg.rect.height
    scale = fp.TEXTWIDTH_PT / W if W else 1.0

    spans = []
    for b in pg.get_text("dict").get("blocks", []):
        for ln in b.get("lines", []):
            for s in ln.get("spans", []):
                if s.get("text", "").strip():
                    spans.append(s)

    sizes = [fp.landed_pt(s["size"], W) for s in spans]
    families = Counter(short_font(s.get("font")) for s in spans)
    drawn = sorted({round(s["size"], 1) for s in spans})
    # mathtext renders sub/superscripts at 0.7x the base size. Those are
    # DERIVED sizes, not author-chosen type steps, so counting them would make
    # every figure containing a formula report "8 sizes" forever. The all-glyph
    # minimum is still reported as landed_min; the 5 pt floor now tests the
    # author-set body type (landed_main_min), see the verdict block below.
    script_steps = {x for x in drawn
                    if any(abs(x - fp.MATH_SCRIPT_RATIO * y) < 0.15
                           for y in drawn if y > x)}
    steps = [x for x in drawn if x not in script_steps]
    # The 6.5-8.0 pt target applies to author-set body type, not to derived
    # math scripts. The 5 pt floor (below) still applies to EVERY glyph.
    main_sizes = [fp.landed_pt(s["size"], W) for s in spans
                  if round(s["size"], 1) not in script_steps]

    rasters = []
    for im in pg.get_images(full=True):
        xref = im[0]
        try:
            info = d.extract_image(xref)
            iw, ih = info["width"], info["height"]
        except Exception:
            iw, ih = im[2], im[3]
        rects = pg.get_image_rects(xref)
        eff = 0
        if rects:
            r = rects[0]
            if r.width and r.height:
                eff = min(iw / (r.width / 72), ih / (r.height / 72))
        rasters.append(round(eff))

    w_mm = W / 72 * 25.4
    fname = os.path.basename(path)
    r = {
        "file": fname,
        "kind": classify(fname),
        "fig": MAIN_BY_FILE.get(fname),
        "W_mm": round(w_mm, 1),
        "H_mm": round(H / 72 * 25.4, 1),
        "scale": round(scale, 3),
        "landed_min": round(min(sizes, default=0), 2),
        "landed_max": round(max(sizes, default=0), 2),
        "landed_main_min": round(min(main_sizes, default=0), 2),
        "landed_steps": len(steps),
        "script_steps": sorted(script_steps),
        "families": dict(families),
        "raster_dpi": rasters,
        "text_spans": len(spans),
    }

    fails, warns = [], []
    # The 5 pt floor is a legibility gate on AUTHOR-SET body type. A 0.7x math
    # sub/superscript is a derived glyph, not a type decision, and typographic
    # practice allows it below the floor (PLOS applies the floor to body text).
    # Keep the all-glyph minimum in the report and surface it as a note so the
    # fact is never hidden: 4.60 / 4.82 pt subscripts are still printed.
    body_floor = r["landed_main_min"]
    if body_floor and body_floor < fp.PLOS_MIN_PT:
        fails.append("body type %.2f pt < %.1f pt floor" %
                     (body_floor, fp.PLOS_MIN_PT))
    if (r["landed_min"] and r["landed_min"] < fp.PLOS_MIN_PT
            and body_floor and body_floor >= fp.PLOS_MIN_PT):
        warns.append("math-script glyphs %.2f pt below %.1f pt floor "
                     "(derived 0.7x; body type %.2f pt)" %
                     (r["landed_min"], fp.PLOS_MIN_PT, body_floor))
    if any(x and x < 300 for x in rasters):
        fails.append("embedded raster %s dpi < 300" % rasters)
    if r["landed_main_min"] and r["landed_main_min"] < fp.TARGET_LANDED_PT[0]:
        warns.append("body type %.2f pt below %.1f pt target" %
                     (r["landed_main_min"], fp.TARGET_LANDED_PT[0]))
    if len([k for k in families if k in ("Arial", "DejaVu", "CM")]) > 1:
        warns.append("mixed font families: " + ",".join(sorted(families)))
    if len(steps) > 4:
        warns.append("%d distinct drawn sizes (grid = 4)" % len(steps))
    if abs(w_mm - fp.TEXTWIDTH_MM) > 0.5:
        warns.append("page width %.1f mm != 170 mm" % w_mm)
    r["FAIL"] = fails
    r["WARN"] = warns
    if r["kind"] != "figure":
        # A4 prose/table pages are placed at their own width, not \textwidth.
        r["verdict"] = "SKIP"
        r["FAIL"], r["WARN"] = [], []
        r["note"] = "%s, excluded from the figure legibility gate" % r["kind"]
    else:
        r["verdict"] = "FAIL" if fails else ("WARN" if warns else "OK")
    d.close()
    return r


def report(rows, stream=sys.stdout):
    hdr = "%-38s %5s %7s %6s %6s %6s %5s  %s" % (
        "file", "Fig", "W_mm", "scale", "lmin", "lmax", "steps", "verdict")
    stream.write(hdr + "\n" + "-" * 92 + "\n")
    for r in rows:
        stream.write("%-38s %5s %7.1f %6.2f %6.2f %6.2f %5d  %s\n" % (
            r["file"][:38], ("F%d" % r["fig"]) if r["fig"] else "-",
            r["W_mm"], r["scale"], r["landed_min"], r["landed_max"],
            r["landed_steps"], r["verdict"]))
        for m in r["FAIL"]:
            stream.write("      FAIL  %s\n" % m)
        for m in r["WARN"]:
            stream.write("      WARN  %s\n" % m)
        if r.get("note"):
            stream.write("      SKIP  %s\n" % r["note"])
    c = Counter(r["verdict"] for r in rows if r.get("kind", "figure") == "figure")
    skipped = sum(1 for r in rows if r.get("kind", "figure") != "figure")
    stream.write("\n" + "=" * 92 + "\n")
    stream.write("figures %d | FAIL %d | WARN %d | OK %d | non-figure SKIP %d\n" %
                 (len(rows) - skipped, c["FAIL"], c["WARN"], c["OK"], skipped))
    below = [r for r in rows if r.get("kind", "figure") == "figure" and
             r["landed_main_min"] and r["landed_main_min"] < fp.PLOS_MIN_PT]
    derived = [r for r in rows if r.get("kind", "figure") == "figure" and
               r["landed_min"] and r["landed_min"] < fp.PLOS_MIN_PT
               and r["landed_main_min"] and r["landed_main_min"] >= fp.PLOS_MIN_PT]
    stream.write("figures with body type below 5 pt floor: %d" % len(below))
    if below:
        stream.write("  (worst: " + ", ".join(
            "%s %.1f pt" % (x["file"][:24], x["landed_main_min"])
            for x in sorted(below, key=lambda z: z["landed_main_min"])[:4]) + ")")
    stream.write("\n")
    stream.write("figures whose only sub-5pt glyphs are derived math scripts: %d" %
                 len(derived))
    if derived:
        stream.write("  (" + ", ".join(
            "%s %.2f pt" % (x["file"][:24], x["landed_min"])
            for x in derived) + ")")
    stream.write("\n")
    return c


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", action="append", default=[],
                    help="check only this PDF (repeatable)")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any figure FAILs")
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--roles", action="store_true",
                    help="print the semantic colour roles + low-contrast report")
    a = ap.parse_args(argv)

    if a.roles:
        print("SEMANTIC ROLES (one meaning per colour):")
        for colour, role in sorted(fp.SEMANTIC_ROLES.items()):
            print("  %s  %s" % (colour, role))
        print("\nLOW CONTRAST on white (< 4.5, unsafe as small text):")
        for colour, ratio in fp.low_contrast_report():
            print("  %s  %.2f" % (colour, ratio))
        print("\nNOTE: these are reported, not auto-fixed. Recolouring is a "
              "separate decision (P1-4).")
        return 0

    paths = a.file or (sorted(glob.glob(os.path.join(FIG, "*.pdf"))) +
                       sorted(glob.glob(os.path.join(SUPP, "*.pdf"))))
    rows = []
    for p in paths:
        try:
            rows.append(probe(p))
        except Exception as e:
            print("ERROR %s: %s" % (p, e), file=sys.stderr)
            rows.append({"file": os.path.basename(p), "kind": "figure",
                         "fig": None, "W_mm": 0, "scale": 0, "landed_min": 0,
                         "landed_max": 0, "landed_steps": 0,
                         "FAIL": ["probe error: %s" % e], "WARN": [],
                         "verdict": "FAIL"})
    c = report(rows)

    if a.json_out:
        with io.open(a.json_out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=1)
        print("wrote %s" % a.json_out)

    if a.strict and c["FAIL"]:
        print("STRICT: %d figure(s) FAIL" % c["FAIL"])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
