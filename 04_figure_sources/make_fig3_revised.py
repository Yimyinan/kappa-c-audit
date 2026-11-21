# -*- coding: utf-8 -*-
"""Redraw Fig. 3 from the exact per-tissue ranges reported in V37 Table 1.

The individual 33 configuration values were not supplied as a raw table.
Therefore, this figure intentionally uses the documented min-max ranges rather
than digitizing or inventing individual observations. Per-configuration values
remain referenced to S1 Table in the manuscript.
"""
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'fig3_revised_output'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))
import figure_palette as fp
fp.apply_sample_style()

# Exact ranges transcribed from V37 Table 1 (kappa); C = (1 - kappa) * 100.
DATA = [
    dict(key='hemo', label='Hematopoietic', n=22, kmin=0.950, kmax=1.003),
    dict(key='neural', label='Neural cortex', n=10, kmin=0.873, kmax=0.957),
    dict(key='lung', label='Lung fibrosis', n=1, kmin=0.893, kmax=0.893),
]
for d in DATA:
    d['cmin'] = (1.0 - d['kmax']) * 100.0
    d['cmax'] = (1.0 - d['kmin']) * 100.0

Y = [2, 1, 0]
COLORS = {d['key']: fp.tissue_color(d['key']) for d in DATA}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fp.figsize(height_mm=125.0))
fig.subplots_adjust(left=0.24, right=0.96, top=0.88, bottom=0.16, hspace=0.52)

# Panel A: global compression range. Values are min-max ranges, not CIs.
for y, d in zip(Y, DATA):
    c = COLORS[d['key']]
    if d['cmin'] == d['cmax']:
        ax1.plot(d['cmin'], y, marker='o', ms=5.2, color=c,
                 markeredgecolor='white', markeredgewidth=0.5, zorder=4)
    else:
        ax1.plot([d['cmin'], d['cmax']], [y, y], color=c,
                 lw=fp.LW_EMPH, solid_capstyle='round', zorder=2)
        ax1.scatter([d['cmin'], d['cmax']], [y, y], s=24, color=c,
                    edgecolors='white', linewidths=0.5, zorder=3)
    label = (f"{d['cmin']:.1f} to {d['cmax']:.1f}%" if d['n'] > 1
             else f"{d['cmin']:.1f}%")
    ax1.text(14.2, y, label, ha='left', va='center',
             fontsize=fp.FS_NOTE, color=fp.INK)

ax1.set_yticks(Y)
ax1.set_yticklabels([f"{d['label']}  (n={d['n']})" for d in DATA])
ax1.set_ylim(-0.55, 2.85)
ax1.set_xlim(-2.0, 23.5)
ax1.set_xlabel('Global compression, C (%)')
ax1.set_title('(a) Observed compression range', loc='left', pad=7)
ax1.axvline(0, color=fp.GRAY, lw=fp.LW_REF, zorder=0)
ax1.text(0.98, 0.94, 'Deep-compression threshold: C > 50% (off scale)',
         transform=ax1.transAxes, ha='right', va='top', fontsize=fp.FS_NOTE,
         color=fp.MUTED)
fp.style_ax(ax1)
ax1.tick_params(axis='y', length=0, pad=5)

# Panel B: the exact kappa ranges, with the published cross-family null band.
null_lo, null_hi = 0.937, 1.064
ax2.axvspan(null_lo, null_hi, color=fp.BAND_CROSS, zorder=0)
ax2.axvline(0.5, color=fp.RED, lw=fp.LW_REF, ls='--', zorder=1)
for y, d in zip(Y, DATA):
    c = COLORS[d['key']]
    if d['kmin'] == d['kmax']:
        ax2.plot(d['kmin'], y, marker='o', ms=5.2, color=c,
                 markeredgecolor='white', markeredgewidth=0.5, zorder=4)
    else:
        ax2.plot([d['kmin'], d['kmax']], [y, y], color=c,
                 lw=fp.LW_EMPH, solid_capstyle='round', zorder=2)
        ax2.scatter([d['kmin'], d['kmax']], [y, y], s=24, color=c,
                    edgecolors='white', linewidths=0.5, zorder=3)
    label = (f"{d['kmin']:.3f}–{d['kmax']:.3f}" if d['n'] > 1
             else f"{d['kmin']:.3f}")
    ax2.text(1.075, y, label, ha='left', va='center',
             fontsize=fp.FS_NOTE, color=fp.INK)

ax2.set_yticks(Y)
ax2.set_yticklabels([f"{d['label']}  (n={d['n']})" for d in DATA])
ax2.set_ylim(-0.55, 2.70)
ax2.set_xlim(0.43, 1.19)
ax2.set_xlabel(r'Aggregate predictability ratio, $\kappa$')
ax2.set_title('(b) Predictability ratio and null band', loc='left', pad=7)
ax2.text(0.505, -0.37, r'Deep compression: $\kappa<0.5$',
         ha='left', va='center', fontsize=fp.FS_NOTE, color=fp.RED)
ax2.legend(handles=[
    Patch(facecolor=fp.BAND_CROSS, edgecolor='none', label='Cross-family G-VAE null band'),
    Line2D([], [], color=fp.RED, ls='--', lw=fp.LW_REF, label=r'Deep-compression boundary ($\kappa=0.5$)')
], loc='upper right', bbox_to_anchor=(0.98, -0.34), ncol=2,
   frameon=False, fontsize=fp.FS_NOTE, handlelength=1.6, columnspacing=1.2)
fp.style_ax(ax2)
ax2.tick_params(axis='y', length=0, pad=5)

fig.suptitle('No audited configuration reaches deep decoder-readout compression',
             x=0.24, y=0.985, ha='left', fontsize=fp.FS_TITLE, fontweight='bold')
fp.assert_design_width(fig)

stem = OUT / 'Fig3_prior_injection_audit'
fig.savefig(stem.with_suffix('.pdf'))
fig.savefig(stem.with_name(stem.name + '_300dpi.png'), dpi=300)
fig.savefig(stem.with_name(stem.name + '_600dpi.tif'), dpi=600,
            pil_kwargs={'compression':'tiff_lzw'})
fig.savefig(stem.with_suffix('.svg'))
plt.close(fig)

caption = """Fig 3. No audited prior-injection configuration reaches deep decoder-readout compression. (a) Minimum-to-maximum ranges of global compression, C, across the 33 audited configurations: 22 hematopoietic, 10 neural-cortex and one lung-fibrosis configuration. The ranges are summaries of the reported configuration values, not confidence intervals. (b) Corresponding ranges of the aggregate predictability ratio, κ, shown against the cross-family G-VAE vanilla-equivalence band (0.937–1.064) and the deep-compression boundary (κ < 0.5). No configuration reaches the deep-compression boundary (0/33). Individual configuration values and the family-matched UNAGI recalibration are reported in S1 and S7 Tables, respectively. The historical directional-collapse panel is not included here because the non-hematopoietic values were not reverified and are excluded from the evidence base."""
(OUT/'Fig3_caption.txt').write_text(caption, encoding='utf-8')

readme = """Fig. 3 revision notes
=====================

Source basis
------------
The figure uses only exact tissue-level min/max kappa values and configuration
counts printed in V37 Table 1. Global compression is calculated as
C = (1 - kappa) * 100. The values are:
  Hematopoietic, n=22: kappa 0.950–1.003
  Neural cortex, n=10: kappa 0.873–0.957
  Lung fibrosis, n=1: kappa 0.893
The cross-family null band 0.937–1.064 and threshold kappa=0.5 are also
reported in the manuscript / Fig. 2 workflow script.

No unreported per-configuration values have been digitized or invented. This
revision therefore replaces the original per-run scatter with honest range
summaries. Per-configuration values remain in S1 Table. Family-matched
recalibration remains referenced to S7 Table and is not drawn here because
its exact bounds were not supplied as a data table.

Panel C of the original Fig. 3 (legacy directional-collapse statistic) is
omitted from the main figure because its neural-cortex and lung-fibrosis
values were not reverified, as stated in the V37 caption. It may be retained
as historical context in Supporting Information with an explicit
unverified/excluded label, but should not be part of the evidence-bearing
main figure.

Outputs
-------
PDF, SVG, 300-dpi PNG and 600-dpi LZW TIFF.
"""
(OUT/'README_Fig3_revision.md').write_text(readme, encoding='utf-8')
print('Wrote', *[p.name for p in OUT.iterdir()])
