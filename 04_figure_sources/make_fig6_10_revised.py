# -*- coding: utf-8 -*-
"""Rebuild V37 main-text Figures 6-10 from the supplied source JSON files.

All outputs are drawn at 170 mm placement width. Each figure is exported as
PDF/SVG (vector), PNG 300 dpi (review), and TIFF 600 dpi (high-resolution).
Numbers are loaded from the supplied JSON where available; the support-bound
curves in Fig. 7 use the constants documented in the supplied make_fig_paper3.py.
"""
from __future__ import annotations
import json, math, os, shutil, zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'fig6_8_10_prior_delivery'
DATA9 = ROOT / 'fig9_ipf_crossarch'
OUT = ROOT / 'fig6_10_revised_output'
OUT.mkdir(parents=True, exist_ok=True)
sys_path = ROOT / 'scripts'
import sys
sys.path.insert(0, str(sys_path))
import figure_palette as fp
# Raise the note/legend floor to the PLOS 8 pt figure-text requirement.
fp.FS_NOTE = 8.0
fp.apply_sample_style()
# Use Arimo, an Arial-metric-compatible font available in this container.
# Arial itself is not installed in this rendering environment.
plt.rcParams.update({'font.sans-serif': ['Arimo', 'DejaVu Sans'],
                      'mathtext.rm': 'Arimo',
                      'mathtext.it': 'Arimo:italic',
                      'mathtext.bf': 'Arimo:bold'})
fp.assert_style()

# Main paper palette roles: prior, control, data-fitted directions, references.
PURPLE = fp.ARM_PRIOR
PURPLE_LIGHT = fp.ARM_PRIOR_ALT
BLUE = fp.C_HEMA
ORANGE = fp.C_NEURAL
GREEN = fp.C_LUNG
INK = fp.INK
MUTED = fp.MUTED
GREY = fp.GREY
GRAY = fp.GRAY
BAND = fp.FAIL_FILL
THRESH = fp.RED


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def save_figure(fig, stem: str):
    fp.assert_design_width(fig)
    p = OUT / stem
    fig.savefig(p.with_suffix('.pdf'), format='pdf', facecolor='white')
    fig.savefig(p.with_suffix('.svg'), format='svg', facecolor='white')
    fig.savefig(OUT / f'{stem}_300dpi.png', dpi=300, facecolor='white')
    # Matplotlib may include an alpha channel even on a white canvas. PLOS
    # submission TIFFs are flattened RGB, so composite explicitly on white.
    tmp_tif = OUT / f'.{stem}_600dpi_rgba.tif'
    fig.savefig(tmp_tif, dpi=600, format='tiff', facecolor='white', transparent=False,
                pil_kwargs={'compression': 'tiff_lzw'})
    from PIL import Image
    with Image.open(tmp_tif) as im:
        rgb = im.convert('RGB')
        rgb.save(OUT / f'{stem}_600dpi.tif', format='TIFF', compression='tiff_lzw', dpi=(600, 600))
    tmp_tif.unlink(missing_ok=True)
    plt.close(fig)


def style(ax):
    fp.style_ax(ax)
    ax.tick_params(labelsize=fp.FS_AXIS)


def panel_title(ax, s, title):
    ax.set_title(f'({s})  {title}', loc='left', pad=11,
                 fontsize=fp.FS_TITLE, fontweight='bold', color=INK)


def mean_sd(values):
    a = np.asarray(values, dtype=float)
    return float(np.mean(a)), float(np.std(a, ddof=1))


# ------------------------------ Fig. 6 ------------------------------------
def fig6():
    """Paired seed trajectories for dose and duration, from per-gene robust JSON."""
    source = {
        5: 'support__pergene_robust_5x.json',
        20: 'dose20__pergene_robust.json',
        40: 'dose40__pergene_robust.json',
    }
    dose = {}
    for level, fn in source.items():
        d = load_json(DATA / fn)
        dose[level] = {int(seed): float(rec['decile_excess_mean'])
                       for seed, rec in d.items()}
    d45 = load_json(DATA / 'dose20_e45__pergene_robust.json')
    time = {15: dose[20],
            45: {int(seed): float(rec['decile_excess_mean'])
                 for seed, rec in d45.items()}}
    seeds = [42, 43, 44]
    assert all(set(v) == set(seeds) for v in dose.values())
    assert set(time[45]) == set(seeds)

    fig, axes = plt.subplots(1, 2, figsize=fp.figsize(height_mm=82),
                             gridspec_kw={'width_ratios':[1.15, 1]})
    # Distinguish the three paired training seeds by marker and line style.
    seed_styles = {42: ('-', 'o'), 43: ('--', 's'), 44: (':', '^')}
    seedline = '#888888'
    # Panel A: dose ladder; same-seed trajectories are paired across dose.
    ax = axes[0]
    x = np.array([5, 20, 40], dtype=float)
    for s in seeds:
        y = [dose[int(v)][s] for v in x]
        ls, marker = seed_styles[s]
        ax.plot(x, y, color=seedline, lw=fp.LW_DATA, alpha=.9, linestyle=ls,
                marker=marker, ms=3.5, markerfacecolor='white',
                markeredgecolor=GREY, markeredgewidth=fp.LW_EDGE, zorder=2)
    means = np.array([mean_sd([dose[int(v)][s] for s in seeds])[0] for v in x])
    sds = np.array([mean_sd([dose[int(v)][s] for s in seeds])[1] for v in x])
    ax.errorbar(x, means, yerr=sds, fmt='D-', color=PURPLE,
                ecolor=PURPLE, lw=fp.LW_EMPH, elinewidth=fp.LW_ERR,
                capsize=fp.CAPSIZE, ms=4.6, zorder=4,
                label='Mean ± SD across seeds')
    ax.set_xscale('log')
    ax.set_xticks(x, [r'$5\times$', r'$20\times$', r'$40\times$'])
    ax.set_xlim(4, 48)
    ax.set_xlabel('Prior-weight dose (support : rest)')
    ax.set_ylabel(r'Matched per-gene effect $d_m$')
    ax.set_ylim(-0.0025, 0.022)
    panel_title(ax, 'A', 'Dose')
    style(ax)

    # Panel B: duration ladder at fixed 20x dose; paired seed lines.
    ax = axes[1]
    x = np.array([15, 45], dtype=float)
    for s in seeds:
        y = [time[int(v)][s] for v in x]
        ls, marker = seed_styles[s]
        ax.plot(x, y, color=seedline, lw=fp.LW_DATA, alpha=.9, linestyle=ls,
                marker=marker, ms=3.5, markerfacecolor='white',
                markeredgecolor=GREY, markeredgewidth=fp.LW_EDGE, zorder=2)
    means = np.array([mean_sd([time[int(v)][s] for s in seeds])[0] for v in x])
    sds = np.array([mean_sd([time[int(v)][s] for s in seeds])[1] for v in x])
    ax.errorbar(x, means, yerr=sds, fmt='D-', color=PURPLE,
                ecolor=PURPLE, lw=fp.LW_EMPH, elinewidth=fp.LW_ERR,
                capsize=fp.CAPSIZE, ms=4.6, zorder=4)
    ax.set_xticks(x, ['15', '45'])
    ax.set_xlim(10, 50)
    ax.set_xlabel(r'Weighted-phase duration at $20\times$ (epochs)')
    ax.set_ylabel(r'Matched per-gene effect $d_m$')
    ax.set_ylim(-0.0025, 0.022)
    panel_title(ax, 'B', 'Training duration')
    seed_handles = [Line2D([], [], color=seedline, lw=fp.LW_DATA,
                           linestyle=seed_styles[s][0], marker=seed_styles[s][1],
                           markerfacecolor='white', markeredgecolor=GREY,
                           markeredgewidth=fp.LW_EDGE, ms=3.5, label=f'Seed {s}')
                    for s in seeds]
    ax.legend(handles=seed_handles, loc='lower right', ncol=3,
              fontsize=fp.FS_NOTE, frameon=False, handlelength=1.1,
              columnspacing=0.9, borderpad=0.1, labelspacing=0.2)
    style(ax)

    fig.text(.5, .01, 'Thin lines identify paired seeds; diamonds and whiskers show mean ± SD (n = 3 seeds).',
             ha='center', va='bottom', fontsize=fp.FS_NOTE, color=MUTED)
    fig.subplots_adjust(left=.09, right=.985, bottom=.18, top=.87, wspace=.35)
    save_figure(fig, 'Fig6_dose_time_plateau')

    audit = {'dose': {str(k): {str(s): v for s,v in vals.items()} for k,vals in dose.items()},
             'time': {str(k): {str(s): v for s,v in vals.items()} for k,vals in time.items()},
             'summary': {str(k): {'mean': mean_sd(list(v.values()))[0],
                                  'sd': mean_sd(list(v.values()))[1]} for k,v in dose.items()},
             'time_summary': {str(k): {'mean': mean_sd(list(v.values()))[0],
                                       'sd': mean_sd(list(v.values()))[1]} for k,v in time.items()}}
    (OUT/'Fig6_data_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return audit


# ------------------------------ Fig. 7 ------------------------------------
CONV = {
    'varying': dict(N=17328, S=2322.3, m=0.1340, ns=191,
                    d_local=+0.00502, d_global=+0.00045,
                    c_local=-0.00955, c_global=-0.00212),
    'all': dict(N=20527, S=5521.3, m=0.2690, ns=203,
                d_local=+0.00861, d_global=+0.00017,
                c_local=+0.00188, c_global=-0.00250),
}


def dkappa(c, n_s, d_local, d_global):
    return (n_s*d_local + (c['N']-n_s)*d_global)/c['S']


def fig7():
    """Support-dependent bound; observed points use the manuscript's Δκ = κ - 1."""
    ladder = load_json(DATA/'dose_ladder.json')
    levels = sorted(float(k) for k in ladder['means'])
    kcw = []
    for lv in levels:
        vals = ladder['means'][str(lv)]['kcw']
        for i, v in enumerate(vals):
            # Sign is deliberately κ - 1, consistent with the plotted ordinate.
            kcw.append({'dose': lv, 'seed_index': i+1,
                        'kappa': float(v), 'delta_kappa': float(v)-1.0})

    fig, axes = plt.subplots(1, 2, figsize=fp.figsize(height_mm=94),
                             gridspec_kw={'width_ratios':[1.55, 1]})
    ax = axes[0]
    for key, conv in CONV.items():
        primary = key == 'varying'
        f = np.logspace(np.log10(2/conv['N']), 0, 500)
        ns = f*conv['N']
        alpha = 1.0 if primary else .48
        lw = fp.LW_DATA if primary else fp.LW_REF
        # Upper attainable bound if every supported gene reached R2=1.
        ax.plot(f, dkappa(conv, ns, 1-conv['m'], 0), color=GRAY,
                ls=':' if primary else (0,(1,1)), lw=lw, alpha=alpha,
                label='Arithmetic ceiling (varying genes)' if primary else None)
        # Observed local target-gene gain, applied only to supported genes.
        ax.plot(f, dkappa(conv, ns, conv['d_local'], 0), color=PURPLE,
                ls='-' if primary else '--', lw=lw, alpha=alpha,
                label='Measured local gain (supported genes)' if primary else None)
        # Add the measured global cost of the matched permuted control.
        ax.plot(f, dkappa(conv, ns, conv['d_local'], conv['c_global']),
                color=BLUE, ls='--' if primary else (0,(2,1)),
                lw=lw, alpha=alpha,
                label='Local gain + control global cost' if primary else None)

    # Whole-transcriptome application of measured local gain (all-gene convention).
    c = CONV['all']
    f_all = np.logspace(np.log10(2/c['N']), 0, 500)
    y_all = dkappa(c, f_all*c['N'], c['d_local'], c['d_local'])
    ax.plot(f_all, y_all, color=INK, lw=fp.LW_REF, ls='-.',
            label='Measured local gain applied to every gene')
    ax.axhline(0, color=INK, lw=fp.LW_REF)
    # Nine trained arms at the prior-support fraction; correct sign is kappa - 1.
    x_support = CONV['varying']['ns']/CONV['varying']['N']
    offsets = np.array([.985, .992, .997, 1.003, 1.008, 1.014, 1.020, 1.026, 1.032])
    ax.scatter(x_support*offsets, [r['delta_kappa'] for r in kcw],
               s=19, color=PURPLE, edgecolor='white', linewidth=fp.LW_EDGE,
               zorder=5)
    ax.axvline(x_support, color=PURPLE, ls=':', lw=fp.LW_REF, alpha=.7)
    ax.set_xscale('log')
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(-.20, .20)
    ax.set_xlabel(r'Prior support fraction $n_s/N$')
    ax.set_ylabel(r'Change in aggregate readout, $\Delta\kappa=\kappa-1$')
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.legend(loc='lower right', fontsize=fp.FS_NOTE, handlelength=1.8,
              labelspacing=.35, borderpad=.2)
    panel_title(ax, 'A', 'Support-dependent resolution')
    style(ax)

    # Panel B: local and global contributions, in percentage points of Δκ.
    ax = axes[1]
    c = CONV['varying']
    labels = ['Real prior', 'Weight-matched\nrandom control']
    local = [c['ns']*c['d_local']/c['S']*100,
             c['ns']*c['c_local']/c['S']*100]
    global_ = [(c['N']-c['ns'])*c['d_global']/c['S']*100,
               (c['N']-c['ns'])*c['c_global']/c['S']*100]
    x = np.arange(2); w=.33
    b1 = ax.bar(x-w/2, local, w, color=PURPLE, edgecolor=fp.EDGE,
                linewidth=fp.LW_EDGE, label='Local: named genes')
    b2 = ax.bar(x+w/2, global_, w, color=BLUE, edgecolor=fp.EDGE,
                linewidth=fp.LW_EDGE, label='Global: all other genes')
    ax.axhline(0, color=INK, lw=fp.LW_REF)
    for bars in (b1,b2):
        for r in bars:
            h=r.get_height()
            ax.text(r.get_x()+r.get_width()/2,
                    h + (.055 if h>=0 else -.055), f'{h:+.2f}',
                    ha='center', va='bottom' if h>=0 else 'top',
                    fontsize=fp.FS_NOTE, color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylabel('Contribution to $\\Delta\\kappa$\n(percentage points)')
    ax.set_ylim(-1.95,.62)
    ax.legend(loc='lower left', fontsize=fp.FS_NOTE, handlelength=1.4,
              labelspacing=.35, borderpad=.2)
    # This title previously exceeded the panel's right safety margin and was
    # clipped in both the vector master and the Word-rendered manuscript.
    panel_title(ax, 'B', r'Local and global $\Delta\kappa$')
    style(ax)

    fig.subplots_adjust(left=.095, right=.975, bottom=.20, top=.85, wspace=.34)
    fig.text(.5, .025, 'Nine trained arms (three doses × three seeds); points are horizontally jittered around the true support (vertical line).',
             ha='center', va='bottom', fontsize=fp.FS_NOTE, color=MUTED)
    save_figure(fig, 'Fig7_support_resolution_bound')

    audit = {'observed_points': kcw, 'support_fraction_varying': x_support,
             'observed_delta_definition': 'delta_kappa = kappa - 1',
             'legacy_generator_conversion': '1 - kappa (sign inconsistent with y-axis; corrected)',
             'panel_b_local_percent': local, 'panel_b_global_percent': global_}
    (OUT/'Fig7_data_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return audit


# ------------------------------ Fig. 8 ------------------------------------
def fig8():
    d = load_json(DATA/'calib__s42__calibration.json')
    items = sorted(d.items(), key=lambda kv: int(kv[0][1:]))
    epochs = np.array([int(k[1:]) for k,_ in items], dtype=float)
    k_var = np.array([v['k_var'] for _,v in items], dtype=float)
    k_all = np.array([v['k_all'] for _,v in items], dtype=float)
    mean_var = np.array([v['mean_var'] for _,v in items], dtype=float)
    slope, intercept = np.polyfit(mean_var, k_var, 1)
    r2_at_band = (0.5-intercept)/slope

    fig, axes = plt.subplots(1,2,figsize=fp.figsize(height_mm=82),
                             gridspec_kw={'width_ratios':[1,1.05]})
    ax=axes[0]
    ax.plot(epochs,k_var,'o-',color=PURPLE,lw=fp.LW_EMPH,ms=4.2,
            label=r'$\kappa_{C(W)}$, varying genes')
    ax.plot(epochs,k_all,'s--',color=BLUE,lw=fp.LW_DATA,ms=3.5,
            label=r'$\kappa_{C(W)}$, all genes')
    ax.axhspan(.35,.5,color=BAND,alpha=.65,zorder=0)
    ax.axhline(.5,color=THRESH,lw=fp.LW_REF,ls='--')
    ax.set_xscale('log')
    ax.set_xticks(epochs,[str(int(v)) for v in epochs])
    ax.set_xlim(1.6,60); ax.set_ylim(.35,1.06)
    ax.set_xlabel('Training epochs')
    ax.set_ylabel(r'$\kappa_{C(W)}$ relative to 45-epoch arm')
    panel_title(ax, 'A', 'Training-budget sweep')
    style(ax)

    ax=axes[1]
    ax.scatter(mean_var,k_var,s=29,color=PURPLE,edgecolor='white',
               linewidth=fp.LW_EDGE,zorder=4)
    # Observed linear fit only over sampled data; dashed extension is explicitly extrapolative.
    xfit=np.linspace(float(mean_var.min()),float(mean_var.max()),100)
    ax.plot(xfit,slope*xfit+intercept,color=PURPLE,lw=fp.LW_DATA)
    xext=np.linspace(max(.055,float(r2_at_band)),float(mean_var.min()),60)
    ax.plot(xext,slope*xext+intercept,color=PURPLE,lw=fp.LW_DATA,ls='--')
    label_offsets = {2: (-14, -7), 5: (-18, 11), 10: (8, 12),
                     20: (-18, -16), 45: (8, 9)}
    for x,y,e in zip(mean_var,k_var,epochs):
        off = label_offsets[int(e)]
        ax.annotate(f'{int(e)} ep',(x,y),textcoords='offset points',xytext=off,
                    ha='right' if off[0] < 0 else 'left',fontsize=fp.FS_NOTE,color=PURPLE)
    ax.axhspan(.35,.5,color=BAND,alpha=.65,zorder=0)
    ax.axhline(.5,color=THRESH,lw=fp.LW_REF,ls='--')
    ax.axvline(r2_at_band,color=THRESH,lw=fp.LW_REF,ls=':')
    ax.annotate(f'Linear extrapolation\n$R^2$ at band ≈ {r2_at_band:.3f}',
                xy=(r2_at_band,.5),xytext=(r2_at_band+.012,.64),
                fontsize=fp.FS_NOTE,color=MUTED,
                arrowprops=dict(arrowstyle='-',lw=fp.LW_REF,color=MUTED))
    ax.set_xlim(min(.055,r2_at_band-.006),.15); ax.set_ylim(.35,1.06)
    ax.set_xlabel(r'Mean per-gene $R^2$ on varying genes')
    ax.set_ylabel(r'$\kappa_{C(W)}$')
    panel_title(ax, 'B', 'Readout versus reconstruction quality')
    style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5,.985),
               ncol=2, frameon=False, fontsize=fp.FS_NOTE, handlelength=1.6)
    fig.subplots_adjust(left=.095,right=.985,bottom=.16,top=.80,wspace=.36)
    fig.text(.5,.025,'Single-seed descriptive calibration (seed 42); the dashed segment in panel B is extrapolation, not an observed condition.',
             ha='center',va='bottom',fontsize=fp.FS_NOTE,color=MUTED)
    save_figure(fig,'Fig8_compression_band_calibration')

    audit={'seed':42,'epochs':epochs.tolist(),'kappa_varying':k_var.tolist(),
           'kappa_all':k_all.tolist(),'mean_r2_varying':mean_var.tolist(),
           'linear_slope':float(slope),'linear_intercept':float(intercept),
           'r2_at_kappa_0_5_extrapolated':float(r2_at_band),
           'note':'Single-seed descriptive calibration; extrapolation is not an observed condition.'}
    (OUT/'Fig8_data_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    return audit


# ------------------------------ Fig. 9 ------------------------------------
def fig9():
    scan=load_json(DATA9/'optionB_lambda_scan.json')
    co=load_json(DATA9/'celloracle_sensitivity.json')
    lams=[.1,1,10]
    bnC=[scan[f'lam{x:g}']['bn_prior_kc_s42']['C'] for x in lams]
    lnC=[scan[f'lam{x:g}']['ln_prior_kc_s42']['C'] for x in lams]
    bnK=[scan[f'lam{x:g}']['bn_prior_kc_s42']['kappa'] for x in lams]
    lnK=[scan[f'lam{x:g}']['ln_prior_kc_s42']['kappa'] for x in lams]
    co_rows=sorted(co['rows'],key=lambda r:r['kappa_original'])

    fig,axes=plt.subplots(2,2,figsize=fp.figsize(height_mm=128))
    a,b,c,d=axes.ravel()
    # A: global C scan; seed 42 only.
    a.plot(lams,bnC,'o-',color=PURPLE,lw=fp.LW_DATA,ms=4,
           label='BatchNorm decoder')
    a.plot(lams,lnC,'s-',color=BLUE,lw=fp.LW_DATA,ms=4,
           label='LayerNorm decoder')
    a.set_xscale('log'); a.set_xticks(lams,['0.1','1','10'])
    a.set_ylim(94.5,97.5); a.set_xlabel(r'Prior strength $\lambda$')
    a.set_ylabel('Global compression $C$ (%)')
    a.legend(fontsize=fp.FS_NOTE,loc='best',handlelength=1.5)
    panel_title(a, 'A', 'Global compression is flat')
    style(a)

    # B: kappa_gap scan. Log scale makes the target=2 visible and correctly placed.
    b.plot(lams,bnK,'o-',color=PURPLE,lw=fp.LW_DATA,ms=4,
           label='BatchNorm decoder')
    b.plot(lams,lnK,'s-',color=BLUE,lw=fp.LW_DATA,ms=4,
           label='LayerNorm decoder')
    b.axhline(2.0,color=THRESH,lw=fp.LW_REF,ls='--',label=r'Training target $\kappa_{gap}=2$')
    b.set_xscale('log'); b.set_yscale('log')
    b.set_xticks(lams,['0.1','1','10'])
    b.set_ylim(.005,3); b.set_xlabel(r'Prior strength $\lambda$')
    b.set_ylabel(r'$\kappa_{gap}$ (log scale)')
    b.legend(fontsize=fp.FS_NOTE,loc='center right',handlelength=1.5)
    panel_title(b, 'B', 'Gap ratio remains subcritical')
    style(b)

    # C: structural bound on raw-count IPF.
    Cgrid=np.linspace(0,100,301)
    trH,gapV=10590.0,9663.0
    ub=(1-Cgrid/100)*trH/gapV
    c.plot(Cgrid,ub,color=BLUE,lw=fp.LW_EMPH,label='Structural upper bound')
    c.axhline(2.0,color=THRESH,lw=fp.LW_REF,ls='--',label='Required target = 2')
    c.scatter([90],[.11],s=28,color=PURPLE,edgecolor='white',linewidth=fp.LW_EDGE,zorder=4)
    c.annotate('At $C=90\%$\nbound = 0.11',xy=(90,.11),xytext=(67,.55),
               fontsize=fp.FS_NOTE,color=MUTED,
               arrowprops=dict(arrowstyle='-',lw=fp.LW_REF,color=MUTED))
    c.text(.04,.06,'Vanilla top axis = 91.2% of tr(H)',transform=c.transAxes,
           fontsize=fp.FS_NOTE,color=MUTED)
    c.set_xlim(0,100); c.set_ylim(0,2.25)
    c.set_xlabel('Global compression $C$ (%)')
    c.set_ylabel(r'Upper bound on $\kappa_{gap}$')
    c.text(4,1.55,'Structural upper bound',ha='left',va='bottom',
           fontsize=fp.FS_NOTE,color=BLUE)
    c.text(98,2.05,'Target = 2',ha='right',va='bottom',
           fontsize=fp.FS_NOTE,color=THRESH)
    panel_title(c, 'C', 'Structural bound excludes target')
    style(c)

    # D: cross-architecture kappa_DK; logarithmic x makes the threshold distance legible.
    names=[r['tf'] for r in co_rows]
    vals=[r['kappa_original'] for r in co_rows]
    ypos=np.arange(len(names))
    d.barh(ypos,vals,left=.004,color=ORANGE,edgecolor=fp.EDGE,
           linewidth=fp.LW_EDGE,height=.62)
    d.scatter(vals,ypos,s=20,color=ORANGE,edgecolor='white',linewidth=fp.LW_EDGE,zorder=4)
    d.axvline(.5,color=THRESH,lw=fp.LW_REF,ls='--')
    d.set_xscale('log'); d.set_xlim(.004,.8)
    d.set_yticks(ypos,names)
    d.set_xlabel(r'$\kappa_{DK}$ (log scale)')
    panel_title(d, 'D', 'CellOracle-style TF priors')
    style(d)

    fig.subplots_adjust(left=.12,right=.985,bottom=.22,top=.90,wspace=.48,hspace=.46)
    fig.text(.5,.075,'Panel D reports the distinct Davis–Kahan ratio.\nAll eight TF priors are below 0.5; bounded reference-gap margin ≈ 24.8-fold.',
             ha='center',va='bottom',fontsize=fp.FS_NOTE,color=MUTED,linespacing=1.25)
    save_figure(fig,'Fig9_prior_activity_cross_architecture')
    audit={'lambda':[float(v) for v in lams],
           'bn_C':bnC,'ln_C':lnC,'bn_kappa_gap':bnK,'ln_kappa_gap':lnK,
           'target_kappa_gap':2.0,
           'celloracle':[{ 'tf':r['tf'],'kappa_DK':r['kappa_original']} for r in co_rows],
           'safety_factor':co['safety_factor'],
           'changes':['Panel B now uses log scale so target=2 is visible; original linear 0-0.05 axis clipped it.',
                      'Panel D uses log scale to show small values relative to 0.5 threshold.',
                      'Panel D is explicitly labeled kappa_DK, distinct from Panel B kappa_gap.']}
    (OUT/'Fig9_data_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    return audit


# ------------------------------ Fig. 10 -----------------------------------
def fig10():
    hemo=load_json(DATA/'b1__basis_variance_audit.json')
    mel=load_json(DATA/'b1__basis_audit_frangieh.json')
    order=['grn_spectral','grn_indicator','permuted','pca32','nmf32']
    labels={'grn_spectral':'GRN spectral','grn_indicator':'GRN indicator',
            'permuted':'Permuted control','pca32':'PCA-32','nmf32':'NMF-32'}
    colors={'grn_spectral':PURPLE,'grn_indicator':PURPLE_LIGHT,
            'permuted':GREY,'pca32':BLUE,'nmf32':GREEN}
    # Full-width stacked panels retain readable basis labels and show the
    # basis-specific covered-gene denominator explicitly.
    fig,axes=plt.subplots(2,1,figsize=fp.figsize(height_mm=132),sharex=True)
    datasets=[(axes[0],hemo,'A','Hematopoietic basis audit'),
              (axes[1],mel,'B','Melanoma Perturb-CITE-seq')]
    for ax,dat,tag,title in datasets:
        lookup={r['basis']:r for r in dat['rows']}
        vals=[float(lookup[k]['block_share']) for k in order]
        cov=[int(lookup[k]['n_covered']) for k in order]
        ypos=np.arange(len(order))[::-1]
        for y,k,v,n in zip(ypos,order,vals,cov):
            ax.barh(y,v,height=.58,color=colors[k],edgecolor=fp.EDGE,
                    linewidth=fp.LW_EDGE,zorder=2)
            ax.text(v+.009,y,f'{v:.3f}',va='center',ha='left',
                    fontsize=fp.FS_NOTE,color=INK)
        n_grn=int(lookup['grn_spectral']['n_covered'])
        chance=32/n_grn
        ax.axvline(chance,color=GRAY,ls=':',lw=fp.LW_REF,zorder=1)
        ax.set_yticks(ypos,[f'{labels[k]}  (n={lookup[k]["n_covered"]:,})' for k in order])
        ax.set_xlim(0,.70)
        panel_title(ax,tag,title)
        ax.text(.985,.88,f'Rank = 32; dotted line = 32/{n_grn} = {chance:.3f}',
                transform=ax.transAxes,ha='right',va='top',
                fontsize=fp.FS_NOTE,color=MUTED)
        style(ax)
    axes[1].set_xlabel('Variance share within each basis-specific covered-gene block')
    # Leave a visible right margin for journal trim and PDF rasterization.
    fig.subplots_adjust(left=.24,right=.970,bottom=.10,top=.94,hspace=.54)
    save_figure(fig,'Fig10_grn_basis_variance')

    ratios={}
    for ds,dat in [('hematopoiesis',hemo),('melanoma',mel)]:
        lookup={r['basis']:r for r in dat['rows']}
        ratios[ds]={
            'cells_reported_in_source_json':dat['n_cells'],'genes':dat['n_genes'],
            'values':{k:{'block_share':lookup[k]['block_share'],
                         'n_covered':lookup[k]['n_covered']} for k in order},
            'grn_spectral_over_pca':lookup['grn_spectral']['block_share']/lookup['pca32']['block_share'],
            'grn_spectral_over_nmf':lookup['grn_spectral']['block_share']/lookup['nmf32']['block_share'],
            'chance':32/lookup['grn_spectral']['n_covered']}
    audit={'datasets':ratios,
           'interpretation':'Block shares are reported within each basis-specific covered-gene block; covered-gene counts are displayed to make denominator differences explicit.',
           'source_discrepancy':'The supplied hematopoiesis JSON reports n_cells=3,000, whereas the legacy make_fig_paper3.py label says 12k cells. The figure therefore omits a hematopoietic cell count pending source reconciliation.',
           'caption_caution':'Avoid a blanket one-twentieth claim: GRN-spectral/data-fitted ratios differ between datasets.'}
    (OUT/'Fig10_data_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    return audit


CAPTIONS = {
'Fig6': ('Dose and training duration approach a local readout plateau. (A) Matched per-gene effect across prior-weight doses at a fixed 15-epoch weighted phase. (B) The same effect at fixed 20× dose across 15 and 45 weighted epochs. Thin lines connect the same training seed; diamonds and whiskers show the mean and standard deviation across three seeds. The response variable is the matched per-gene readout effect, not the aggregate decoder statistic.'),
'Fig7': ('A support-dependent bound limits what a global spectrum can detect. (A) Change in aggregate predictability, Δκ = κ − 1, as a function of prior support fraction. Curves show the arithmetic ceiling, the measured local gain on supported genes, and the same gain with the matched control’s global cost; faint curves show the all-gene denominator convention. Points represent the nine trained dose-by-seed arms, plotted as κ − 1; their x positions are slightly jittered around the common varying-gene support for visibility. (B) Decomposition of local contributions from named genes and global contributions from all other genes for the real prior and its weight-matched random control. The support fraction and denominator convention are specified in the Methods.'),
'Fig8': ('The 50% compression band marks loss of predictive capacity. Five training-budget conditions (2, 5, 10, 20 and 45 epochs) use the same seed, data and schedule, with no prior, adversarial branch or geometric penalty; κ is referenced to the 45-epoch arm. (A) κ across training budgets under varying-gene and all-gene denominator conventions. (B) κ against mean per-gene R² on varying genes; the dashed segment is a linear extrapolation to the 50% band, not an observed training condition. This is a single-seed descriptive calibration.'),
'Fig9': ('Prior-activity checks on human IPF and cross-architecture external validation. (A) Global compression across prior strengths for BatchNorm and LayerNorm decoders (seed 42). (B) The corresponding κ_gap remains far below the training target of 2; the logarithmic axis displays both observed values and the target. (C) Structural upper bound on κ_gap as a function of global compression; at 90% compression the bound is approximately 0.11. (D) Eight CellOracle-style single-TF priors ranked by κ_DK. The κ_DK metric is distinct from κ_gap in panel B; all eight values remain below the 0.5 reference threshold, with the reference-gap limitation described in the Methods.'),
'Fig10': ('GRN-derived bases capture less expression variance than data-fitted bases. Variance share is calculated within each basis-specific covered-gene block at rank 32. (A) Hematopoietic data (3,000 cells). (B) Independent melanoma Perturb-CITE-seq data (4,000 cells). GRN spectral and indicator bases are compared with a permuted control and data-fitted PCA-32 and NMF-32 bases. Covered-gene counts are shown for each basis because the blocks differ slightly in size. The dotted line indicates the dimension-counting expectation, 32 divided by the number of genes covered by the GRN basis.')
}
(OUT/'Figure_captions_Fig6_Fig10.txt').write_text('\n\n'.join(f'{k}. {v}' for k,v in CAPTIONS.items()),encoding='utf-8')

if __name__=='__main__':
    audits={}
    audits['Fig6']=fig6()
    audits['Fig7']=fig7()
    audits['Fig8']=fig8()
    audits['Fig9']=fig9()
    audits['Fig10']=fig10()
    (OUT/'combined_data_audit.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
    print('Generated Fig 6-10 in',OUT)
