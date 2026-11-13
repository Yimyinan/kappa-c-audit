# NUMBERS — every quantity in main.tex and where it comes from

Discipline borrowed from paper1 and paper2: **a number may only appear in main.tex if it
appears here.** Each row names the JSON and the script that produced it, so any figure can
be re-derived without trusting the prose.

Paths are relative to `F:\UNAGI_Project\`.

## Design constants

| quantity | value | source |
|---|---|---|
| cells / genes | 12,000 / 20,527 | `prior_delivery_runs/dose20/s42/real/hemato.h5ad` (stratified, 2,400/stage) |
| support $n_s$ | 203 curated TRRUST targets | `arms.py: arm_binary` over the 11 curated haematopoietic TFs |
| $S=\sum_g R^2_g(\text{none})$ | 5,536.3 | `dose20_e45/s*/empty_e45/pergene_readout_r2.npy`, summed |
| mean baseline $R^2$ (all genes) | 0.270 | same |
| trained models | 73 distinct; 88 arm directories, 15 of them byte-identical copies of three shared no-prior baselines | `find prior_delivery_runs -name latent.npz` (88) then sha256 (73); per-family table in `si/SI_NUMBERS.md` |

## Delivery

| quantity | value | source |
|---|---|---|
| $n_s\times$cells per stage file | 487,200 = 2,400 $\times$ 203 | `arm_meta.json: stagedata_final` |
| realised weight max, $5\times$ / $20\times$ / $40\times$ | 4.8097 / 16.8365 / 28.8665 | `dose20`, `dose40` `arm_meta.json` |
| basis vs permuted Gram, Frobenius | $2.5\times10^{-13}$ | `modules.py` output → `modules_p32.npz` metadata |

## Per-gene effect against a matched 75-epoch no-prior baseline

| quantity | value | source |
|---|---|---|
| real, named genes | $+0.00861\pm0.00338$ (3/3) | `dose20_e45/pergene_vs_baseline.json` |
| real, other 20,122 genes | $+0.00017\pm0.00021$ | same |
| control, its own genes | $+0.00188\pm0.00055$ | same |
| control, other genes | $-0.00250\pm0.00078$ (0/3) | same |
| arithmetic closure | $0.01572 = 0.00861+0.00711$ | `pergene_vs_baseline.json` |
| baseline $R^2$: targets vs other genes | 0.3005 / 0.2669 | `dose20_e45/pergene_robust.json` (`baseline_r2_real`, `baseline_r2_rest`) |
| slope of $\Delta R^2$ on baseline $R^2$ | $-0.004$ | `pergene_robust.py` (both roots) |
| decile-matched retention | 72\% (0.01144 / 0.01594) | `dose20_e45/pergene_robust.json` |
| OLS-residual retention | 84\% (0.01335 / 0.01594) | same |

## Ceiling

| quantity | value | source |
|---|---|---|
| matched $d$, $5\times$ / $20\times$ / $40\times$ | 0.00243 / 0.01144 / 0.01533 | `dose_ladder.json` |
| matched $d$, 15 vs 45 weighted epochs at $20\times$ | 0.01144 / 0.01439 | `dose20_e45/pergene_robust.json` |
| ratio / time exponent | 1.26 / 0.21 | derived |
| log--log slopes, $5\to20\times$ and $20\to40\times$ | 1.12 / 0.42 | derived from the matched $d$ above |
| ceiling, quoted as a single number | $\approx+0.0145$ | 0.01439 (time route) and 0.01533 (dose route) |
| between-run noise floor, $\Sigma R^2$ and per gene | 1.14\% / 0.0117 | `b1/probe/s42_real_l0` vs `support/s42/empty` |
| between-seed s.d., $20\times$ vs $40\times$ | 0.00057 vs 0.00291 | `pergene_robust` roots |

## Spectrum

| quantity | value | source |
|---|---|---|
| $\kCW$ across nine arms | 0.9867–1.0145 | `dose_ladder.json` |
| $\kCW$ real vs matched 75-epoch baseline | $1.0009\pm0.0007$ (0.09\%) | `dose20_e45/s*/audit_vs_e45empty.json` |
| $\kCW$ control vs same baseline | $0.9907\pm0.0028$ | same |

## The bound (Section 5)

| quantity | value | derivation |
|---|---|---|
| ceiling $\Delta\kappa$ at $f=1\%$, perfect readout | $+0.027$ | Eq. (2) with $\delta=1-0.27$ |
| measured $\Delta\kappa$ (real) | $+0.0009$ | Eq. (2), Local term +0.03\%, global +0.06\% |
| $\Delta\kappa$ if measured $\delta$ applied to all genes | $+0.032$ | $N\delta/S$ |
| support needed for $\Delta\kappa=-0.5$ | $1,566\%$ (all genes), $1,335\%$ (varying) | $0.5\,ar{R^2}/\delta$ inside each convention |
| control's global term | $-0.91\%$ | Eq. (2) |
| figure | `fig/fig1_resolution.pdf` | `b1_expansion/prior_delivery/fig_resolution.py` |

## Geometry operator (B1)

| quantity | value | source |
|---|---|---|
| $A$ at $\lambda$ = 0 / 0.1 / 1 / 10 | 0.158 / 0.534 / 0.912 / 0.969 | `b1/probe/s42_real_l*/arm_meta.json` |
| contrast $\Delta A$ at $\lambda=0.1$ | $-0.040/-0.008/+0.025$ | `b1/contrast/s*/{real,perm}/arm_meta.json` |
| contrast $\Delta R^2$ | $+0.00069\pm0.00236$ | `pergene_readout_r2.npy` in each contrast arm |
| same-configuration replicate | $A=0.7515$ vs $0.4682$ (spread 0.283) | `b1/contrast/s42/{real,real_dup}` |
| module basis, haematopoiesis | 5.27\% / 3.91\% / 60.7\% / 57.4\% | `b1/basis_variance_audit.json` |
| module basis, melanoma Perturb-CITE-seq | 3.46\% / 2.88\% / 18.8\% / 16.1\% | `b1/basis_audit_frangieh.json` |
| annotation-to-permuted ratio | 1.35$\times$ (haematopoiesis), 1.20$\times$ (K562) | derived |
| annotation-to-data-fitted ratio | $\approx$1/11 (haematopoiesis), $\approx$1/5.4 (K562) | derived |

## Literature numbers (must be verified against PDFs before submission)

| quantity | value | source | status |
|---|---|---|---|
| PVAED, prior only | $+32.27\%$ | Ovid full text | verbatim quote held; **PDF check pending** |
| PVAED, neighbourhood loss | $+35.24\%$ | Ovid full text | same |
| PVAED, prior + diffusion | $+43.16\%$, $+8.39\%$ over PVAE | Ovid full text | same |
| GEDI | no with/without-$C$ comparison; no spectral metrics | Nature Comms full text | body truncated; check Sup Fig 8–12 |
| Inferelator, shuffled prior | AUPR $0.20\to0.02$ | eLife 51254 | from abstract/results text |
| iAODE, six spectrum metrics | self-defined, zero citations (Eq. 26–30) | Commun Biol full text | verbatim quote held |

## Rounding used in the abstract (allowed, but only these)

| in the text | exact |
|---|---|
| $A$: 0.16 $\to$ 0.97 | 0.158 $\to$ 0.969 |
| $+0.86$ pp | $+0.00861$ |
| ceiling $+1.45$ points | $+0.0145$ |
| "thirteen to sixteen times the transcriptome" | 1,335% (varying) to 1,566% (all genes) |

## Figures

| figure | file | produced by | reads |
|---|---|---|---|
| Fig 1 | `fig/fig1_resolution.pdf` | `../b1_expansion/prior_delivery/fig_resolution.py` | `dose_ladder.json`, constants in the script |
| Fig 2 | `fig/fig2_ceiling.pdf` | `make_figs.py` | `support/pergene_robust_5x.json`, `dose20/pergene_robust.json`, `dose40/pergene_robust.json`, `dose20_e45/pergene_robust.json` |
| Fig 3 | `fig/fig3_basis.pdf` | `make_figs.py` | `b1/basis_variance_audit.json`, `b1/basis_audit_frangieh.json` |

## Readout audit, 2026-10-28 (held-out + interaction + support ladder + sign)

| quantity | value | source |
|---|---|---|
| real prior on own 203 genes, in-sample / 5-fold / kNN | +0.00879 / +0.00641 / +0.00565 | `dose20_e45/cv_nonlinear_null.json` |
| shuffled prior on own genes, in-sample / 5-fold / kNN | +0.00174 / +0.00426 / +0.00221 | same |
| "gain over baseline", in-sample vs 5-fold | 0.00705 vs 0.00215 | derived |
| real prior evaluated on the control's genes (5-fold / kNN) | +0.00060 / +0.00058 | `dose20_e45/cross_support_did.json` |
| shuffled prior evaluated on the real 203 genes (5-fold / kNN) | -0.00601 / -0.00239 | same |
| difference-in-differences, per seed (5-fold) | +0.00644 / +0.01649 / +0.02531 | same |
| difference-in-differences, per seed (kNN) | +0.00918 / +0.01105 / +0.00877 | same |
| permutation p, 20,000 gene-set pairs (5-fold / kNN) | 0.0098, <0.0001, 0.0019 / <0.0001 x3 | same |
| permutation z | +2.6 to +8.0 | same |
| support ladder kappa-C(W): 203 / 1,000 / 2,027 genes at 5x | 0.9974 +/- 0.0097 / 1.0014 +/- 0.0045 / 1.0004 +/- 0.0072 | `support/support_summary.json` |
| predicted aggregate change at 2,027 genes | 0.31% = 2027 x 0.0086 / 5536 | derived |
| TRRUST edges / signed fraction | 9,396 / 54.0% (3,149 act, 1,922 rep, 4,325 unknown) | `prior_sign_audit.json` |
| edges touching the 203 genes | 3,726 (1,281 / 778 / 1,667) | same |
| genes receiving both activation and repression | 83 of 203 | same |
| genes hit by more than one edge | 171 of 203 | same |
| kNN readout, overall R2 vs ridge OOF | 0.1892-0.1917 vs 0.1878-0.2019 | `cv_nonlinear_null.json` |

Note: the 5-fold and kNN columns come from the same folds (seed 0) for every arm and
every baseline, so all contrasts are paired. The gene-level null resamples 203-gene sets
from the 20,527 genes outside the support.

### Rounding used for the interaction paragraph

| in the text | exact |
|---|---|
| DiD $+0.0064/+0.0165/+0.0253$ | +0.00644 / +0.01649 / +0.02531 |
| DiD $+0.0092/+0.0111/+0.0088$ (kNN) | +0.00918 / +0.01105 / +0.00877 |
| $p=0.010$ | 0.0098 (20000 draws, +1 smoothing) |
| baseline weights 0.9619 / 0.8418 / 0.7217 | $1/\mathrm{mean}(w)$ at 5x / 20x / 40x; matches `arm_meta.json` |

DOIs in the reference list (10.1038, 10.1093, 10.7554) are identifiers, not measurements.

## Noise-source control, 2026-10-28 (adversarial branch off)

| quantity | value | source |
|---|---|---|
| same-config replicate A, with GAN | 0.4682 vs 0.7515, gap 0.2833 | `b1/contrast/s42/{real,real_dup}` |
| same-config replicate A, GAN off | 0.7809 vs 0.7780, gap 0.0029 | `b1/ganoff/s42/{real_a,real_b}` |
| reduction in replicate gap | 98x | derived |
| A under GAN off vs on | ~0.779 vs 0.468-0.752 | same |
| A, permuted basis, GAN off | see next line when landed | `b1/ganoff/s42/perm` |

### Resolved values for the repaired geometry comparison

| in the text | value |
|---|---|
| real basis, two replicates, GAN off | 0.7809, 0.7780 (mean 0.7794, gap 0.0029) |
| permuted basis, GAN off | **0.7706** |
| $\Delta A$ real - perm | **+0.0088** = 1.1% of A, ~4x the two-replicate gap |
| range of A swept by lambda alone | 0.158 -> 0.969 |

## Common-basis contrast (amplified, 2026-10-28)

| quantity | value | source |
|---|---|---|
| A_C / A_P, real replicate 1 | 0.7139 / 0.6325 | `b1/paired_basis_contrast.json` |
| A_C / A_P, real replicate 2 | 0.7053 / 0.6233 | same |
| A_C / A_P, permuted basis | 0.6495 / 0.7051 | same |
| within-decoder delta, real (two replicates) | +0.0814 / +0.0820 | same |
| within-decoder delta, permuted | -0.0556 | same |
| interaction | +0.1373 (19.4% of A) | derived |
| replicate spread of the real within-contrast | 0.00066 | derived |
| top-32 delta, real / permuted | +0.0263, +0.0282 / -0.0186 | same |
| top-32 interaction | +0.0459; real/permuted ratio > 1.6 | derived |
| principal cosines between span(C) and span(P) | max 0.872, mean 0.120 | same |
| with GAN on: within-delta real / real_dup / perm | +0.1957 / +0.0923 / -0.2801 | same |

Note: A here is recomputed from the saved decoder and the stored latent sample, so it is
offset from the training-time `align_stats` value (0.7809); the contrast, not the level, is
the quantity used.

## Expression-derived basis, 2026-10-28 (the pincer)

| quantity | value | source |
|---|---|---|
| basis variance share, held-out cells: expression PCA-32 / annotation / permuted / chance | 57.9% / 5.9% / 3.2% / 1.8% | `b1/modules_expr_p32.json` |
| non-zero support (both bases) | 1,768 of 1,999 (231 constant in this dataset) | same |
| principal cosines, expression vs annotation subspace | max 0.56, mean 0.15 | same |
| training A: annotation real / perm | 0.7809, 0.7780 / 0.7706 | `b1/ganoff/s42`, `b1/contrast/s42` |
| training A: expression real / perm | 0.4113 / 0.7152 | `b1/expr2/s42` |
| A_C vs A_P on the expression basis: expr real / expr perm | 0.4012 vs 0.2095 / 0.0972 vs 0.6174 | `b1/paired_expr2.json` |
| within-decoder delta, expression family | +0.1918 / -0.5202 | same |
| covered-block per-gene R2: expr real / expr perm / annot real x2 / annot perm | 0.18691 / 0.18986 / 0.18891, 0.18797 / 0.18819 | `b1/basis_source_readout.json` |

Pre-registered predictions (written 21:18, before the arms finished): P1 interaction same
order as the annotation family -- **wrong** (+0.712); P2 |readout difference| < 0.005 --
**right** (-0.0020, -0.0011); P3 expression family real > perm -- **wrong** (-0.00295).

Rounding: the top-32 interaction quoted as $+0.046$ is 0.04585 in
`b1/paired_basis_contrast.json`.

## Phantom-gene denominator, 2026-10-28

| quantity | value | source |
|---|---|---|
| genes constant in this dataset | 3,199 of 20,527 | `support/s42/empty/latent.npz` |
| their per-gene R2 | 1.000 exactly (0/0 convention) | same |
| their share of Sum R2 | **58.4%** (3,199 of 5,478) | same |
| S with / without them | 5,478 / 2,279 | derived |
| kappa, permuted control: all genes vs varying only | 0.9931 vs 0.9835 (-0.69% vs -1.65%) | `support/s42` and `dose20_e45/s42` |
| support fraction, all vs varying genes | 0.99% vs 1.17% | derived |
| support needed for the 50% band, all vs varying | 1,566% vs 1,335% | derived, each inside its own convention |
| kappa shift when constant genes are dropped, real / permuted | +0.22% / -0.96% | same |

Three-point test of Eq. 1 against the support ladder: the predicted dkappa is +0.02% to
+0.10%, while the observed between-seed scatter of kappa itself is ±0.6%. The ladder
therefore cannot confirm the bound; it can only show that the predicted term sits below the
drift floor, which is consistent with the bound and with everything in Section 3.


## Correction, 2026-10-29 (mixing conventions)

An earlier draft quoted "759% of the transcriptome" for the varying-gene convention. That
number combined the all-gene delta (0.0086) with the varying-gene mean R2 (0.134) and is
wrong. Recomputed inside each convention: the ceiling at the observed support is 2.69%
(all genes) or 7.12% (varying), and the support needed for the band is 1,566% or 1,335%
respectively. Fig. 1 now draws both, and the text quotes the range.

| quantity | all genes | varying only | source |
|---|---|---|---|
| N | 20,527 | 17,328 | `dose20_e45/s42/empty_e45` |
| S | 5,521.3 | 2,322.3 | same |
| mean R2 | 0.2690 | 0.1340 | same |
| support (named / varying) | 203 | 191 | `s42/real/support.npz` |
| f | 0.99% | 1.10% | derived |
| delta_local, real | +0.00861 | +0.00502 | same |
| delta_global, real | +0.00017 | +0.00045 | same |
| delta_local, control | +0.00188 | -0.00955 | same |
| delta_global, control | -0.00250 | -0.00212 | same |
| ceiling (R2 -> 1) | 2.69% | 7.12% | derived |
| measured, real | +0.094% | +0.373% | derived |
| measured, control | -0.913% | -1.643% | derived |
| needed for the band | 1,566% | 1,335% | derived |

## Calibration ladder (Fig. 4)

| epochs | kappa (all) | kappa (varying) | mean R2 (varying) |
|---|---|---|---|
| 2 | 0.9413 | 0.8616 | 0.1173 |
| 5 | 0.9539 | 0.8915 | 0.1213 |
| 10 | 0.9660 | 0.9198 | 0.1252 |
| 20 | 0.9790 | 0.9506 | 0.1294 |
| 45 | 1.0000 | 1.0000 | 0.1361 |

Source: `prior_delivery_runs/calib/s42/calibration.json`. Linear extrapolation of
kappa on mean R2 reaches 0.5 at R2 = 0.068 (slope 7.35); a random decoder lies below zero.

## Gene-level geometry (Section 6)

| arm | rho(|J_g|, R2) | rho(in-basis share, R2) | pooled share |
|---|---|---|---|
| annot real 1 | -0.050 | +0.046 | 0.736 |
| annot real 2 | -0.053 | +0.049 | 0.728 |
| annot perm | -0.049 | +0.057 | 0.669 |
| expr real | +0.336 | -0.052 | 0.493 |
| expr perm | +0.086 | +0.063 | 0.168 |

n = 1,827 covered genes that vary; Spearman. Source: `b1/geometry_predicts.json`.

## Constant-gene share vs audit sample size (Section 5, corrected)

| audited cells | constant genes | % of genes | share of Sum R2 |
|---|---|---|---|
| 500 | 4,744 | 23.1% | 46% |
| 1,000 | 3,357 | 16.4% | 38% |
| 2,000 | 2,199 | 10.7% | 28% |
| 3,000 (uniform draw) | 1,505 | 7.3% | 21% |
| 6,000 | 598 | 2.9% | 10% |
| 12,000 | 167 | 0.8% | 3% |
| 16,161 (all cells) | 108 | 0.5% | 2% |

Source: `hemato_input/0/stagedata/0.h5ad`, uniform random subsample, seed 0. Our own audit
draw is stratified and lands at 3,199 constant genes (15.6%, 58% of Sum R2) for the same
3,000 cells, i.e. worse than a uniform draw -- stratification favours low-depth cells.

**paper1 is not affected:** its audits run on whole datasets (29,985 / 21,078 / 19,999
cells), where constant genes are under 1% and contribute about 2% of Sum R2.

Rounding: the expression-arm correlation quoted as $+0.34$ is 0.336 in
`b1/geometry_predicts.json`.

## VEGA audit (Section 1, quoted from the companion manuscript)

| quantity | value | source |
|---|---|---|
| cells x genes, pathways, epochs, seeds | 16,893 x 6,998, 675, 150, 3 | `VEGA/audit_out/vega_*_seed*.json` |
| sum R2, vanilla / vega / vega+rewired | 44.2-52.4 / 244.2-259.3 / 218.0-238.6 | same, `vega_audit_summary.json` |
| median cross-arm kappa-C(W), nine pairings | 5.04 (4.67-5.86) | `vega_audit_verdict.json` |
| vanilla-vs-vanilla null | median 1.00, range 0.845-1.183 | same |
| equivalence band used in that audit | [0.78, 1.23] | same (`preregistered_band`) |
| paired gain retained by degree-preserving rewiring, 11 rewirings (seed 1) | 86.9-98.3%, median 91.9% | `paper1/code/s1i_content_null.py` |
| paired gain retained by degree-preserving rewiring, 30 rewirings x 3 seeds | 89.9% (content 10.1%, 95% CI 3.5-16.7 across seeds) | `_infra/tools/vega_rewire_extend.py --to 30`, `_infra/tools/content_random_summary.py` |
| content readout per seed, 30 rewirings each | 7.4 / 10.3 / 12.7% | same script |
| single-seed rewiring-sampling CI (seed 1, 30 rewirings) | content 6.5-8.3% | `_infra/tools/content_ci.py --sizes 5,11,20,30` |
| implied content share, rewiring control | 10.1% (readout; not an upper bound -- see below) | derived |
| sum R2 of a same-size random mask, three seeds | 282.2 +/- 7.0 (274.04 / 286.95 / 285.52) | `VEGA/audit_out/vega_vega_random_seed*_rnd*.json`, `_infra/tools/content_random_summary.py` |
| sum R2 of a row-size-matched random mask, three seeds | 321.6 +/- 2.0 (319.34 / 322.41 / 323.10) | `VEGA/audit_out/vega_vega_random_rows_seed*.json`, `_infra/tools/vega_audit_train.py --arm vega_random_rows` |
| sum R2 after reassigning gene-to-pattern rows, three seeds | 243.7 (228.12 / 253.17 / 249.68) | `VEGA/audit_out/vega_vega_rows_perm_seed*.json`, `_infra/tools/vega_audit_train.py --arm vega_rows_perm` |
| sum R2 pooled over three seeds, masked / prior-free | 253.9 / 49.3 | same script (`vega_content_random.json`) |
| implied content share, random-mask control (per seed) | -14.9 / -14.0 / -12.7%, mean -13.8% | derived, same script |

## RegVelo: a learned prior, same split (Discussion) -- RE-SCORED 2026-10-29

Both arms recovered from the saved state dicts (no retraining), 5-fold with shared folds.

| quantity | vanilla | GRN | source |
|---|---|---|---|
| per-gene R2, in-sample | 0.41959 | 0.39775 | `b1_expansion/prior_delivery/rescore_regvelo.json` |
| per-gene R2, held-out | 0.41247 | 0.39030 | same |
| sum R2, held-out | 824.93 | 780.61 | same |
| kappa-C(W), in-sample / held-out | -- | **0.9480 / 0.9463** | same |
| delta kappa, held-out | **5.37%** (compression) | | derived |
| genes improved, held-out | -- | 34.0% (i.e. 66% worse) | same |
| decoder Gram eig1 / eig2 | 16.79 / 9.30 | 20.59 / 8.59 | same |

Setup: 1,947 cells x 2,000 genes, 10 latent dims, Ms layer, Ridge(Z -> Ms) with intercept,
alpha = 1. No constant genes in this subset (all 2,000 vary), so the two conventions coincide.

Supersedes the earlier note in this file that quoted the 159-HVG run (eig 30.97 -> 57.06,
GATA1/GATA2 loadings). Those weights are a different, unverified checkpoint set; the numbers
above are the ones reproduced from saved state dicts and are what the text cites.

Rounding in the RegVelo paragraph: $0.4125$ is 0.41247 and $0.3903$ is 0.39030;
$16.79$ / $20.59$ are exact to the digits reported by the script.

## Prose-only quantities, tabulated 2026-10-31 (added with the automated gate)

`verify_numbers.py` was added on 2026-10-31. It extracts every numeric literal from `main.tex`
and requires it to appear in this file, so a claim cannot reach the prose without a ledger row.
The rows below were previously stated only in the prose. Sources name the artefact that the
surrounding rows already cite; where the exact field was not re-read, the script is named rather
than the field, and that is marked.

| quantity | value | source |
|---|---|---|
| edges in the curated haematopoietic subset | 325 | `arms.py: arm_binary` over the 11 curated TFs |
| support fractions 203 / 1,000 / 2,027 of 20,527 | 1.0% / 4.9% / 9.9% | derived |
| dose-step fold change, $5\times\to20\times$ | 4.7x | derived (0.01144 / 0.00243) |
| control arm's apparent compression | 0.9% | derived (1 - 0.9907) |
| training-budget cost of a 22x shorter run | 14% of kappa (varying), 6% (all-gene) | `calib/s42/e2` vs `calib/s42/e45` |
| data-derived basis, share of the metric | 19% | `b1/expr2/s42` interaction |
| interaction $+0.137$ as a share of $A$ | 19.4% | `b1/contrast/s42/{real,real_dup}` |
| replicate spread of $A$, same configuration | 0.00066 | same |
| fraction of target genes improved, $20\times$, three seeds | 76.7 / 74.3 / 78.7% | `pergene_robust.py` (field not re-read) |
| fraction of control genes improved, $20\times$ | 37.6 / 43.1 / 35.1% | same |
| fraction of target genes improved, $40\times$ | 71.3 / 71.3 / 75.7% | same |
| constant-gene share at 500 cells, as quoted in prose | 23% (23.1% exact) | table above |
| covered genes in the basis figure, haematopoiesis / K562 | 1,999 / 2,111 | `b1/basis_variance_audit.json`, `b1/basis_audit_frangieh.json` |
| quoted from paper 1: configuration audit size (the 0/33 null) | 33 | paper 1 |
| quoted from paper 1: released stage files scanned | 1,116 | paper 1 |

## Correction, 2026-10-31 (one dataset label, one support figure)

**Dataset label.** `data/public/frangieh_2021.h5ad` was called "K562 CRISPRi" in the main
text, in Fig. 3's axis label and in `basis_audit_second.py`'s docstring. The file's own
`uns/doi` is `10.1038/s41588-021-00779-1`, i.e. Frangieh et al. 2021, a Perturb-CITE-seq
screen in patient-derived melanoma cells co-cultured with autologous TIL (CRISPR-Cas9;
conditions Control / IFNg / Co-culture). No number changes: the audit reads the expression
matrix and the TRRUST edges over that gene space. Text, Fig. 3 and the script are corrected;
the deposited file is unchanged.

**Support needed for the band.** The main text quoted 1,562% in one place and 1,566% in
another for the same quantity (all-gene convention). 0.5*S/delta = 0.5 * 5,536.3 / 0.00861 =
321,504 genes = 1,566.2% of 20,527, so 1,566 is the value the formula returns and 1,562 was
stale. Both documents now say 1,566.

**Model count.** The text said 63 models in 78 directories. The tree now holds 88 directories;
hashing `latent.npz` gives 73 distinct artefacts and 15 byte-identical copies, all of them
copies of the three shared no-prior baselines. 88 - 15 = 73, and both documents now say so.
