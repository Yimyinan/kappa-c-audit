# Correction history

This file records what changed and **why**. It is kept because the corrections are part of the
evidence chain: several earlier numbers were wrong or not reproducible, and the entries below say
which, and on what basis they were replaced. Read the current definitions in `README.md` and in the
script/docstring that actually computes each quantity — **not** here.

---

## Paper 1 V49 repository refresh (2026-11-21)

- Updated the project title and metadata to the V49 manuscript.
- Added the Gaussian MLP and ZINB count-VAE delivered-prior experiments.
- Added the V49 Author Summary and manuscript metadata.
- Added the current figure generators, including the corrected Figure 1
  estimand and Figure 7 panel title.
- Added current PG-UNAGI delivered-prior code and JSON result summaries.


## Tool: `kappa-c-audit`

| Version | Change |
|---|---|
| **0.1.2** | `DEFAULT_BAND` updated to the corrected **predictability-form** vanilla-equivalence band **`[0.937, 1.064]`** (1.001 ± 0.032 over 15 G-VAE hematopoietic seeds / 210 ordered seed-pair comparisons). It had carried the superseded **trace-form** band `[0.78, 1.23]`. This changes only the `vanilla-equivalent` vs `amplified` / `sub-vanilla` boundaries; the band-independent deep-compression verdict (`κ < 0.5`) is unaffected. |
| **0.1.1** | Default switched to the predictability form; `predictability_ratio()` corrected (0.1.0 averaged per-output `score()` values rather than summing, and used an intercept on unstandardised data, so it did not match `kappa_C_W` as its docstring claimed). |
| **0.1.0** | First release; trace-form estimator (still available as `form="trace"` so previously published numbers reproduce exactly). |

The 0.1.2 wheel was **rebuilt once after the first build**, with no version bump (nothing had been
published yet): its `METADATA` carried `Project-URL: Homepage, https://github.com/[REPOSITORY_URL]`,
a literal placeholder baked in at build time. The `[project.urls]` block is now omitted until the
repository exists, so no broken URL is archived with the DOI. The four source files in the wheel are
byte-identical between the two builds; only `METADATA` / `WHEEL` / `RECORD` differ.

**If you add the real Homepage URL, rebuild the wheel in the same sitting** — the URL is captured at
build time, so editing `pyproject.toml` alone will not change the shipped artefact.

---

## Packaging refresh for the r38 manuscript (2026-11-10)

The archive platform moved from Zenodo to **GitHub (live code) + figshare (versioned snapshot)**, and
the repository metadata was re-audited against the r38 manuscript. Nothing had been published, so the
refresh carries no version bump:

| Item | Was | Now |
|---|---|---|
| `.zenodo.json` | present, Zenodo-specific | **removed** — figshare takes its metadata from the deposit form, not from a file in the repo |
| `README.md` → title | quoted the pre-r33 title (*When the prior does not arrive…*) | the r38 title (*A delivery-and-content audit finds…*) |
| `README.md` → usage | `rep.kappa_C_W`, `rep.classify()` | `rep["kappa_C_W"]`, `rep["verdict"]` — `audit()` has returned a `dict`, not an object, since 0.1.1, and the README was the last place still showing the old access pattern |
| `pyproject.toml` → authors | `Yan Yinan` (family-first) | `Yinan Yan` + `Lan Wang`, matching the manuscript |
| `CITATION.cff` | one author; *School of Life Sciences*; 2026-09-23 | two authors; the manuscript's laboratory affiliation; 2026-11-10 |
| wheel | built 2026-09-23, embedding the **old** README and `Author: Yan Yinan` | rebuilt from these sources; same version, because nothing was published |

The rebuild is why the wheel hash changes without a version bump: `METADATA` embeds the README and
the author fields at build time. `.gitignore` still commits exactly one wheel, because the README's
primary install line points at it.

---

## Packaging and metadata corrections (2026-09-23)

A pre-publication audit of this release found that its top-level `README.md` and `.zenodo.json`
described a *much larger* package than the one actually present, and that several metadata values
were stale. Because the Zenodo DOI description and author name are **immutable once published**,
these were fixed before any upload:

| Item | Was | Now |
|---|---|---|
| `.zenodo.json` → `version` | `v1.0` | `0.1.2` (matches the wheel) |
| `.zenodo.json` → `creators` | `Yim, Yinan` | `Yan, Yinan` (the manuscript's author name since r24) |
| `.zenodo.json` → `title` / `description` | promised manuscript sources, figures and tables S1–S12 | scoped to what ships: the tool, the analysis code, and the curated result JSONs |
| `README.md` | listed 17 `.tex` / `figures/` / `data/` paths that were absent; presented the superseded band `[0.78, 1.23]` as current; published internal correction history | rewritten to match the contents; band caveat stated; correction history moved **here** |
| release contents | 13 files / 96 KB | the tool **plus** `code/` (134 scripts) and `data/` (54 curated result JSONs) |
| `.so`-free / licence | — | `LICENSE` (MIT) and `CITATION.cff` present; `.gitignore` now keeps the one released wheel committed, since the README's primary install line points at it |

Two stale artefacts were moved out of the working tree (not deleted) to
`paper1/_quarantine/20260923_repo_metadata/`: an older `.zenodo.json` (`v1.0`, `Yim, Yinan`) left
over from a plan to use the working tree itself as the repository root, and the superseded
`kappa_c_audit-0.1.1` wheel. Both are documented there. The 0.1.1 and earlier bands are the
**retired trace form** and must not be used for classification.

---

## Figures removed in the 2026-08-29 re-audit

The following had no source data / contained fabricated values and were deleted:

- `figS2_grq_sensitivity` (fake GRQ sensitivity), `figS4_gata1_loadings` (fake GATA1 loading claim),
- `figS5_latent_umap` (empty), `figS6_prior_schematic` (unreferenced), `figS7_perturbation_pca` (negative-sign bug + fake data),
- `fig5_ctcp_polarity` / `fig6_pareto` / `fig7_cross_tissue` (uncited leftovers, duplicated by Fig. S8/S9), `supp_figS10_cross_tissue` (a stale copy of Fig. 2, not a cross-tissue figure), `_fig1`/`_fig2_preview.png`, `kc_phase_diagram.*`.

All remaining figures are regenerable from the generator scripts in `code/`.

---

## Supplementary-table re-audit (2026-08-29)

**Pass 5**
- `table_s2`: encoder-divergence variance recomputed from the s9–s12 checkpoints; top-10 list corrected (7/10 erythroid, not 10/10); variance column now real (0.409–0.460).
- `table_s3`: target set is 31 genes; EPO removed (not expressed in the blood-stage dataset, excluded by the GRQ/CTCP protocol).
- `table_s5` / `table_s7`: Human IPF row updated to full-transcriptome fp32 values (trH 10,590; C = 95.3–96.7%; κ_gap 0.010–0.025); superseded phase-2 values (κ = 1.016, C = 83%) removed.
- `table_s6`: GRQ per-seed values replaced with the real decoder-traversal re-audit (11.45 / 13.83 / 16.80 / 11.32); the earlier 10.8–12.3× values were fabricated. CTCP column (19.2 / 19.4 / 18.8 / 19.6) and the p = 0.52 shuffle null verified.

**Pass 6**
- `table_s5` / `table_s7`: Data-ED / Latent-ED / Comp / Gini context columns removed — the original protocol is not reproducible from surviving artifacts (hema pilot: raw-data ED 3.9, z-scored 266.6 vs reported 114.6; latent ED 13.2 / 5.3 / 8.7 vs 19.9; no ED code survives).
- `main.tex` L73: `13.4 ± 2.4×` restated as `mean 13.4×` (SD 2.23 / 2.57 not reproducible).

**Pass 7**
- A dangling "Cross-tissue analysis" Methods paragraph (PR / Gini / Spearman) removed; the `main.tex` L73 restatement carried through.

---

## Band-related values in older result JSONs

Some files under `data/` still contain the superseded trace-form band `[0.78, 1.23]` in their
recorded statistics (notably `vanilla_null_v2.json`, `grq_seeds_s9s12_kappa.json`). They are kept
**as recorded** so that earlier reports remain traceable — they document the old→new transition.
**Use `DEFAULT_BAND = (0.937, 1.064)` from `kappa_c_audit` for any classification.**
