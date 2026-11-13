# Delivery experiment: results, scripts and number tables (v0.1.3 addition)

This directory is the v0.1.3 addition to the figshare deposit. Everything outside it in the
archive (`kappa_c_audit/`, `code/`, `data/`, and the top-level package files) is **byte-identical
to the published v0.1.2 deposit** (`kappa-c-audit-code-data-v0.1.2.zip`, SHA-256
`1d93e657663c68f36fc5ac60b4d56e171f8d5c9fc62684d7c794c60f53f2d778`). The v0.1.2 deposit covers
the audit half of the manuscript; this directory covers the **delivery half** (73 trained models,
the local tilt and the hard ceiling) that the merged manuscript reports, and which v0.1.2 did not
contain.

## Provenance

| content here | source on the authoring machine |
|---|---|
| `results/prior_delivery_runs/` | `F:\UNAGI_Project\prior_delivery_runs` (result JSON, 245 files) |
| `code/prior_delivery/` | `F:\UNAGI_Project\b1_expansion\prior_delivery` (analysis and driver scripts) |
| `code/pg_unagi/` | `F:\UNAGI_Project\b1_expansion\pg_unagi` (the patched UNAGI package that delivers the prior) |
| `code/paper3/` | `F:\UNAGI_Project\paper3` (figure/SI builders and the number verifier) |
| `NUMBERS.md`, `SI_NUMBERS.md` | `F:\UNAGI_Project\paper3\NUMBERS.md`, `paper3\si\SI_NUMBERS.md` |
| `prior/unagi_custom_data/trrust_raw_human.txt` | the curated TRRUST edge table used as the prior |

`NUMBERS.md` is the number ledger: in this project a number may appear in the manuscript only if it
appears there, with the JSON and the script that produced it. `SI_NUMBERS.md` gives the per-family
model counts (88 directories, 73 distinct models).

## What the results JSON cover

`results/prior_delivery_runs/` preserves the run-tree layout of the experiment:

- `dose_ladder.json`, `prior_sign_audit.json` (top level) — dose ladder and prior-sign audit.
- `dose20/`, `dose20_e45/`, `dose40/`, `ladder1/`, `seeds/`, `support/`, `r2/`, `b1/`, `calib/`,
  `b1_smoke/` — per family. Each family carries its summary files (`dose_summary.json`,
  `support_summary.json`, `seeds_summary.json`, `audit_results.json`, `pergene_vs_baseline.json`,
  `pergene_robust.json`, `bound_validation.json`, `cross_support_did.json`,
  `cv_nonlinear_null.json`, basis and pairing JSON, `fig1_numbers.json`, `run_status.json`).
- inside the arm directories, `arm_meta.json` records the delivered arm (weights, support set,
  realised weight maximum, weighted-epoch count, stagedata references); individual arms additionally
  keep `training_parameters.json`. Together these are the per-arm record from which the model count
  (88 directories, 73 distinct models after hashing) is re-derived.

## What is deliberately not here

- **Model `latent.npz` files, checkpoints (`.pth`) and model snapshots (`.npz`, `.npy`, `.pkl`)** —
  large binaries; the manuscript's claims are reproducible from the analysis scripts plus the JSON
  summaries, and the raw arrays are reconstructible from the model registry.
- **Expression matrices (`.h5ad`)** — third-party public data (GSE150774 and the other datasets
  named in the manuscript) and far over the deposit size budget.
- **Run logs (`.log`/`.txt`)** — operational, not results.
- **Every file above 5 MB** — excluded by the deposit rule for this archive.

## Mapping to the merged manuscript

- Results §delivery design / Table 2 — `dose_ladder.json`, `arm_meta.json` family summaries.
- Results §local tilt (Fig and text) — `pergene_vs_baseline.json`, `pergene_robust.json`.
- Results §ceiling / Table 3 / Fig 6 — the `dose_summary.json` and `*_summary.json` per family.
- Results §spectral resolution / Fig 7–8 / Table 4 — `bound_validation.json`, `support_summary.json`.
- Results §annotation basis / Fig 10 — `basis_*.json`, `paired_basis_contrast.json`,
  `geometry_predicts.json`.
- Supporting Notes 9–16 / Tables S12–S19 — `code/prior_delivery/*.py` and the JSON above.

## Reproducing

The scripts are plain Python; dependencies are `numpy`, `pandas`, `scipy`, `matplotlib`,
`scikit-learn` (and the patched UNAGI package in `code/pg_unagi/`). Entry points:

- `run_arm.py` — one arm; `run_dose.py`, `run_support.py`, `run_ladder.py`, `run_seeds.py`,
  `run_empty_e45.py`, `run_b1.py` — the families; `run_all.py` — the driver.
- `arms.py` — arm definitions (dose, support set, weight-matched gene permutation, no-prior
  baselines); `modules.py`, `modules_expr.py` — the delivered reconstruction term.
- `dose_summary.py`, `pergene_fit.py`, `pergene_robust.py`, `pergene_vs_baseline.py` — the readout
  statistics quoted in the manuscript; `bound_validation.py` — the resolution bound;
  `cross_support_did.py`, `cv_nonlinear_null.py` — the controls; `fig_*.py` — the manuscript figures.
- `code/paper3/make_figs.py`, `make_si.py`, `verify_numbers.py` — figure/SI builders and the
  independent number verifier.

Paths in the scripts and in `NUMBERS.md` are the authoring-machine absolute paths
(`F:\UNAGI_Project\...`); `rebase_paths.py` in the archive's `code/` directory re-maps them.

## Licence

Code in this archive is MIT (see the top-level `LICENSE`). The TRRUST table in `prior/` is
redistributed from the public TRRUST v2 database (Han et al., 2018) under its own terms; the
manuscript cites the database at `https://www.grnpedia.org/trrust/`.
