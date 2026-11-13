# kappa-C dual-spectrum audit — code and result tables

Reference code and curated result tables for the manuscript

> *A delivery-and-content audit finds gene-regulatory priors present but unused in single-cell VAEs*

This work audits whether a gene-regulatory-network (GRN) prior is actually **used** by a single-cell
variational autoencoder. It ships two things: a **standalone audit tool** (`kappa_c_audit/`) and the
**analysis code + result tables** that produced the reported numbers and supplementary figures.

---

## Layout

| Path | What it is |
|---|---|
| `kappa_c_audit/` | The released Python package (`kappa-c-audit`, MIT). Installs with numpy / scipy / scikit-learn only. Includes `tests/` (12 tests) and `examples/`. |
| `code/` | Analysis scripts that generate the results, tables and supplementary figures, plus the decision-audit tooling (`audit_guard.py`, `latex_silent_linter.py`, `freeze_release.py`). |
| `data/` | Curated result tables as JSON — the processed audit outputs the manuscript reports (per-configuration κ-C, latent-spectrum statistics, null calibrations, perturbation readouts, provenance records). |

## Install

```bash
pip install kappa_c_audit/dist/kappa_c_audit-0.1.2-py3-none-any.whl
# or, from source:
pip install ./kappa_c_audit
```

## Use the audit

```python
from kappa_c_audit import audit

rep = audit(Z_vanilla, X_vanilla, Z_prior, X_prior)
print(rep["kappa_C_W"], rep["verdict"])   # e.g. 1.02  'vanilla-equivalent'
```

`audit()` returns a plain `dict` (not an object): `kappa_C_W`, `kappa_C_Z`, `verdict`,
`band`, `C_percent` and the per-gene readouts all come back as keys, so a report can be
serialised with `json.dump` without a conversion step.

`audit()` reports whether a model's **latent→expression** map was reorganised by the prior. It does
**not** certify perturbation-readiness — that requires the four-check prior-activity protocol
(gradient check, prior-strength scan, structural bound, injection-path check) described in the
manuscript, and is explicitly **not** claimed for prior-injected models.

## The equivalence band (read this before interpreting κ)

The released tool uses the **predictability-form** vanilla-equivalence band

```
DEFAULT_BAND = (0.937, 1.064)      # G-VAE hematopoietic null: κ = 1.001 ± 0.032
```

over 15 G-VAE hematopoietic vanilla seeds (210 ordered seed-pair comparisons, corrected statistic).
Early releases (≤ 0.1.1) carried the superseded **trace-form** band `[0.78, 1.23]`; that band is
retained only as a documented counter-example — it inflates the null SD by ≈56× — and must **not**
be used for classification. The band-independent deep-compression verdict (`κ < 0.5`) holds under
either choice. See `CHANGELOG.md` for the full correction history.

## Reproducibility notes

- **Data are not bundled.** The expression matrices are third-party single-cell datasets; scripts
  resolve them through a single data-quality-gated registry (`code/run_registry.py`), so the data
  root is configured in **one** place rather than in each script. Point it at your local copy of the
  datasets before re-running anything end-to-end.
- **Absolute paths in `code/` are recorded provenance.** 72 of the 136 analysis scripts carry the
  author's local session directory (e.g. `C:/Users/12901/WorkBuddy/2026-09-03-18-47-34/...`).
  They are shipped unchanged on purpose: these are the scripts that produced the reported numbers,
  and rewriting them would erase the record of where they ran. To point them at your own checkout,
  run `python code/rebase_paths.py --root /your/checkout --apply`. It replaces **only the path
  prefix**, defaults to a dry run, and backs up every file it touches into `code/_rebase_backup/`.
  Verify with `python code/rebase_paths.py --verify` (expect 0). Rebasing makes the scripts
  addressable on your machine; it does not recreate the raw expression matrices, which are
  third-party datasets and are not bundled (see the note above).
- **`data/` holds results, not inputs.** Every JSON here is a processed audit output; nothing in it
  is required to *install* or *use* the tool.
- Some scripts in `code/` are deliberately **provenance / correction** scripts (e.g.
  `fix_methods_band.py`, `*_recompute*`). They are kept on purpose: they document the old→new
  transitions and are part of the evidence chain. Do not read them as current definitions.
- Figure PDFs/PNGs are distributed with the manuscript package, not here; they are regenerable from
  the generator scripts in `code/`.

## Citation

See `CITATION.cff`.

## Releasing

This repository is the *live* copy; the citable snapshot lives on figshare. The two move together,
in one sitting:

1. **Publish the GitHub repository**, then tag the release `v0.1.2` so the tag, the wheel and
   `CITATION.cff` agree.
2. **Upload the snapshot to figshare by hand.** There is no GitHub→figshare auto-archiving hook (the
   thing Zenodo had), so the zip assembled in the manuscript package is uploaded and **Published**
   explicitly. Then take the **versioned** DOI, `10.6084/m9.figshare.<id>.v1` — the unversioned form
   always resolves to the newest version and is the wrong thing to cite for a reproduction.
3. **Fill both values once** — `\RepositoryURL` and `\ArchiveDOI` in `submission_metadata.tex`, and
   the `[project.urls]` Homepage in `kappa_c_audit/pyproject.toml` — then **rebuild the wheel**: the
   URL is captured in the wheel's `METADATA` at build time, so editing `pyproject.toml` alone does
   not change the shipped artefact.

`CITATION.cff` deliberately carries no `repository-code` field until step 1 is done: a literal
placeholder there would be published verbatim.

## License

MIT — see `LICENSE`.
