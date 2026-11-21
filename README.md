# kappa-C audit and delivered-prior experiment

Code, figure generators, and curated result tables for:

> **Auditing gene-regulatory prior delivery and decoder response in single-cell variational autoencoders**

This repository accompanies the V49 manuscript. It contains:

- `kappa_c_audit/`: the standalone Python package, including the decoder and
  latent spectral audit and its tests.
- `code/`: the core audit and provenance scripts.
- `data/`: curated result tables for the 33-configuration audit and related
  controls.
- `delivery_experiment/`: the 73-fit PG-UNAGI delivered-prior experiment.
- `b1_expansion/prior_delivery/`: the current delivered-prior analysis code.
- `prior_delivery_runs/`: per-arm metadata and result JSON.
- `04_figure_sources/`: manuscript figure generators and the Gaussian and ZINB
  cross-architecture experiments.
- `si/`: number ledgers and SI generation/verification scripts.
- `docs/`: manuscript metadata and the canonical Author Summary.

## Scope

The audit separates four claims that are often conflated:

1. a prior is registered in the model configuration;
2. it reaches the training objective;
3. it changes the latent representation; and
4. it changes reconstruction or downstream utility.

The reported experiments cover one hematopoietic dataset, one curated prior and
one weighting operator, with three training seeds in each replication. The
audit does not certify perturbation readiness.

## Install and use

```bash
pip install kappa_c_audit/dist/kappa_c_audit-0.1.2-py3-none-any.whl
```

```python
from kappa_c_audit import audit

report = audit(Z_vanilla, X_vanilla, Z_prior, X_prior)
print(report["kappa_C_W"], report["verdict"])
```

The package version remains `0.1.2`; `paper1-v49` is the repository/snapshot
release label.

## Data policy

Third-party expression matrices are not redistributed. Small result JSON and
derived arrays used to rebuild the manuscript figures are included according to
the payload variant:

- GitHub payload: source code, metadata, JSON, CSV and Markdown.
- Figshare payload: the same files plus derived `.npz` and `.npy` experiment
  outputs needed for exact figure reconstruction.
- Raw model checkpoints and `.h5ad` expression matrices are excluded.

A few historical JSON files contain Python's non-standard `NaN` token. Parse
those files with Python's `json` module or another permissive JSON reader.

## Reproduction

See `04_figure_sources/*/README.md`, `delivery_experiment/README.md`, and
`si/SI_NUMBERS.md`. Paths recorded in scripts are authoring-machine provenance;
use `code/rebase_paths.py --root <checkout>` to retarget them.

## Citation

Use `CITATION.cff`. Cite the versioned DOI printed on the Figshare record for
this snapshot together with the article.

## Licence

Code is released under the MIT licence. Third-party datasets and TRRUST retain
their original terms.
