# Prior delivery experiment — does a prior that reaches the loss change the spectra?

`paper1` established that in PG-UNAGI the TRRUST prior is built, written to a tensor,
and then never read: the `geneWeight` layer that would carry it is all zeros in every
released stage file (1,116 files, nnz = 0), so the reconstruction term is unweighted.
The obvious next question is the one the paper deliberately left open:

> if the prior were actually delivered, would the dual-spectrum diagnostic move?

This directory answers it with a paired four-arm run. It is the "delivery" arm that
paper1 could not run.

## The wiring under test

Three breakages were found and fixed before this experiment could run (see
`../PRIOR_WIRING_FIX.md`):

1. `pg_unagi/patcher.py` Patch 3 used an unbounded regex and corrupted `UNAGI_tool.py`
   (`self.iDREM_parameters` → `self.iDREM\n self._apply_tf_prior_weights()_parameters`).
   Restored from `UNAGI_source.bak/`, patch disabled.
2. The apply function was `pass` and the boosting helper had no call site.
   `UNAGI_tool.run_UNAGI` now hands the prior to the runner, and
   `runner.update_gene_weights_table()` applies it to `stagedata/geneWeight`.
3. That layer was all zeros to begin with, so multiplying was a no-op. The apply step
   now *seeds* the layer from the prior when it is empty.

The prior reaches the loss through `trainer.train_model`:
`geneWeight → transfer_to_ranking_score → reconstruction_loss(gene_weights=...)`.

## Why the arms are not "raw vs sqrt"

`transfer_to_ranking_score` (`utils/attribute_utils.py:49`) turns the per-gene weight into
`1 + 1/sqrt(rank)`. **Only the ordering survives.** Any monotone rescaling of the prior —
sqrt, log1p, min-max — is therefore training-identical, and the worry that
`1.5**9 = 38.4` "blows up" the loss is misplaced: 38.4 and 1.5 differ only by rank, and
only for genes with different TF in-degree.

So the degrees of freedom that can actually change training are:

| arm | what it changes | what it tests |
|---|---|---|
| `empty` | no prior at all (`disable_tf_prior`) | the paper1 condition: `geneWeight` ≡ 0 |
| `graded` | real TRRUST prior, weight `1.5**(in-degree)` | does delivery move the spectra |
| `binary` | same 203 genes, flat weight | does the *grading* matter, or only the support |
| `shuffle` | real multiset of boost values on random genes | the content control: same support size, same in-degree distribution, different genes |

`shuffle` is paper1's degree-preserving content control, now applied to a prior that is
actually delivered.

## Protocol, and where it departs from paper1

Run to run the same cells, the same seed, the same schedule; only the prior differs.

| | paper1 | here | why |
|---|---|---|---|
| cells | 32,100 (all) | 2,400/stage = 12,000 | one epoch costs O(N²) at fixed batch size (the GCN runs the whole graph per mini-batch). At 32k an epoch is ~3.7 min; 4 arms × 45 epochs would not finish. |
| CPO | on | **off** | CPO spends ~6 min/arm choosing a per-stage resolution and still produced degenerate graphs on a subsample. Resolution is fixed at 1.0 instead: less variable, and clustering is not the object of study here. |
| edge cutoff | 0.05 | **0.30** | `connectNodes` keeps a stage pair only if its nearest-cluster distance falls below the cutoff in the left tail. On 12k cells too few pairs qualify and `buildEdges` raises `No edges found`. |
| epochs | 300 + 4×100 | 30 + 15 | compute, as above |
| iDREM | on | on | kept, so the pipeline is otherwise unchanged |

These deviations cost comparability with the *published* κ-C values. They do not cost the
internal validity of the comparison this experiment makes: every arm shares them.

Two consequences to keep in mind when reading the numbers:

- **The prior only acts during the iterative phase.** `trainer.train_model` is called
  with `geneWeights=geneWeights if is_iterative else None`, so iteration 0 (the
  30-epoch initial pass) trains unweighted and is byte-identical across arms up to
  float rounding; iteration 1 (15 epochs) is the only weighted pass, and it is the
  model that gets audited. Every arm therefore shares a 30-epoch common ancestor and
  differs only by 15 epochs of prior-weighted refinement. That is internally valid —
  it is the same 15 epochs in every arm — but it is the budget in which the effect is
  measured, and `--epoch-iter` is the cheapest lever on it.
- With 45 epochs the models are far from the 700-epoch schedule of the manuscript.
  A null result here is weaker evidence than a null at convergence.
- Fixed resolution 1.0 produces fewer clusters than CPO, and iDREM sometimes finds no
  trajectory spanning all five stages. `getPosNegMaxChangesNodes` used to crash on that
  (`argmax of an empty sequence`); it now skips such files. Since the gene weights are
  seeded from the prior, the ENCODE/iDREM contribution — which is all zeros on this
  dataset anyway — is not load-bearing.

## Files

- `arms.py` — the four priors, plus `ranking_profile()` which reports the effective
  weight the trainer actually sees (not the raw boost factor).
- `run_arm.py` — one arm: subsample, train two iterations, extract latents, save
  `latent.npz` + `arm_meta.json`. `--weight-mode direct` bypasses the rank transform
  and uses the stored value as the per-gene multiplier (see Part 2 of the results).
- `audit_arms.py` — dual-spectrum audit of each arm against a chosen reference.
- `run_all.py` — driver for the four-arm contrast, `empty → graded → binary → shuffle`.
- `run_ladder.py` — dose ladder in direct mode (`d1/d2/d5/d20`).
- `run_seeds.py` — repeats the four-arm contrast for extra seeds and aggregates;
  seed 42 is reused from `r2` rather than retrained.
- `run_support.py` — support ladder in direct mode (`k203/k1000/k2027`), widening a
  real TRRUST prior while rescaling so total reconstruction mass stays fixed.
- `pergene_fit.py` — paired per-gene decomposition of the readout: `R2_g(real) −
  R2_g(shuffled)`, with the genome-wide median shift removed.
- `run_dose.py` — the Part 6 dose-response: real vs shuffled at 20×, three seeds,
  six arms. Skips any arm whose `latent.npz` already carries the requested level.
- `dose_summary.py` — reads the 5×/20× pair off disk and applies the decision rule
  registered in Part 6 before the 20× arms were trained.
- `pergene_robust.py` — the baseline-R² controls for the per-gene effect: decile
  matching and an OLS residual, so that "easier genes gain more" is ruled out.
- `run_empty_e45.py` — the matched no-prior reference for the 3×-budget arms. Verifies
  `arm`, `epoch_initial` and `epoch_iter` from `arm_meta.json` before reusing anything,
  because `run_dose.py` leaves an e15 copy of `empty` in every seed directory.
- `pergene_vs_baseline.py` — the absolute per-gene test: each arm against the no-prior
  model, so "real beats shuffled" can be checked against "does it beat nothing".

## Reproduction

```bash
export PATH="/c/Program Files/Eclipse Adoptium/jdk-17.0.10.7-hotspot/bin:$PATH"
py -3.13 -u run_all.py --root F:/UNAGI_Project/prior_delivery_runs/r1 \
    --per-stage 2400 --epoch-initial 30 --epoch-iter 15 --max-iter 2 \
    --edge-cutoff 0.3 --cpo off
```

Results land in `<root>/audit_results.json` and are summarised in
`PRIOR_DELIVERY_RESULTS.md`. The project is closed; start with
**`PRIOR_DELIVERY_CLOSING.md`** — verdict, the auditable numbers, an arithmetic
bound on why an aggregate spectrum diagnostic cannot see a support-limited prior,
the six wrong headlines this project produced, and what to do next.
