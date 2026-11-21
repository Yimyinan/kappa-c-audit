# Prior delivery: does a prior that reaches the loss change the spectra?

Run `prior_delivery_runs/r2`, 2026-10-26. Four arms, 12,000 cells (2,400/stage),
45 epochs (30 + 15), fixed resolution 1.0, seed 42. Only the prior differs.

## Headline

**The prior now reaches the loss — verified on disk — and the decoder spectrum
still does not compress.** κ-C(W) = 0.9825 (C = 1.75%) against the no-prior arm,
nowhere near the 50% the deep-compression band requires. The latent spectrum moves
instead, by 4–7% in trace, in the *expansion* direction.

Two things came out of the run that are sharper than the run itself:

1. **The dose is capped structurally, not by the boost factor.** This injection path
   converts weights to `1 + 1/sqrt(rank)` before they reach the loss, so the
   per-gene ratio cannot exceed 1.98 whatever the raw factor is. Escalating the
   factor changes nothing; only the number of genes the prior covers does.
2. **The latent response is already saturated at a 1.09× dose** — a 1.09× arm and a
   1.98× arm land at the same κ-C(Z). Above ~1.1× there is nothing left to buy.

## What was verified to be delivered

`geneWeight` in every stage file, counted from disk after training:

| arm | nnz per stage file | cells × genes |
|---|---|---|
| `empty` | 0 | — (paper1's released condition) |
| `graded` | 487,200 | 2,400 × 203 |
| `binary` | 487,200 | 2,400 × 203 |
| `shuffle` | 487,200 | 2,400 × 203 |

203 genes is the same figure paper1 reports as "203 per-gene scalars". The layer the
released pipeline leaves all-zero is now populated exactly where the loss reads.

## Results

Reference arm = `empty` (same architecture, cells, seed and schedule; `geneWeight ≡ 0`).

| arm | κ-C(W) | C % | κ-C(Z) | eff. rank | max per-gene ratio | genes boosted |
|---|---|---|---|---|---|---|
| `empty` | 1.0000 | 0.00 | 1.0000 | 3.379 | 1.00 | 0 |
| `graded` | **0.9825** | **1.75** | **1.0671** | 3.459 | 1.98 | 203 |
| `binary` | 0.9940 | 0.60 | 1.0702 | 3.285 | 1.09 | 203 |
| `shuffle` | 0.9991 | 0.09 | 1.0388 | 3.320 | 1.98 | 203 |

The `max per-gene ratio` column is the quantity that actually enters the loss, not
the raw boost factor; see the next section for why they differ.

### 1. Delivery is not sufficient for the decoder null

A 1.75% compression is noise against a 50% threshold. Whatever keeps the decoder
spectrum flat in paper1's 33 configurations, it is not only the unread tensor: with
the prior genuinely inside the reconstruction term, the decoder still does not
compress. This closes the obvious objection to paper1 ("you never delivered it")
with a direct answer.

### 2. The latent spectrum responds, and it expands

κ-C(Z) > 1 in all three prior arms: 1.067 / 1.070 / 1.039. paper1's latent-side
finding was *compression* (effective rank falling in 17/18 hemato configurations);
here the delivered prior does the opposite, modestly. Not contradictory — different
reference, different dose — but "delivering the prior compresses the latent space"
is not a safe extrapolation.

### 3. Dose saturates below 1.1×; the gene set is what moves the latent

`graded` (max ratio 1.98) and `binary` (max ratio 1.09) land at 1.0671 and 1.0702 —
the same within any plausible noise. So an 80% cut in dose changes nothing: the
response is **saturated at the lower dose**. There is no gradient to exploit between
1.1× and 2×.

`shuffle` carries the *same* dose as `graded` (identical score distribution, max
1.98) but assigns it to random genes, and lands at 1.0388. So at matched dose and
matched support size, real genes give 1.067–1.070 and random genes give 1.039.

Read together: the ordering is

```
real genes  (1.067, 1.070)  >  random genes at same dose  (1.039)  >  no prior  (1.000)
```

which points at **content** rather than dose or degree.

**But Part 2 killed this reading.** A run that should be a null replicate — the same
model with no per-gene weighting, reached through the other weighting mode — landed at
κ-C(Z) = 1.0386. That is the same size as the graded-vs-shuffle gap. So the ordering
above is not resolvable at one seed per arm, and 1.067 vs 1.039 should be read as
indistinguishable. See "The null replicate" in Part 2.

"What content matters" was the weakest claim in this report when it was written, and it
has since failed its own control. It stays in the record as a hypothesis that this
design could not test, not as a finding.

## The dose is capped by the ranking, not by the factor

`trainer.train_model` turns each cell's gene-weight row into `1 + 1/sqrt(rank)`
before it becomes `gene_weights` in the reconstruction term
(`utils/attribute_utils.py:49`). Two consequences, both exact:

- **The raw boost factor is decorative.** Any monotone transform — `1.5**k`,
  `sqrt(1.5**k)`, `log1p`, a plain ranking — produces an identical training run,
  because only the *order* of the weights survives. The `1.5**9 = 38.4` that looked
  alarming never reaches the loss as a magnitude. This also dissolves the
  "raw vs sqrt" question I raised earlier: those arms are the same arm.
- **The per-gene ratio is bounded at 1.98.** Top rank scores
  `1 + 1/sqrt(1) = 2.0`; everything tied below scores `1 + 1/sqrt(N)`, which for
  N = 20,527 is 1.00982. So the maximum expressible ratio is 2.0/1.00982 = 1.98,
  whatever the boost factor.

And a corollary that explains the `binary` arm: **ties are expensive.** Give all 203
genes the same weight and they share the average top rank, so each scores
`1 + 1/sqrt(102) = 1.099` instead of up to 2.0 — a dose of 1.09× rather than 1.98×.
Flattening a prior to escape the grading also flattens its dose by nearly half.

Which means the only lever on dose is **how many genes the prior covers**, not how
strongly it weights them. In the `graded` arm the 203 boosted genes carry
**1.10% of the total reconstruction weight**. That is the measured form of the "dose
ceiling" paper1 could only infer: the ceiling is structural, and it sits two orders
of magnitude below a reweighting that could plausibly reshape a decoder.

It also explains why paper1's Check 2 λ scan came back flat. Raising λ scales the
prior *before* the ranking, and scaling does not change ranks.

## Caveats that bound these numbers

- **45 epochs, not 700.** Effective rank is 3.4 of 64 latent dimensions — the model
  uses very few dimensions, itself a sign of limited training. Absolute κ-C values
  are not comparable to the manuscript; only the between-arm contrast is, since all
  four arms share the budget.
- Fixed resolution 1.0 instead of CPO, edge cutoff 0.30 instead of 0.05, 12,000 of
  32,100 cells. Reasons in `README.md`; all shared by every arm.
- iDREM is skipped and UNAGI's ENCODE-derived gene weights are replaced by the prior
  (`UNAGI_runner._write_gene_weights_from_prior`). On this dataset that route writes
  zeros anyway, and on a subsampled stage graph it crashes.
- One seed per arm. No interval on any of the four numbers.

## What this changes in the paper

paper1 says the prior never reaches the objective, so the decoder null cannot be read
as prior inertness. That survives, and gains a sharper second half: **when the prior
is delivered, the decoder still does not compress — and the delivery mechanism caps
any gene's influence at ~2×, with the response already saturated near 1.1×.** The
0/33 null is attributable to the injection path twice over: the tensor was unread,
and the path that would read it is dose-bounded by construction.

## Next, if this is taken further

1. **Three seeds per arm** (12 runs, ~6 h at this budget) to put an interval on the
   0.028 graded-vs-shuffle gap. Until then the content effect is a hypothesis.
2. **Replace the rank transform** with a direct multiplicative weight on the
   reconstruction term. That is the only way to test a dose above 2×, it is a
   one-function change, and it makes a falsifiable prediction: if the ceiling story
   is right, compression should appear only once the per-gene ratio clears ~2×.
3. **Sweep the support, not the factor**: 203 → 1,000 → 5,000 genes at 2×. By the
   analysis above this is the only dose axis that moves, so it is the informative one.

## Reproduce

```bash
export PATH="/c/Program Files/Eclipse Adoptium/jdk-17.0.10.7-hotspot/bin:$PATH"
cd b1_expansion/prior_delivery
py -3.13 -u run_all.py --root F:/UNAGI_Project/prior_delivery_runs/r2 \
    --per-stage 2400 --epoch-initial 30 --epoch-iter 15 --max-iter 2 \
    --edge-cutoff 0.3 --cpo off
py -3.13 -u audit_arms.py --root F:/UNAGI_Project/prior_delivery_runs/r2
```

Per-arm wall time 1,760–1,780 s (45 epochs at ~34 s/epoch, plus ~3 min of setup).

---

# Part 2 — the dose ladder above the rank ceiling

Run `prior_delivery_runs/ladder1`, 2026-10-27. Same cells, seed, budget and support
as Part 1 (203 genes); the only change is that the stored value is used **directly**
as the per-gene multiplier on the reconstruction term
(`trainer.train_model`, `direct_gene_weights`), so the rank ceiling of ~1.98 no
longer applies.

| arm | per-gene factor | κ-C(W) | C % | κ-C(Z) | eff. rank | PC1 share |
|---|---|---|---|---|---|---|
| `d1` | 1 | 1.0000 | 0.00 | 1.0000 | 3.527 | 0.479 |
| `d2` | 2 | 0.9924 | 0.76 | 0.9874 | 3.336 | 0.507 |
| `d5` | 5 | 0.9977 | 0.23 | 1.0158 | 3.739 | 0.463 |
| `d20` | 20 | 0.9941 | 0.59 | 0.9877 | 3.979 | 0.451 |

Delivery is again confirmed on disk: `geneWeight` nnz = 487,200 per stage file in
every dosed arm (2,400 cells × 203 genes), 0 in `d1`.

## The prediction failed

Part 1 ended with a falsifiable claim: if the dose-ceiling reading is right,
compression should appear once the per-gene ratio clears ~2×. It does not.

**κ-C(W) stays inside 0.992–1.000 across a 20-fold range of dose**, i.e. C ≤ 0.76%
at every rung against a 50% threshold. There is no threshold, no knee, and no trend.
Twenty-fold the weight on those genes does nothing measurable to the aggregate
decoder readout.

κ-C(Z) is not monotone either — 0.987, 1.016, 0.988 — and all three sit inside the
null band established below.

What *does* move is the shape of the latent, weakly and in the anti-compression
direction: effective rank climbs 3.53 → 3.34 → 3.74 → 3.98 and the leading
component's share of latent variance falls 0.479 → 0.451 as the factor rises. So
heavier weighting on 1% of genes, if it does anything, *spreads* the latent code
rather than compressing it. One seed, so this is a direction, not a number.

## The null replicate — and what it costs Part 1

`d1` (direct mode, factor 1) and Part 1's `empty` (rank mode, uniform 1.0098) are the
same model in intent: no per-gene weighting. They differ only in the global scale of
the reconstruction term, by 1%.

Auditing one against the other:

```
κ-C(W) = 0.9946      κ-C(Z) = 1.0386
```

**A null replicate moves κ-C(Z) by 3.9% and κ-C(W) by 0.5%.** Everything measured in
Part 1 lives inside that band: 1.067, 1.070, 1.039 against a reference of 1.000. The
"latent responds by 4–7%" reading, and the "real genes beat random genes by 0.028"
reading that rested on it, are both within the movement a no-effect run produces.

This is the most important number in either part, and it is a negative one: at this
budget, on this dataset, **κ-C(Z) has a noise floor of order 4%**, and every latent
claim made here has been at or below that floor.

## Why the factor was the wrong lever

The support is 203 of 20,527 genes — 1.1% of the reconstruction weight mass
(Part 1). κ-C(W) is an aggregate over all genes: `sum_g R2_g(arm) / sum_g R2_g(ref)`.
Reweighting 1% of the sum, however hard, cannot move an aggregate dominated by the
other 99%. The ladder confirms the arithmetic rather than refuting it.

So the conclusion from the structural analysis gets a measured endorsement: **the
only dose axis that can matter is how many genes the prior covers, not how strongly
it weights them.** `d20` at 203 genes is the same experiment as `d2` at 203 genes.

## What this changes

Part 1's headline survives and gets stronger: delivering the prior does not compress
the decoder. Part 1's secondary claim does not survive; the latent movement is at the
noise floor. And the dose-ceiling reading sharpens from "the path is capped at 2×" to
"even 20× on a 1% support does nothing, because the ceiling is the support."

The honest one-line summary of both parts:

> Handing the prior to the loss changes nothing measurable, at any dose this
> mechanism can express, because the mechanism only ever touches 1% of the
> reconstruction term.

## Next

1. **Support ladder**: 203 → 1,000 → 5,000 genes at a fixed 5×. This is now the only
   informative axis. Prediction: the aggregate κ-C(W) only starts to move when the
   support reaches a few percent of genes.
2. **Three seeds** for the four-arm contrast, to state the noise floor properly rather
   than infer it from one cross-mode comparison. Running as `prior_delivery_runs/seeds`.
3. Anything that wants to move a decoder has to weight a large fraction of the
   transcriptome, which is no longer a "prior" in any useful sense — that, rather
   than the tensor wiring, may be the real limit.

## Reproduce (Part 2)

```bash
cd b1_expansion/prior_delivery
py -3.13 -u run_ladder.py --root F:/UNAGI_Project/prior_delivery_runs/ladder1 \
    --levels 1 2 5 20 --per-stage 2400 --epoch-initial 30 --epoch-iter 15 \
    --max-iter 2 --edge-cutoff 0.3 --cpo off --seed 42
```

Per-arm wall time 1,617–1,629 s.

---

# Part 3 — three seeds, and the ordering does not replicate

Run `prior_delivery_runs/seeds`, 2026-10-27. Seeds 43 and 44 trained all four arms
anew; seed 42 is the Part 1 run reused rather than retrained. Same cells, budget and
edge cutoff as Part 1. `κ` is computed against each seed's own `empty` arm, so the
reference moves with the seed.

| arm | κ-C(W) mean ± sd | range | κ-C(Z) mean ± sd | range |
|---|---|---|---|---|
| `empty` | 1.0000 ± 0.0000 | — | 1.0000 ± 0.0000 | — |
| `graded` | 0.9971 ± 0.0131 | [0.9825, 1.0079] | 1.0235 ± 0.0378 | [1.0000, 1.0671] |
| `binary` | 0.9966 ± 0.0036 | [0.9940, 1.0007] | 0.9887 ± 0.0706 | [0.9464, 1.0702] |
| `shuffle` | 0.9989 ± 0.0017 | [0.9972, 1.0005] | 1.0585 ± 0.0638 | [1.0078, 1.1301] |

Per seed:

| seed | `graded` | `binary` | `shuffle` | latent eff. rank (`empty`) |
|---|---|---|---|---|
| 42 | 1.0671 | 1.0702 | 1.0377 | 3.379 |
| 43 | 1.0000 | 0.9497 | 1.1301 | 2.414 |
| 44 | 1.0034 | 0.9464 | 1.0078 | 2.518 |

## 1. The decoder null now has an interval

κ-C(W) = **0.9966–0.9971 ± 0.013** across arms, with `graded` reaching 1.0079 on one
seed — *above* 1, i.e. the prior arm's decoder is marginally *more* predictable than
the no-prior arm. In compression terms: C = 0.3% ± 1.3 percentage points, range −0.8%
to +1.8%, against a 50% threshold. There is no reading of these numbers that supports
compression.

This is the one Part 1 claim that replication strengthens.

## 2. The latent ordering reverses between seeds

Part 1, on seed 42 alone, ranked the arms `graded` ≈ `binary` > `shuffle`. Across seeds:

- **seed 42** — `binary` (1.070) ≈ `graded` (1.067) > `shuffle` (1.038)
- **seed 43** — `shuffle` (1.130) > `graded` (1.000), `binary` well below 1 (0.950)
- **seed 44** — `shuffle` (1.008) > `graded` (1.003), `binary` below 1 again (0.946)

`shuffle` goes from last to first. `graded`'s 1.067 — the number Part 1 leaned on — is
its *maximum* across seeds; on the other two seeds it is 1.000 and 1.003, i.e. exactly
the no-effect value. The between-seed sd (0.038–0.071) is larger than every
between-arm difference in Part 1, and the reference arm is fixed by construction, so
this is not a reference artefact: it is model-level seed noise.

The `binary` arm is the only one with a consistent sign at two of three seeds
(0.950, 0.946 — a slight compression), but with seed 42 at 1.070 the spread swamps it.

**Verdict: no reproducible effect of the prior on either spectrum.** Part 1's
`graded`-vs-`shuffle` ordering, and the "content matters" reading built on it, do not
survive replication. They were seed noise, exactly as the null replicate in Part 2
predicted before these runs existed.

## 3. Seed 42 was not a representative seed

The latent effective rank of the `empty` arm is 3.379 at seed 42 but 2.414 and 2.518 at
seeds 43 and 44. The seed-42 models occupy a markedly higher-dimensional latent than
the other two. Every absolute number in Part 1 came from that seed.

This does not invalidate the Part 1 contrast — all four arms shared the seed — but it
does mean Part 1's numbers describe one draw from a wide distribution, and it is worth
saying plainly: the single-seed design, not the prior, was the fragile part.

---

# Part 4 — the support ladder, and the second prediction fails too

Run `prior_delivery_runs/support`, 2026-10-27. Parts 1–3 left one lever standing: the
prior covered 203 of 20,527 genes, so the aggregate decoder readout could not move by
construction, whatever the dose. Part 4 widens the support while holding the factor at
5×.

Only **2,027** of the 20,527 genes in this dataset are targets of any TRRUST TF (735
TFs, 7,731 edges in the data), so rungs are 203 → 1,000 → 2,027, all real GRN content
rather than a real core padded with random genes. **5,000 was not available and would
have required inventing coverage that does not exist.**

Weights are rescaled to keep the total reconstruction mass fixed, so the manipulation
is *concentration*, not a global loss rescale. Verified on disk per rung
(nnz = 2,400 × k exactly):

| rung | support | share of genes | weight on selected | baseline weight |
|---|---|---|---|---|
| `k203` | 203 | 0.99% | 4.810 | 0.962 |
| `k1000` | 1,000 | 4.87% | 4.185 | 0.836 |
| `k2027` | 2,027 | 9.87% | 3.584 | 0.717 |

Three seeds each; each seed uses its own already-trained `empty` arm as the reference.

| arm | support | κ-C(W) mean ± sd | κ-C(Z) mean ± sd | eff. rank |
|---|---|---|---|---|
| `empty` | 0 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 2.770 |
| `k203` | 203 | 0.9974 ± 0.0097 | 0.9784 ± 0.0520 | 2.869 |
| `k1000` | 1,000 | 1.0014 ± 0.0045 | 1.0318 ± 0.1387 | 3.059 |
| `k2027` | 2,027 | 1.0004 ± 0.0072 | 1.0848 ± 0.0771 | 2.991 |

## The decoder null survives a ten-fold increase in support

κ-C(W) is **1.000 across the whole range**, from zero coverage to 9.87% of the
transcriptome. The widest rung weights 2,027 genes at 3.58 while down-weighting the
rest to 0.72 — a five-fold concentration over a tenth of the likelihood — and the
aggregate decoder readout moves by **+0.04%**.

The per-seed values say the same thing: every one of the nine dosed runs lands
between 0.9867 and 1.0081. Several are above 1, i.e. the prior arm's decoder is
marginally *more* predictable, which is the opposite sign from compression.

**The Part 2 prediction — "κ-C(W) should start to fall once the support reaches a few
percent" — is falsified.** So is the structural argument that motivated it. Both
levers have now been tried: dose to 20×, support to 10%. Neither moves the decoder.

## The latent spectrum still has no reproducible trend

κ-C(Z) drifts upward with support at the mean (0.978 → 1.032 → 1.085), which is the
one thing in Parts 1–4 that looks like a trend. It should not be read as one:

- `k1000`'s sd is **0.139**, four times its deviation from 1, with per-seed values of
  1.150, 0.879 and 1.066.
- `k2027` splits 1.113 / 0.998 / 1.144 — two of three well above 1, one exactly at it.
- The drift is not monotone: `k203` sits *below* 1 (0.978), so the series is
  0.978 → 1.032 → 1.085 across a 10× support change while jumping ±0.14 between seeds
  at fixed support.

With n = 3 and between-seed sd of 0.05–0.14, the 95% interval on `k2027` spans roughly
0.90–1.27. A directional hint at best; not a measured effect. If it were real, the
effect size would be ~8% latent expansion for a five-fold concentration over 10% of
genes — i.e. the latent moves when the loss stops being nearly uniform, not when the
prior's content changes.

---

# What the four parts say together

| question | answer | evidence |
|---|---|---|
| does the prior reach the loss? | **yes**, verified on disk (nnz = 2,400 × k per stage) | Parts 1, 2, 4 |
| does delivering it compress the decoder? | **no** — κ-C(W) = 0.997 ± 0.013, C ≈ 0.3% vs a 50% threshold | Part 3 |
| does more dose help? | **no** — flat from 1× to 20× | Part 2 |
| does more support help? | **no** — flat from 0.99% to 9.87% of genes | Part 4 |
| does the latent spectrum respond? | **not reproducibly** — between-seed sd ≥ every between-arm gap; orderings reverse | Parts 3, 4 |
| does the prior's *content* matter? | **not measurable with this design** | Parts 3, 4 |

The one-line result:

> A GRN prior delivered into the reconstruction term of a single-cell VAE moves neither
> spectrum by more than seed noise — at any dose up to 20×, and at any support up to
> 10% of the transcriptome.

That is a sharper form of paper1's conclusion than paper1 could make, and it is the
opposite of the direction the argument was expected to go. paper1 said the prior never
reached the objective, so the 0/33 null could not be read as prior inertness. True —
and now also the smaller half of the story. **Even when it reaches the objective,
delivery does not move the diagnostic.** The bottleneck is neither the wiring between
tensor and loss nor the size of the prior; it is that a reconstruction-term
reweighting, at any concentration this pipeline can express, is not the mechanism by
which a GRN prior would act on a VAE's latent geometry.

## What this means for the two papers

For **paper1**, the delivery story is now closed in both directions, which is stronger
than leaving it open: the tensor was unread, *and* delivering it changes nothing. The
"confound we could not rule out" paragraph can be replaced by a measured statement.
The cost is that the paper loses its most attractive follow-up hook.

For **paper2**, nothing changes. The benchmark results are about measurement, not about
delivery, and Part 4 does not touch them.

## The methodology that made the difference

One cheap control did more than any of the training runs: a **same-intent replicate**
(the same model with no per-gene weighting, reached through the other weighting mode)
gave κ-C(Z) = 1.0386 against itself. That single number predicted, before the seed
replication existed, that Part 1's 0.028 "content effect" was noise. It was right.

Any future single-seed κ difference should have its noise floor measured this way
first. It costs half an hour; it would have saved this project two wrong headlines.

## Reproduce (Part 4)

```bash
cd b1_expansion/prior_delivery
py -3.13 -u run_support.py --root F:/UNAGI_Project/prior_delivery_runs/support \
    --seeds 42 43 44 --k 203 1000 2027 --level 5 --per-stage 2400 \
    --epoch-initial 30 --epoch-iter 15 --max-iter 2 --edge-cutoff 0.3 --cpo off
```

27 minutes per arm (nine arms).

## What is worth doing next, if anything

The two obvious ladders have both been run and both are flat. What remains is not more
of the same.

1. **Stop reweighting the reconstruction term and change the mechanism.** Every arm
   here — rank form, direct form, 20× dose, 10% support — is a per-gene multiplier on
   `log_prob`. If that operator cannot move the decoder, the next test is a different
   operator: add the prior as a penalty on decoder Jacobian structure, or as a
   constraint on the latent covariance, rather than as a likelihood weight. That is a
   model change, not an experiment, and it is the only remaining way to ask whether
   *any* GRN prior can shape a VAE's geometry.
2. **Report the noise floor with every κ**, not just the point estimate. The Part 3
   and Part 4 tables are the form these numbers should always have been in. A single κ
   without a between-seed sd is not interpretable — this project produced four
   headlines that a 30-minute control would have prevented.
3. **If the field is going to keep reporting prior-injection gains, the denominator
   matters more than the numerator.** Parts 2–4 say the effect of a pathway-sized
   prior on a reconstruction-term weight is below seed noise. Any reported gain from
   adding a prior therefore needs a same-intent replicate control before it is
   attributed to the prior — the same lesson paper1 argues for content, applied one
   level up.
4. **Do not extend this ladder further.** 5,000 genes is not available from real
   TRRUST coverage in this dataset, and inventing coverage to fill the rung would test
   a prior nobody would use. The boundary is informative: 2,027 genes is already 10%
   of the transcriptome, which is far past any plausible definition of "a pathway".

---

# Part 5 — the per-gene test: a signal, in the right direction, not resolved

Run `prior_delivery_runs/support/s{42,43,44}`, 2026-10-27. Every earlier part measured
an *aggregate*, and Parts 2–4 showed the aggregate cannot move because the prior touches
1% of the sum. But a flat aggregate can hide parts moving in opposite directions. So the
last question is the one the mechanism actually operates at: **per gene.**

## Design

Matched on everything except gene identity:

| | real arm (`k203`) | shuffled arm (`shuffleflat`) |
|---|---|---|
| support | 203 curated TRRUST targets | 203 uniformly random genes |
| weight on support | 5.0, rescaled to 4.810 | 5.0, rescaled to 4.810 |
| baseline weight | 0.962 | 0.962 |
| cells / seed / schedule / resolution | 12,000 / 42-43-44 / 45 epochs / 1.0 | identical |

The two sets share one gene, so 202 genes are exclusive to each. The comparison is paired
on gene: `d_g = R2_g(real) - R2_g(shuffled)`.

`R2_g` is that gene's ridge-readout R² from the latent — the exact summand of
κ-C(W), computed with the same standardise-per-column, `Ridge(alpha=1, no intercept)`
estimator. Verified: `sum(R2_g)` reproduces the reported aggregate to six digits
(0.986713 against κ-C(W) = 0.9867 for seed 42 / `k203` vs `empty`).

Two controls that the earlier parts established as necessary:

- **The genome-wide shift is subtracted.** Adding any prior moves almost every gene a
  little, so each subset is reported as its mean `Δ` minus the median `Δ` over all genes.
  Without this, a subset-vs-nothing comparison would just re-measure the global drift.
- **The per-gene t is not a p-value.** Genes are strongly co-expressed, so `n = 202` is
  not 202 independent observations. It is reported as a shape, and the inference rests on
  the sign and size across seeds instead.

## Results

Mean `ΔR²` after removing the genome-wide shift:

| seed | real targets | shuffled targets | difference | frac. improved (real / shuffled) |
|---|---|---|---|---|
| 42 | **+0.00265** | −0.00065 | **+0.00330** | 0.535 / 0.401 |
| 43 | +0.00029 | +0.00031 | −0.00002 | 0.495 / 0.426 |
| 44 | **+0.00850** | +0.00301 | **+0.00549** | 0.639 / 0.495 |

Across seeds the difference-of-differences is **+0.00293 ± 0.00277** (mean ± s.d.,
n = 3), positive in **2 of 3** seeds, `t(df=2) = 1.83`. With two degrees of freedom that
does not reach significance, and it is not claimed to.

The aggregate ratio for the same pairs is 1.0009, 1.0015 and 1.0101 — i.e. the total
predictability is unchanged. What the table shows, where it shows anything, is a
**redistribution**: the prior's own genes gain predictability relative to a
capacity-matched random set, while the total stays put. That is the shape the mechanism
predicts, and it is the first content-specific signal in this entire project.

It is also small. Against a median per-gene R² of about 0.105, +0.003 is roughly 3%
relative. Two of three seeds show it; one shows exactly nothing.

## A dead end worth recording

The obvious metric — per-gene reconstruction error — is unusable at this budget. UNAGI's
decoder mean does not reconstruct the input: median per-gene R² of its mean is about
**−100**, i.e. far worse than predicting each gene's average, because a ZILN decoder that
is only 45 epochs in lowers its loss by inflating the learned variance rather than by
tracking the mean. Any per-gene comparison on that quantity measures the fit failure, not
the prior. Hence the readout decomposition above. (`recon_pass.py` retains the machinery
and documents the finding.)

## What it means

Parts 1–4 said the prior does nothing to either spectrum at any expressible dose or
coverage. Part 5 says that is true *in aggregate* and leaves open a small per-gene
translation: the prior may tilt predictability toward the genes it names without changing
the total. Unresolved at n = 3, but sign-consistent in two of three seeds and of exactly
the predicted shape.

The cheapest way to settle it is **not more seeds** — it is a dose-response in the same
paired design. If the effect is real it should grow with the factor: at 5× it is +0.003,
and the fold change from 5× to 20× is a within-design prediction that noise does not make.
Six runs (real and shuffled at 20×, properly normalised, three seeds) would decide it more
convincingly than twelve repeated 5× seeds.

## Reproduce (Part 5)

```bash
cd b1_expansion/prior_delivery
# the shuffled control, three seeds
for S in 42 43 44; do
  py -3.13 -u run_arm.py --arm shuffleflat \
    --workdir F:/UNAGI_Project/prior_delivery_runs/support/s$S/shuffleflat \
    --per-stage 2400 --max-iter 2 --epoch-initial 30 --epoch-iter 15 \
    --edge-cutoff 0.3 --cpo off --weight-mode direct --prior-level 5 --seed $S
  py -3.13 -u pergene_fit.py \
    --real F:/UNAGI_Project/prior_delivery_runs/support/s$S/k203 \
    --shuffled F:/UNAGI_Project/prior_delivery_runs/support/s$S/shuffleflat
done
```

---

# Part 6 — the dose-response, registered before it was run

Written 2026-10-27 15:15, before the first arm of this part finished training.
The prediction is fixed here so that it cannot be read backwards off the result —
this project has already produced four headlines that a control overturned, and a
fifth one written after the fact would be worth nothing.

## Design

Six arms: real and shuffled, at **20×**, on the three seeds that already carry the
5× pair. Nothing else moves.

| | real (`real`) | shuffled (`shuffled`) |
|---|---|---|
| support | 203 curated TRRUST targets | 203 uniformly random genes |
| weight on support, raw | 20.0 | 20.0 |
| weight on support, after normalisation | 16.837 | 16.837 |
| baseline weight | 0.8418 | 0.8418 |
| effective ratio (support / rest) | **20.0** | **20.0** |
| shuffle draw | — | `--shuffle-seed 7`, the same gene set as at 5× |
| cells / schedule / resolution | 12,000 / 45 epochs / 1.0 | identical |

The 5× comparator is Part 5's `k203` vs `shuffleflat` pair, unchanged.

Normalisation is on, as in every part since Part 2: the mean weight is held at 1.0
so the manipulation is *concentration*, not a global rescale of the reconstruction
term against the KL. Two consequences worth stating plainly, because they bound
what a dose-response can mean here:

- The support/rest ratio is exactly the factor at both levels (20.0 and 5.0), so
  the *contrast the prior creates* is a clean 4× step.
- The overall reconstruction term does lose 12.5% of its scale between 5× and 20×
  (0.9619 → 0.8418), which is a small change in the effective β. It is identical
  in the real and shuffled arms at a given level, so the paired contrast is
  unaffected; only the fold-change *between levels* carries this second-order
  confound.

## The registered prediction

Let `d(level)` be the mean per-gene readout ΔR² of the 203 real targets minus the
203 shuffled targets, after subtracting the genome-wide median shift — the same
estimator as Part 5.

> **If the Part 5 tilt is real, `d(20) > d(5)` with a paired fold change materially
> above 1.** The mechanism is a per-gene multiplier, so within the regime where the
> activation is not saturated the tilt should scale with the factor; the step from
> 5× to 20× is a 4× increase in the multiplier.
>
> **If `d(20) ≈ d(5)` or `d(20) < d(5)`, Part 5's +0.0029 was seed noise**, and the
> programme closes on a negative result: no expressible dose and no coverage of a
> real GRN prior moves either spectrum, and the per-gene tilt does not survive
> replication.

Decision rule fixed in advance: `d(20)` is read as supporting the tilt only if
**(i)** its mean exceeds `d(5)` in at least 2 of 3 seeds and **(ii)** the paired
`d(20) − d(5)` is positive and larger than its own between-seed s.d. Anything less
is reported as unresolved, not as a win on the mean alone.

## Results

Six arms, all `rc = 0`, 2 h 42 m total
(`prior_delivery_runs/dose20/run_status.json`). Delivery verified on disk before
any number was read: `geneWeight` max = **16.8365** in all six arms, `nnz` =
487,200 = 2,400 × 203 per stage file, as predicted pre-run.

### The registered statistic

`d(level)` = mean ΔR² of the 203 real targets minus that of the 203 shuffled
targets, each centred on the genome-wide median shift.

| seed | `d(5×)` | `d(20×)` | fold | κ-C(W) 5× | κ-C(W) 20× | κ-C(Z) 5× | κ-C(Z) 20× |
|---|---|---|---|---|---|---|---|
| 42 | +0.00330 | **+0.01559** | 4.7 | 0.9867 | 1.0041 | 0.9459 | 1.1880 |
| 43 | −0.00002 | **+0.01429** | – | 0.9999 | 1.0042 | 0.9509 | 0.9287 |
| 44 | +0.00549 | **+0.01794** | 3.3 | 1.0057 | 1.0145 | 1.0384 | 1.1435 |
| mean | +0.00293 ± 0.00277 | **+0.01594 ± 0.00185** | 5.4 | 0.9974 | 1.0076 ± 0.0060 | 0.9784 ± 0.0516 | 1.0867 ± 0.1386 |

Paired step: `d(20) − d(5)` = **+0.01301 ± 0.00112**, positive in **3 of 3** seeds.

**Registered verdict: SUPPORTED.** Both pre-registered conditions hold — the mean
rises in every seed, and the paired step is an order of magnitude larger than its
own between-seed s.d. (ratio 11.6).

The most informative single row is **seed 43**. At 5× it read −0.00002, i.e.
exactly nothing, and it was the reason Part 5 could not close. At 20× it reads
+0.01429. The 5× null in that seed was a power failure, not an absence of effect.

### Independent controls on the same pairs

Two controls for the one confound the shuffle does not remove — the real targets
were already easier to read out (baseline R² 0.30–0.32) than random genes
(0.25–0.28), so they might simply respond more to any reweighting
(`pergene_robust.py`):

| | 5× | 20× | fold |
|---|---|---|---|
| raw `d` | +0.00293 ± 0.00277 | +0.01594 ± 0.00185 | 5.4 |
| decile-matched excess (baseline-R² matched) | +0.00243 ± 0.00205 | **+0.01144 ± 0.00057** | 4.7 |
| OLS residual (slope on baseline R² removed) | +0.00286 ± 0.00318 | **+0.01335 ± 0.00127** | 4.7 |

The fitted slope of ΔR² on baseline R² is **−0.004 ± 0.001**: easier genes gain
*slightly less*, not more. The baseline confound therefore runs against the
effect; the raw estimate is not inflated by it. After matching, roughly 70–85% of
the raw effect survives.

What changed between 5× and 20× is not only the mean but the *signal-to-noise
ratio of the estimate itself*: at 5× the effect was about 1 s.d. wide
(0.00243 vs 0.00205), at 20× it is about 20 s.d. wide (0.01144 vs 0.00057).

Per-gene direction, for the record: at 20× the 202 real targets improve in
**76.7 / 74.3 / 78.7%** of genes, the 202 shuffled targets in
**37.6 / 43.1 / 35.1%** — i.e. the controls sit at coin-flip-minus, as they must.

### What the dose did *not* buy

- **No decoder compression.** κ-C(W) went 0.9974 → 1.0076, i.e. from 0.3%
  compression to 0.8% *expansion*. A 4× dose moved the aggregate by 1% of the 50%
  the deep-compression band requires. The Part 1–4 null at the level of the
  decoder spectrum now holds up to a 20× dose on the support.
- **No reproducible latent direction.** κ-C(Z) at 20× is 1.188 / 0.929 / 1.144 —
  it crosses 1 between seeds and its between-seed s.d. (0.139) tripled relative to
  5× (0.052). The dose amplifies the latent scatter, not a latent structure: this
  is Part 3's "the ordering reverses between seeds" at higher amplitude.

### How the two scales divide

At 20× the real and shuffled arms differ on two separate scales, and only one of
them is what the design was built to test:

| scale | size | seeds | reading |
|---|---|---|---|
| the 203 named genes, over their baseline-matched peers | **+0.0114** | 3/3 | content-specific; scales with dose; the registered effect |
| the 20,122 genes weighted in neither arm | +0.0038 to +0.0055 | 3/3 | a real-vs-shuffled gap on untouched genes; Part 8 resolves it as the control's global cost, not the real prior's gain |
| aggregate Σ R² (real / shuffled) | ×1.015 – ×1.021 | 3/3 | dominated by the row above, not by the 203 genes |

Arithmetic makes the third row inevitable: 203 genes × 0.011 is 0.06% of Σ R²,
while the arms differ by 1.5–2%. So κ-C(W) at 20× is **not** a content
measurement at all — it is measuring how far two optimisation paths drifted
apart. That is precisely why Part 5 had to move to a per-gene, median-centred
estimator, and it is the reason no aggregate κ can decide this question in either
direction.

## What Part 6 settles

The programme now has a positive result and a negative one, on the same six runs:

1. **A real GRN prior, delivered into the loss at 20× on a 203-gene support,
   produces a content-specific per-gene readout tilt that grows with the dose.**
   +0.011 after baseline matching, 3/3 seeds, ~76% of target genes improved
   against ~38% for a matched random set. This is the first effect in Parts 1–6
   that survives every control the project has learned to impose.
2. **The same intervention still does not compress the decoder spectrum.**
   κ-C(W) = 1.008 ± 0.006 at 20×, versus a 50% threshold. The prior is a local
   readout tilt, not a global geometry change — which is exactly paper1's
   content-vs-structure distinction, reproduced one level up in a *delivered*
   setting.

Both statements are needed. The first alone would be oversold as "priors work";
the second alone would be oversold as "priors are useless". What the six runs
support is narrower and more useful than either: **a per-gene likelihood weight
is a readout-tilt operator with a dose-response, and it is not a
spectrum-shaping operator at any dose that still normalises to a fixed loss.**

That sharpens the next step from "try a different mechanism" to a specific claim
worth testing: if the tilt saturates only where the readout saturates, then any
operator that acts on the *decoder geometry* rather than on the likelihood must
be the thing tested next — a Jacobian penalty or a latent-covariance prior, not
another weight vector. (Part 7 inserts one cheaper test ahead of it: the length of
the weighted phase, which was never varied. The operator change remains the
fallback, not yet the only reading.)

## Reproduce (Part 6)

```bash
cd b1_expansion/prior_delivery
py -3.13 -u run_dose.py --root F:/UNAGI_Project/prior_delivery_runs/dose20 \
    --seeds 42 43 44 --level 20 --per-stage 2400 --max-iter 2 \
    --epoch-initial 30 --epoch-iter 15 --edge-cutoff 0.3 --cpo off
py -3.13 -u dose_summary.py          # registered statistic + verdict
py -3.13 -u pergene_robust.py        # baseline-R² matching, 20x
py -3.13 -u pergene_robust.py --root F:/UNAGI_Project/prior_delivery_runs/support \
    --real-name k203 --shuf-name shuffleflat \
    --real-support-from "F:/UNAGI_Project/prior_delivery_runs/dose20/s{seed}/real"
```

Six arms at ~27 min, 2 h 42 m wall clock.

---

# Part 7 — does the tilt saturate? 40×, registered before it was run

Written 2026-10-27 18:36, minutes after the 40× driver was launched and before a
single arm had finished training. Same discipline as Part 6: the reading is fixed
in advance.

## Why one more point, and why only one

Part 6 measured two points: `d(5×) = +0.0029` and `d(20×) = +0.0159` raw, or
`+0.0024` and `+0.0114` after baseline matching. A 4× dose step produced a 4.7×
increase. Two points cannot distinguish a straight line from a curve, and the
difference matters for how the result is used:

- if the tilt keeps scaling, it is a **tunable knob** — a practitioner can trade
  prior strength against fit along a known slope;
- if it saturates between 5× and 20×, it is a **bounded local effect** — real, but
  not a control surface, and pushing harder only perturbs the optimisation path.

A third point separates these. More than three is not worth the compute until
that answer is known.

## Design

Six arms, real and shuffled at **40×**, same three seeds, nothing else moved.

| | real | shuffled |
|---|---|---|
| weight on support, after normalisation | **28.867** | **28.867** |
| baseline weight | **0.7217** | **0.7217** |
| effective ratio (support / rest) | **40.0** | **40.0** |
| shuffle draw | — | `--shuffle-seed 7`, the same 203 genes as at 5× and 20× |

The confound bound, stated plainly because it grows with dose: normalisation holds
the mean weight at 1.0, so the reconstruction term's overall scale falls
1.000 → 0.962 → 0.842 → **0.722** between 0×, 5×, 20× and 40×. Since `d` is a
contrast between two arms at the *same* level, that scale cancels inside every
`d(level)` exactly. It does **not** cancel in the ratio `d(40)/d(20)`, so the
saturation test carries this second-order term, and it is named in the result
rather than hidden.

## The registered prediction

Let `d_m(level)` be the baseline-R²-matched excess (`pergene_robust.py`, decile
version) — the stricter of the two estimators, and the one used for the verdict.

| registration | condition on `d_m(40) / d_m(20)` | reading |
|---|---|---|
| **still scaling** | > 1.5 | the operator has a usable slope; the 5→20 fold was not a fluke of one step |
| **saturated** | 0.85 – 1.5 | the tilt plateaus above 20×; 40× buys nothing, and the effect is a bounded local one |
| **collapsed** | < 0.85 | the 20× value was not a stable point; something in the fit degrades before the tilt grows |

Secondary, non-load-bearing check: κ-C(W) at 40× should stay within noise of 1 if
the operator still cannot touch decoder geometry. If κ-C(W) were to fall
materially (say below 0.95), that would reopen the geometry question and would
have to be reported as a contradiction of Parts 1–6, not as a footnote.

## Results

Six arms, all `rc = 0`, 2 h 41 m (`prior_delivery_runs/dose40/run_status.json`).
Delivery verified before any number was read: `geneWeight` max = **28.8665** in all
six arms (predicted 28.867), `nnz` = 487,200, `n_boosted` = 203, `prior_level` = 40.

### The ladder

| level | raw `d` | matched `d` | κ-C(W) | κ-C(Z) | Σ R² ratio |
|---|---|---|---|---|---|
| 5× | +0.00293 ± 0.00277 | +0.00243 ± 0.00205 | 0.9974 ± 0.0097 | 0.978 ± 0.052 | 1.0009 / 1.0015 / 1.0101 |
| 20× | +0.01594 ± 0.00185 | +0.01144 ± 0.00057 | 1.0076 ± 0.0060 | 1.087 ± 0.139 | 1.0148 / 1.0197 / 1.0212 |
| **40×** | **+0.01997 ± 0.00402** | **+0.01533 ± 0.00291** | **1.0023 ± 0.0091** | **1.078 ± 0.151** | 1.0039 / 1.0130 / 1.0236 |

Per-seed matched `d`, the quantity the verdict rests on:

| seed | 5× | 20× | 40× | 40/20 |
|---|---|---|---|---|
| 42 | +0.00317 | +0.01196 | +0.01451 | 1.21 |
| 43 | +0.00011 | +0.01153 | +0.01291 | 1.12 |
| 44 | +0.00401 | +0.01083 | +0.01856 | 1.71 |

### Registered verdict: SATURATED

`d_m(40) / d_m(20)` = 0.01533 / 0.01144 = **1.34**, inside the pre-registered
0.85–1.5 band. A linear response to a 2× dose step would have required 2.0.

The per-seed ratios are heterogeneous (1.12 / 1.21 / 1.71) and one of them crosses
1.5, so the exact word matters: **every seed is sub-linear** — all three sit below
the 2.0 a proportional response needs — and the level mean lands in the plateau
band. The registered reading is saturation; the per-seed spread says the plateau is
fuzzy, not sharp.

Log–log slope between the levels makes the shape explicit:

| step | dose ratio | effect ratio | slope |
|---|---|---|---|
| 5× → 20× | 4.0 | 4.7 | **1.12** |
| 20× → 40× | 2.0 | 1.34 | **0.42** |

The exponent drops from ~1.1 to ~0.4, i.e. the response inflects between 20× and
40× and is already more than half-saturated by 20×. With three levels this is
reported as two slopes, not as a fitted curve.

### The 40× point is not a better measurement, only a slightly larger one

The matched mean rises 34% between 20× and 40×, but the between-seed s.d. rises
**5×** (+0.00057 → +0.00291). The fraction of target genes improved actually falls
slightly, 76.7 / 74.3 / 78.7% → **71.3 / 71.3 / 75.7%**, while the controls stay
flat at 32.2 / 39.1 / 28.7%.

So beyond 20× the tilt deepens on fewer genes while dispersion grows across seeds.
If the aim is a clean, usable dose, 20× is the better operating point than 40× —
which is the opposite of what "push the dose" intuition predicts, and it is a
measured statement, not a preference.

### The secondary check holds

κ-C(W) at 40× is **1.0023 ± 0.0091** (range 0.9930–1.0112). Across the whole
ladder — 5×, 20×, 40×, 9 arms — the observed span is 0.9867 to 1.0145. That is
2.8% of a 50% target. The geometry question does not reopen; the aggregate decoder
spectrum is flat in dose, in both directions, at 40× as at 5×.

κ-C(Z) also plateaus: 1.087 ± 0.139 at 20× → 1.078 ± 0.151 at 40×. The latent
scatter inflation saturates at the same place the tilt does, which is what one
expects if both are reading the same underlying refinement.

## What Part 7 settles

Combined with Part 6, the dose axis is now closed in both directions:

1. **The per-gene tilt is real and dose-responsive between 5× and 20×**, then
   sub-linear to 40×. Its shape is a saturating response with the inflection
   between 20× and 40×, not a linear knob and not a switch.
2. **The aggregate decoder spectrum never moves**, at any of the four doses tested
   (1×, 5×, 20×, 40×) on this support. The maximum observed departure from
   κ-C(W) = 1 is 1.45%, against a 50% threshold.
3. **20× is the practical optimum, not 40×.** Above it the mean gain per unit dose
   collapses and the between-seed dispersion grows. Anyone reaching for this
   operator should stop at the low end of its plateau.

### The bound that is still open — and it is not the dose

The preceding sections assumed the 15-epoch weighted budget without naming it.
`trainer.train_model` is called with `geneWeights = geneWeights if is_iterative else
None`, so iteration 0 (30 epochs) trains unweighted and is byte-identical across
arms; **iteration 1 (15 epochs) is the only weighted pass, and the only model that
differs**. Every number in Parts 5–7 therefore measures what 15 epochs of
prior-weighted refinement does on top of a shared 30-epoch ancestor.

That is why the plateau deserves a careful reading. What Part 7 shows is that the
tilt saturates in *dose* at a fixed 15-epoch budget. It does not show that it
saturates in *training time* — and `--epoch-iter` has never been varied. Since the
measurement budget is 15 epochs and the effect is already 4–5% relative on the
named genes, the cheapest remaining experiment is not a new operator but the same
operator with a longer weighted phase. That test is one flag away and costs the
same as any of the six-arm runs above.

Only if a longer weighted phase fails to move the aggregate spectrum does the
"change the operator" conclusion become unavoidable. As it stands it is the
supported reading, not yet an exclusive one.

## Reproduce (Part 7)

```bash
cd b1_expansion/prior_delivery
py -3.13 -u run_dose.py --root F:/UNAGI_Project/prior_delivery_runs/dose40 \
    --seeds 42 43 44 --level 40 --per-stage 2400 --max-iter 2 \
    --epoch-initial 30 --epoch-iter 15 --edge-cutoff 0.3 --cpo off
py -3.13 -u pergene_robust.py --root F:/UNAGI_Project/prior_delivery_runs/dose40
py -3.13 -u dose_summary.py     # three-level ladder + both registered verdicts
```

Six arms at ~27 min, 2 h 41 m wall clock. `dose_summary.py` writes
`prior_delivery_runs/dose_ladder.json` with every per-seed value and both verdicts.

---

# Part 8 — the axis nobody varied: 20× at a 3× longer weighted budget

Registered 2026-10-27 21:26, at launch, before any arm finished.

## The hole Part 7 exposed

`trainer.train_model` receives `geneWeights = geneWeights if is_iterative else None`.
So in every run so far:

- iteration 0 — 30 epochs, **unweighted**, byte-identical across arms (epoch-000 loss
  is the same 5202.8007 in every arm of every dose);
- iteration 1 — **15 epochs**, weighted, and the model that gets audited.

Parts 5–7 therefore measured a 15-epoch intervention. The "dose saturation" in Part 7
is saturation *within that budget*. Training length was never a variable, and it is
one.

## Design

Six arms, dose held at **20×**, weighted phase raised 15 → **45 epochs** (3×). Nothing
else changes.

| | 20× / e15 (Part 6) | 20× / e45 (this part) |
|---|---|---|
| weighted epochs | 15 | **45** |
| unweighted epochs | 30 | 30 — same seed, same data, same schedule |
| total epochs | 45 | 75 |
| dose, ratio to rest | 20.0 | 20.0 |
| arms | real + shuffled × 3 seeds | identical |
| shuffle draw | `--shuffle-seed 7` | identical |

Because `epoch_initial` is deliberately left at 30, **the shared ancestor is the same
model as in Part 6** — the two parts differ only in how long the weighted phase runs
on top of it. The `d` comparison between them is therefore a clean time contrast, not
a re-run.

Cost, corrected: 75 epochs ≈ 39 min per arm, six arms ≈ **3.9 h** (my earlier estimate
of 1.5 h forgot that the arm itself grows by 30 epochs).

## The registered prediction

`r = d_m(20×, e45) / d_m(20×, e15)`, with `d_m` the baseline-R²-matched excess as in
Parts 6–7, and `d_m(20×, e15) = +0.01144`.

| registration | condition | reading |
|---|---|---|
| **still time-limited** | `r > 1.8` | the Part 4–7 "plateau" was a budget artefact; the operator has more room and the dose ladder must be reinterpreted |
| **partly** | `1.2 ≤ r ≤ 1.8` | both axes matter; the tilt is neither dose- nor time-saturated, just slow |
| **converged** | `r < 1.2` | the tilt saturates in time as well as in dose — two independent ceilings on the same operator |

## What this part can and cannot answer

**Can:** whether more weighted training grows the per-gene tilt. That contrast is
between two arms at the same 75-epoch budget, so it is unaffected by anything the
empty reference does.

**Cannot:** κ-C(W) against a no-prior baseline. The `empty` arm available here was
trained at 45 total epochs (30 + 15); these arms run 75. Comparing them would credit
the prior with 30 epochs of extra training. So the geometry question at this budget
needs an `empty` at 75 epochs, and that is a **contingent step, run only if
`r > 1.2`** — spending 2.3 h on a control for a branch that may not occur is the kind
of thing this project has learned not to do.

What *is* interpretable at matched epochs is the content contrast: Σ R²(real) /
Σ R²(shuffled) at 75 epochs, both arms, same budget. That will be reported as before,
with the same caveat that it is dominated by inter-arm drift rather than by the 203
genes.

## Results

Six arms, all `rc = 0`, 4 h 12 m (`prior_delivery_runs/dose20_e45/run_status.json`,
41.6–42.4 min per arm). Dose held at 20×; only the weighted phase changed.

| | 20× / e15 | 20× / e45 |
|---|---|---|
| raw `d` (centred) | +0.01594 ± 0.00185 | +0.01875 ± 0.00140 |
| **matched `d`** | **+0.01144 ± 0.00057** | **+0.01439 ± 0.00161** |
| OLS residual | +0.01335 ± 0.00127 | +0.01323 ± 0.00214 |
| real targets, frac. improved | 76.7 / 74.3 / 78.7% | 73.3 / 73.3 / 71.3% |
| shuffled targets, frac. improved | 37.6 / 43.1 / 35.1% | 35.1 / 33.2 / 33.2% |
| Σ R² ratio (real / shuffled) | 1.015 / 1.020 / 1.021 | 1.009 / 1.010 / 1.013 |

### Registered verdict: PARTLY

`r = d_m(e45) / d_m(e15)` = 0.01439 / 0.01144 = **1.26**, inside the pre-registered
1.2–1.8 band.

But the band verdict is the least informative thing that can be said about this
number. The informative one is the exponent: **three times the weighted budget bought
1.26× the effect, i.e. `log(1.26)/log(3) = 0.21`.** A time-limited effect would have
scaled near 1; a converged one would sit at 0. At 0.21 the tilt is close to
converged, and the registered band's "both axes matter" reading overstates how much
room is left on this one.

Per seed the matched ratios are 1.10 / 1.41 / 1.27 — all above 1, none close to 3.

### Two levers, one ceiling

This is the sharpest way to state what Parts 6–8 jointly show:

| route to a larger tilt | parameter | matched `d` |
|---|---|---|
| more dose, 15-epoch budget | 5× → 20× → **40×** | 0.0024 → 0.0114 → **0.0153** |
| more time, 20× dose | 15 → **45** weighted epochs | 0.0114 → **0.0144** |

The two routes land within 6% of each other (0.0153 vs 0.0144) with overlapping
s.d. — i.e. **the operator saturates at roughly +0.0145 on its own 203 genes,
and it does not matter whether the last increment comes from dose or from training
time.** Two independent levers arriving at the same ceiling is much stronger
evidence for a real ceiling than either ladder alone: an artefact of one axis
would not reproduce on the other.

Relative to the targets' own baseline readout (R² ≈ 0.30), +0.0145 is about 5%.
Against the aggregate Σ R² it is invisible: 203 × 0.0145 is 0.06% of the total.

A side observation worth one line: at e45 the *shuffled* targets read
−0.0040 / −0.0024 / −0.0041 below the genome-wide median shift (they were ≈ 0 at
e15). The redistribution has acquired a losing side, and it is exactly the control
set. The aggregate ratio, meanwhile, fell from 1.015–1.021 to 1.009–1.013 — the
inter-arm gap that dominated the aggregate at e15 shrinks with longer training.

### The geometry check, against a matched no-prior baseline

Three `empty` arms at 30 + 45 epochs, same cells, same seeds, ran to completion
(45.6–45.9 min each; `arm_meta` confirms `epoch_initial = 30`, `epoch_iter = 45`).
Their first attempt was rejected before it produced a number: the reuse guard
checked only for `latent.npz`, and `run_dose.py` had already copied the *e15*
reference into that directory, so 75-epoch arms would have been audited against a
45-epoch baseline — precisely the confound the arms exist to remove. The guard now
verifies `arm`, `epoch_initial` and `epoch_iter` from `arm_meta.json`, and the arms
are written to `empty_e45/`.

| seed | κ-C(W) real vs `empty_e45` | κ-C(W) shuffled vs `empty_e45` | κ-C(Z) real |
|---|---|---|---|
| 42 | 1.0016 (−0.16%) | 0.9931 (+0.69%) | 1.0358 |
| 43 | 1.0008 (−0.08%) | 0.9914 (+0.86%) | 0.9379 |
| 44 | 1.0003 (−0.03%) | 0.9876 (+1.24%) | 0.9889 |
| mean | **1.0009 ± 0.0007** | **0.9907 ± 0.0028** | 0.9875 ± 0.0490 |

**The real prior's effect on the decoder spectrum at 20× and a 3× budget is
0.09% — indistinguishable from nothing.** That is 550× short of the 50% threshold.
The weight-matched random control, by contrast, reads 0.9% *compression*, 3/3
seeds. The content of the prior does not compress the decoder; if anything it is
the control that does.

κ-C(Z) shows no consistent sign (one seed below 1, mean 0.9875 ± 0.049), as at
every other level.

### The absolute per-gene test — and a correction to what Part 8 said one turn ago

The matched baseline also makes the *absolute* per-gene question answerable for the
first time, and that question matters: "real beats shuffled" is compatible with
both arms sitting below no-prior. They do not. Against `empty_e45`:

| arm | on the 203 genes it names | on the other 20,122 genes |
|---|---|---|
| **real prior** | **+0.00861 ± 0.00338** (3/3) | +0.00017 ± 0.00021 (2/3, nil) |
| random control | +0.00188 ± 0.00055 (3/3) | **−0.00250 ± 0.00078** (0/3) |

So the real prior is a **strictly local operator**: it raises readout by 0.86 pp on
the genes it names and leaves everything else exactly where it was. The control
gains 4.6× less on its own genes *and* pays 0.25 pp on the rest.

This forces a correction. The previous subsection, written before these arms
landed, read the ~1% real-vs-shuffled aggregate gap as the real prior's *global
gain* on unnamed genes (+0.45 pp). With a matched baseline that attribution is
wrong: the real prior's global effect is +0.02 pp, i.e. nil. The entire gap is the
control's global *cost* (−0.25 pp), which is what shows up as its 0.9% compression.
The decomposition is consistent — real vs shuffled on the real targets (+0.01572)
= real above baseline (+0.00861) + control below baseline on those same genes
(−0.00711).

That makes the project's conclusion sharper, not weaker:

> **A GRN prior delivered as a per-gene likelihood weight is a purely local readout
> operator. At 20× dose and 3× the weighted budget it buys +0.86 pp R² on the 203
> genes it names, costs nothing elsewhere, and moves the decoder spectrum by 0.09%
> — against a diagnostic that asks for 50%. Its content is what buys the locality:
> a weight-matched random set gains 4.6× less locally and pays for it globally.**

## Where the programme now stands

Nine parts, 30 trained arms, one operator:

| question | answer | evidence |
|---|---|---|
| does the prior reach the loss | yes, verified on disk | nnz = 2,400 × 203, max weight as designed, every arm |
| does it compress the decoder spectrum | **no** | κ-C(W) ∈ [0.9867, 1.0145] across 5×/20×/40× and 3× budget; 0.09% at the matched baseline |
| does it move the latent spectrum | **no reproducible direction** | κ-C(Z) crosses 1 at every level; between-seed s.d. grows with dose (0.05 → 0.15) |
| does it change per-gene readout | **yes, locally** | +0.86 pp on its own 203 genes over no prior, 3/3, nil elsewhere |
| is that gain from content | **yes** | random set of the same size and weight: +0.19 pp locally, −0.25 pp globally |
| does the gain scale | **to a ceiling near +0.0145** | dose 5→40× and time 15→45 epochs both land at 0.0144–0.0153 |

The operator is characterised. What remains is not another arm of this design — it
is a different operator, and the report already states which one and why.

## Reproduce (Part 8)

```bash
cd b1_expansion/prior_delivery
py -3.13 -u run_dose.py --root F:/UNAGI_Project/prior_delivery_runs/dose20_e45 \
    --seeds 42 43 44 --level 20 --per-stage 2400 --max-iter 2 \
    --epoch-initial 30 --epoch-iter 45 --edge-cutoff 0.3 --cpo off
py -3.13 -u pergene_robust.py --root F:/UNAGI_Project/prior_delivery_runs/dose20_e45
py -3.13 -u run_empty_e45.py --root F:/UNAGI_Project/prior_delivery_runs/dose20_e45 \
    --seeds 42 43 44 --epoch-initial 30 --epoch-iter 45 --per-stage 2400 \
    --max-iter 2 --edge-cutoff 0.3 --cpo off
py -3.13 -u pergene_vs_baseline.py
```

Six weighted arms at ~42 min (4 h 12 m) plus three no-prior arms at ~46 min
(2 h 18 m). The `run_empty_e45.py` step must come after `run_dose.py`, because the
latter copies the e15 reference into each seed directory and the former must not
mistake it for its own output.










