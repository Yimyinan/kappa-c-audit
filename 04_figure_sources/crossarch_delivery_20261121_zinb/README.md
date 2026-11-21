# Third architecture: ZINB count VAE

This experiment adds a count-likelihood architecture to the delivered-prior
analysis. It is structurally distinct from PG-UNAGI and from the Gaussian MLP
VAE in V43.

## Architecture

- two-layer MLP encoder and decoder with LayerNorm and GELU;
- 64 latent dimensions;
- zero-inflated negative-binomial likelihood on reconstructed raw counts;
- no GCN encoder, adversarial branch, staged graph, or iZILN decoder;
- gene-specific inverse dispersion parameter.

## Protocol

- same 12,000 cells and 20,527 genes;
- same real 203-gene TRRUST prior and matched random 203-gene control;
- no-prior, real-prior and random-prior arms;
- seeds 42, 123 and 456;
- 150 epochs, beta 0.001, boost 10;
- same initialization, batch order and dropout stream across arms within each
  seed.

Raw counts are reconstructed from the deposited log-normalized matrix as
`round(expm1(X) * total_counts / 10000)`. The encoder sees the deposited
log-normalized matrix, standardized per gene. The ZINB likelihood uses the
reconstructed counts and recorded library size.

## Health rule

The ZINB-specific gate was fixed after a no-prior pilot: effective latent rank
>= 2.5, total in-sample Ridge R2 >= 1000, latent standard deviation > 0.2,
KL > 0.1, and finite weighted NLL. This avoids applying the Gaussian-VAE rank
range mechanically to a different likelihood.

## Result

- Real minus random on 202 true targets:
  `+0.0118 +/- 0.0016 R2`, positive in 3/3 seeds.
- Content interaction:
  `+0.0202 +/- 0.0020`, positive in 3/3 seeds.
- Five-fold fixed-latent readout:
  target effect `+0.0136 +/- 0.0013`; interaction
  `+0.0206 +/- 0.0020`; both positive in 3/3 seeds.
- Aggregate decoder ratio:
  real `0.9763 +/- 0.0035`; random `0.9889 +/- 0.0057`.
- All 9 fits passed the ZINB health gate.
- Total training time: 39.96 minutes on one RTX 4060 Laptop GPU.

The local content effect transfers to a count-likelihood architecture, while
the aggregate decoder readout remains near unity and far from deep compression.

