# Cross-architecture delivered-prior experiment

## Question

Does a gene-set prior that multiplies the differentiable reconstruction loss
produce the same qualitative effect outside PG-UNAGI?

## Architecture

A plain two-layer MLP VAE:

- 2 x 256 hidden units, 64 latent dimensions;
- Gaussian likelihood with squared-error reconstruction;
- no GCN encoder, adversarial branch, staged graph, or iZILN decoder.

This is structurally distinct from PG-UNAGI and avoids reusing the architecture
whose delivery path failed.

## Fixed protocol

- Data: GSE150774 delivery matrix, 12,000 cells x 20,527 genes.
- Real prior: the same 203 TRRUST hematopoietic target genes used in the main
  delivery experiment.
- Random control: the same 203-gene weight-matched permutation used by the main
  experiment; one overlapping gene is excluded from paired target sets.
- Reconstruction: `mean_g[w_g (x_g - xhat_g)^2]`.
- Boost: 10, normalized so mean weight equals 1.
- Seeds: 42, 123, 456.
- Epochs: 150.
- Arms: `empty`, `real`, `random`.
- Batch order, initialization, dropout stream and data are seed-paired across arms.

## Decision rule fixed before the run

The experiment passes only if:

1. every arm has effective latent rank >= 3, nonzero KL and non-degenerate latent sd;
2. the mean two-by-two content interaction is positive;
3. real versus random improves the real targets in at least 2/3 seeds;
4. both real and random aggregate predictability ratios remain within 0.90-1.10.

## Run

```powershell
$env:PYTHONPATH = 'F:\contrastvae_env\Lib\site-packages'
& 'C:\Program Files\WindowsApps\OpenAI.CodexPrimaryRuntime.v26-1119-907-0_26.1119.907.0_x64__3k8sg7r9htsxt\dependencies\python\python.exe' `
    'experiments\crossarch_delivery_20261120\crossarch_delivery.py'
```

