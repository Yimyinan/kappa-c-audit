#!/usr/bin/env python3
"""Cross-architecture test of a gene prior that is actually delivered to the loss.

Architecture:
    plain 2-layer MLP VAE with Gaussian MSE reconstruction.

Protocol:
    12,000 hematopoietic cells x 20,527 genes, exactly the delivery-experiment
    matrix. Three paired arms per training seed:
      empty   unweighted reconstruction
      real    the 203 TRRUST hematopoietic targets, boost 10, normalized to mean 1
      random  a size-matched gene permutation of the same prior (203 targets)

    The reconstruction term is
        mean_g [ w_g * (x_g - xhat_g)^2 ].
    This is an explicit delivery path: the prior enters the differentiable loss.

Readouts:
    * per-gene Ridge R2 for every arm;
    * the 2x2 content interaction on the disjoint real/random target sets;
    * the aggregate predictability ratio and latent-spectrum ratios;
    * latent-health gates.

Outputs are resumable. Each completed arm writes:
    runs/seed_<seed>/<arm>/model.pt
    runs/seed_<seed>/<arm>/latent.npz
    runs/seed_<seed>/<arm>/metrics.json

The aggregate result is written to crossarch_results.json.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
from scipy import stats as scipy_stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
H5AD = Path(
    r"F:\UNAGI_Project\prior_delivery_runs\dose20_e45\s42\real\hemato.h5ad"
)
REAL_SUPPORT = Path(
    r"F:\UNAGI_Project\prior_delivery_runs\dose20_e45\s42\real\support.npz"
)
RANDOM_SUPPORT = Path(
    r"F:\UNAGI_Project\prior_delivery_runs\dose20_e45\s42\shuffled\support.npz"
)

SEEDS = (42, 123, 456)
ARMS = ("empty", "real", "random")
EPOCHS = 150
BETA = 0.05
BOOST = 10.0
HIDDEN = 256
LATENT = 64
BATCH = 512
LR = 1e-3
HEALTH_MIN_EFF_RANK = 3.0


class Encoder(nn.Module):
    def __init__(self, n_input: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, HIDDEN),
            nn.BatchNorm1d(HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN, HIDDEN),
            nn.BatchNorm1d(HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.mu = nn.Linear(HIDDEN, LATENT)
        self.logvar = nn.Linear(HIDDEN, LATENT)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, n_output: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT, HIDDEN),
            nn.BatchNorm1d(HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN, HIDDEN),
            nn.BatchNorm1d(HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.head = nn.Linear(HIDDEN, n_output)

    def forward(self, z):
        return self.head(self.net(z))


class MLPVAE(nn.Module):
    def __init__(self, n_genes: int):
        super().__init__()
        self.encoder = Encoder(n_genes)
        self.decoder = Decoder(n_genes)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return self.decoder(z), mu, logvar, z

    @torch.no_grad()
    def latent(self, x):
        return self.encoder(x)[0]


def parameter_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def effective_rank(cov: np.ndarray) -> float:
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0.0, None)
    denom = float(np.square(eig).sum())
    return float(eig.sum() ** 2 / denom) if denom > 0 else float("nan")


def ridge_per_gene_r2(z: np.ndarray, x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Fit Ridge on all cells and return one in-sample R2 per gene."""
    z = np.asarray(z, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    sd = z.std(axis=0)
    sd[sd < 1e-8] = 1.0
    z = (z - z.mean(axis=0)) / sd

    gram = z.T @ z
    gram.flat[:: gram.shape[0] + 1] += alpha
    coef = np.linalg.solve(gram.astype(np.float64), z.T.astype(np.float64) @ x.astype(np.float64))
    sst = np.square(x).sum(axis=0, dtype=np.float64)
    sse = np.zeros(x.shape[1], dtype=np.float64)
    block = 2048
    for start in range(0, x.shape[0], block):
        stop = min(start + block, x.shape[0])
        pred = z[start:stop].astype(np.float64) @ coef
        sse += np.square(pred - x[start:stop]).sum(axis=0)
    r2 = np.ones(x.shape[1], dtype=np.float64)
    varying = sst > 1e-12
    r2[varying] = 1.0 - sse[varying] / sst[varying]
    r2[~varying] = 0.0
    return r2.astype(np.float32)


def ridge_per_gene_r2_cv(
    z: np.ndarray,
    x: np.ndarray,
    n_splits: int = 5,
    seed: int = 20261120,
    alpha: float = 1.0,
) -> np.ndarray:
    """Cross-validated Ridge R2 on fixed latent codes.

    The VAE was trained on all cells. This is therefore cross-validation of the
    readout, not an end-to-end held-out model evaluation.
    """
    z = np.asarray(z, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    indices = np.random.default_rng(seed).permutation(x.shape[0])
    folds = np.array_split(indices, n_splits)
    sse = np.zeros(x.shape[1], dtype=np.float64)
    sst = np.zeros(x.shape[1], dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(x.shape[0], dtype=bool)
        train_mask[test_idx] = False
        z_train, z_test = z[train_mask], z[test_idx]
        x_train, x_test = x[train_mask], x[test_idx]
        z_mean, z_sd = z_train.mean(axis=0), z_train.std(axis=0)
        z_sd[z_sd < 1e-8] = 1.0
        z_train = (z_train - z_mean) / z_sd
        z_test = (z_test - z_mean) / z_sd
        gram = z_train.T @ z_train
        gram.flat[:: gram.shape[0] + 1] += alpha
        coef = np.linalg.solve(
            gram.astype(np.float64),
            z_train.T.astype(np.float64) @ x_train.astype(np.float64),
        )
        pred = z_test.astype(np.float64) @ coef
        sse += np.square(pred - x_test).sum(axis=0)
        sst += np.square(x_test).sum(axis=0)
    r2 = np.ones(x.shape[1], dtype=np.float64)
    varying = sst > 1e-12
    r2[varying] = 1.0 - sse[varying] / sst[varying]
    r2[~varying] = 0.0
    return r2.astype(np.float32)


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    a = ad.read_h5ad(H5AD)
    x = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    x = np.asarray(x, dtype=np.float32)
    x = np.log1p(x)
    mean, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd < 1e-8] = 1.0
    x = ((x - mean) / sd).astype(np.float32)

    real = np.load(REAL_SUPPORT, allow_pickle=True)
    rand = np.load(RANDOM_SUPPORT, allow_pickle=True)
    real_idx = np.asarray(real["idx"], dtype=np.int64)
    random_idx = np.asarray(rand["idx"], dtype=np.int64)
    genes = np.asarray(a.var_names).astype(str)
    if not np.array_equal(np.asarray(real["gene_names"]).astype(str), genes):
        raise ValueError("real support gene order does not match the expression matrix")
    if not np.array_equal(np.asarray(rand["gene_names"]).astype(str), genes):
        raise ValueError("random support gene order does not match the expression matrix")
    a.file.close()
    return x, real_idx, random_idx, genes.tolist()


def make_weight(n_genes: int, idx: np.ndarray, boost: float) -> np.ndarray:
    w = np.ones(n_genes, dtype=np.float32)
    w[idx] = 1.0 + boost
    return (w / w.mean()).astype(np.float32)


def train_arm(
    x: np.ndarray,
    arm: str,
    seed: int,
    weight: np.ndarray | None,
    output: Path,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random = np.random.default_rng(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPVAE(x.shape[1]).to(device)
    initial_digest = parameter_digest(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(x))
    loader = DataLoader(
        dataset,
        batch_size=BATCH,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )
    weight_t = (
        torch.as_tensor(weight, dtype=torch.float32, device=device)
        if weight is not None
        else None
    )

    started = time.time()
    history: list[dict] = []
    for epoch in range(EPOCHS):
        model.train()
        total_loss = total_recon = total_kl = 0.0
        n_batches = 0
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            recon_x, mu, logvar, _ = model(xb)
            sq_err = (recon_x - xb).square()
            if weight_t is None:
                recon = sq_err.mean()
            else:
                recon = (sq_err * weight_t.unsqueeze(0)).mean()
            kl = -0.5 * torch.mean(1 + logvar - mu.square() - logvar.exp())
            loss = recon + BETA * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
            total_recon += float(recon.detach())
            total_kl += float(kl.detach())
            n_batches += 1
        scheduler.step()
        rec = {
            "epoch": epoch,
            "loss": total_loss / n_batches,
            "recon": total_recon / n_batches,
            "kl": total_kl / n_batches,
        }
        history.append(rec)
        if epoch == 0 or (epoch + 1) % 15 == 0 or epoch + 1 == EPOCHS:
            print(
                f"[{arm} seed={seed}] epoch {epoch+1:03d}/{EPOCHS} "
                f"loss={rec['loss']:.5f} recon={rec['recon']:.5f} kl={rec['kl']:.5f}",
                flush=True,
            )

    model.eval()
    latents = []
    xt = torch.from_numpy(x)
    with torch.no_grad():
        for start in range(0, x.shape[0], 2048):
            latents.append(
                model.latent(xt[start : start + 2048].to(device, non_blocking=True))
                .cpu()
                .numpy()
            )
    z = np.concatenate(latents, axis=0).astype(np.float32)
    cov = np.cov(z.astype(np.float64), rowvar=False)
    trace_z = float(np.trace(cov))
    eff_rank = effective_rank(cov)

    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arm": arm,
            "seed": seed,
            "weight": weight,
            "architecture": "plain MLP VAE (2x256, latent 64); no GCN, GAN, or iZILN",
        },
        output / "model.pt",
    )
    np.savez_compressed(
        output / "latent.npz",
        Z=z,
        weight=weight if weight is not None else np.ones(x.shape[1], dtype=np.float32),
    )
    metrics = {
        "arm": arm,
        "seed": seed,
        "n_cells": int(x.shape[0]),
        "n_genes": int(x.shape[1]),
        "epochs": EPOCHS,
        "beta": BETA,
        "boost": BOOST if weight is not None else None,
        "weight_min": float(weight.min()) if weight is not None else 1.0,
        "weight_max": float(weight.max()) if weight is not None else 1.0,
        "weight_mean": float(weight.mean()) if weight is not None else 1.0,
        "initial_parameter_sha256": initial_digest,
        "trace_Z": trace_z,
        "eff_rank_Z": eff_rank,
        "latent_sd": float(z.std()),
        "final_loss": history[-1]["loss"],
        "final_recon": history[-1]["recon"],
        "final_kl": history[-1]["kl"],
        "epoch_history": history,
        "seconds": float(time.time() - started),
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    del model, optimizer, scheduler, loader, dataset, xt
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def summarize(x: np.ndarray, real_idx: np.ndarray, random_idx: np.ndarray) -> dict:
    overlap = np.intersect1d(real_idx, random_idx)
    real = np.setdiff1d(real_idx, overlap)
    random = np.setdiff1d(random_idx, overlap)
    other = np.setdiff1d(np.arange(x.shape[1]), np.union1d(real_idx, random_idx))
    rows = []
    all_r2: dict[tuple[str, int], np.ndarray] = {}
    all_r2_cv: dict[tuple[str, int], np.ndarray] = {}
    latent: dict[tuple[str, int], dict] = {}

    for seed in SEEDS:
        for arm in ARMS:
            out = ROOT / "runs" / f"seed_{seed}" / arm
            m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
            z = np.load(out / "latent.npz")["Z"]
            r2 = ridge_per_gene_r2(z, x)
            r2_cv = ridge_per_gene_r2_cv(z, x)
            np.save(out / "pergene_r2.npy", r2)
            np.save(out / "pergene_r2_cv.npy", r2_cv)
            all_r2[(arm, seed)] = r2
            all_r2_cv[(arm, seed)] = r2_cv
            latent[(arm, seed)] = m
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "sum_R2": float(r2.sum()),
                    "sum_R2_cv": float(r2_cv.sum()),
                    "mean_R2_other": float(r2[other].mean()),
                    "mean_R2_real_targets": float(r2[real].mean()),
                    "mean_R2_random_targets": float(r2[random].mean()),
                    "trace_Z": m["trace_Z"],
                    "eff_rank_Z": m["eff_rank_Z"],
                    "healthy": bool(
                        m["eff_rank_Z"] >= HEALTH_MIN_EFF_RANK
                        and m["latent_sd"] > 0.01
                        and m["final_kl"] > 0
                    ),
                }
            )

    paired = []
    for seed in SEEDS:
        empty = all_r2[("empty", seed)]
        real_r2 = all_r2[("real", seed)]
        random_r2 = all_r2[("random", seed)]
        empty_cv = all_r2_cv[("empty", seed)]
        real_cv = all_r2_cv[("real", seed)]
        random_cv = all_r2_cv[("random", seed)]
        d_real = real_r2 - empty
        d_random = random_r2 - empty
        d_real_cv = real_cv - empty_cv
        d_random_cv = random_cv - empty_cv
        cells = {
            "real_prior_on_real_targets": float(d_real[real].mean()),
            "real_prior_on_random_targets": float(d_real[random].mean()),
            "random_prior_on_real_targets": float(d_random[real].mean()),
            "random_prior_on_random_targets": float(d_random[random].mean()),
        }
        interaction = (
            cells["real_prior_on_real_targets"]
            - cells["real_prior_on_random_targets"]
            - cells["random_prior_on_real_targets"]
            + cells["random_prior_on_random_targets"]
        )
        real_vs_random_on_real = float((real_r2[real] - random_r2[real]).mean())
        real_vs_random_on_random = float((real_r2[random] - random_r2[random]).mean())
        ratio_real = float(real_r2.sum() / empty.sum())
        ratio_random = float(random_r2.sum() / empty.sum())
        paired.append(
            {
                "seed": seed,
                **cells,
                "interaction": float(interaction),
                "real_vs_random_on_real_targets": real_vs_random_on_real,
                "real_vs_random_on_random_targets": real_vs_random_on_random,
                "cv_real_vs_random_on_real_targets": float(
                    (real_cv[real] - random_cv[real]).mean()
                ),
                "cv_interaction": float(
                    (d_real_cv[real].mean() - d_real_cv[random].mean())
                    - (d_random_cv[real].mean() - d_random_cv[random].mean())
                ),
                "real_gt_random_on_real_genes": int((real_r2[real] > random_r2[real]).sum()),
                "random_gt_real_on_random_genes": int((random_r2[random] > real_r2[random]).sum()),
                "real_above_empty_on_real_genes": int((real_r2[real] > empty[real]).sum()),
                "random_above_empty_on_random_genes": int((random_r2[random] > empty[random]).sum()),
                "real_above_empty_on_random_genes": int((real_r2[random] > empty[random]).sum()),
                "random_above_empty_on_real_genes": int((random_r2[real] > empty[real]).sum()),
                "kappa_C_W_real": ratio_real,
                "C_percent_real": 100.0 * (1.0 - ratio_real),
                "kappa_C_W_random": ratio_random,
                "C_percent_random": 100.0 * (1.0 - ratio_random),
                "kappa_C_Z_real": float(
                    latent[("real", seed)]["trace_Z"]
                    / latent[("empty", seed)]["trace_Z"]
                ),
                "kappa_C_Z_random": float(
                    latent[("random", seed)]["trace_Z"]
                    / latent[("empty", seed)]["trace_Z"]
                ),
            }
        )
    return {
        "architecture": "plain MLP VAE (2x256, latent 64; Gaussian MSE)",
        "dataset": "GSE150774 delivery matrix, 12,000 cells x 20,527 genes",
        "prior_delivery": "weight multiplies the per-gene squared reconstruction error",
        "boost": BOOST,
        "epochs": EPOCHS,
        "beta": BETA,
        "n_real_targets": int(real.size),
        "n_random_targets": int(random.size),
        "n_other_genes": int(other.size),
        "overlap_excluded": [str(i) for i in overlap],
        "per_arm": rows,
        "paired": paired,
    }


def aggregate(summary: dict) -> dict:
    rows = summary["paired"]
    keys = [
        "interaction",
        "real_vs_random_on_real_targets",
        "real_prior_on_real_targets",
        "random_prior_on_random_targets",
        "cv_real_vs_random_on_real_targets",
        "cv_interaction",
        "kappa_C_W_real",
        "kappa_C_W_random",
    ]
    stats = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=float)
        stats[key] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
            "min": float(values.min()),
            "max": float(values.max()),
            "positive_seeds": int((values > 0).sum()),
        }
        target = 1.0 if key.startswith("kappa_C_W") else 0.0
        test = scipy_stats.ttest_1samp(values, popmean=target)
        se = values.std(ddof=1) / math.sqrt(values.size)
        crit = scipy_stats.t.ppf(0.975, values.size - 1)
        stats[key].update(
            {
                "null_value": target,
                "paired_t": float(test.statistic),
                "two_sided_p": float(test.pvalue),
                "ci95": [float(values.mean() - crit * se), float(values.mean() + crit * se)],
            }
        )
    all_healthy = all(row["healthy"] for row in summary["per_arm"])
    passed = bool(
        all_healthy
        and stats["interaction"]["mean"] > 0
        and stats["real_vs_random_on_real_targets"]["mean"] > 0
        and stats["real_vs_random_on_real_targets"]["positive_seeds"] >= 2
        and abs(stats["kappa_C_W_real"]["mean"] - 1.0) < 0.10
        and abs(stats["kappa_C_W_random"]["mean"] - 1.0) < 0.10
    )
    return {
        **summary,
        "stats": stats,
        "all_healthy": all_healthy,
        "cross_architecture_delivery_pass": passed,
        "decision_rule": (
            "PASS iff all arms pass latent health, mean content interaction > 0, "
            "real-vs-random target effect > 0 in at least 2/3 seeds, and both "
            "aggregate predictability ratios remain within 0.90-1.10."
        ),
    }


def main() -> None:
    global EPOCHS, SEEDS
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    EPOCHS = args.epochs
    SEEDS = tuple(args.seeds)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.set_float32_matmul_precision("high")

    print(f"[data] loading {H5AD}", flush=True)
    x, real_idx, random_idx, genes = load_data()
    real_w = make_weight(x.shape[1], real_idx, BOOST)
    random_w = make_weight(x.shape[1], random_idx, BOOST)
    print(
        f"[data] X={x.shape}; real support={real_idx.size}; "
        f"random support={random_idx.size}; overlap={np.intersect1d(real_idx, random_idx).size}",
        flush=True,
    )
    print(
        f"[weight] target={real_w[real_idx[0]]:.3f}, background={real_w[0]:.3f}, "
        f"support={real_idx.size}",
        flush=True,
    )

    for seed in SEEDS:
        for arm, weight in [
            ("empty", None),
            ("real", real_w),
            ("random", random_w),
        ]:
            out = ROOT / "runs" / f"seed_{seed}" / arm
            if not args.force and (out / "metrics.json").exists():
                print(f"[skip] seed={seed} arm={arm} already complete", flush=True)
                continue
            train_arm(x, arm, seed, weight, out)

    summary = summarize(x, real_idx, random_idx)
    result = aggregate(summary)
    (ROOT / "crossarch_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["stats"], indent=2), flush=True)
    print(
        f"[result] PASS={result['cross_architecture_delivery_pass']} "
        f"healthy={result['all_healthy']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
