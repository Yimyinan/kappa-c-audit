#!/usr/bin/env python3
"""Third-architecture delivered-prior test: zero-inflated count VAE.

The VAE uses a zero-inflated negative-binomial likelihood. The prior multiplies
the per-gene negative log-likelihood. All cells, genes, targets, random controls
and training seeds match the V43 plain-MLP experiment.

Outputs are resumable and intentionally separate from the Gaussian-MLP run.
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
import torch
import torch.nn as nn
from scipy import stats as scipy_stats
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
BETA = 0.001
BOOST = 10.0
HIDDEN = 256
LATENT = 64
BATCH = 512
LR = 1e-3
HEALTH_MIN_EFF_RANK = 2.5
HEALTH_MIN_SUM_R2 = 1000.0
HEALTH_MIN_KL = 0.1
HEALTH_MIN_LATENT_SD = 0.2


class Encoder(nn.Module):
    def __init__(self, n_genes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.mu = nn.Linear(HIDDEN, LATENT)
        self.logvar = nn.Linear(HIDDEN, LATENT)
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, -4.0)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class ZINBDecoder(nn.Module):
    def __init__(self, n_genes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.mean_log_norm = nn.Linear(HIDDEN, n_genes)
        self.zero_logits = nn.Linear(HIDDEN, n_genes)
        nn.init.constant_(self.mean_log_norm.bias, -3.0)
        nn.init.zeros_(self.zero_logits.weight)
        nn.init.constant_(self.zero_logits.bias, -4.0)

    def forward(self, z):
        h = self.net(z)
        log_norm_mean = self.mean_log_norm(h).clamp(-12.0, 6.0)
        zero_logits = self.zero_logits(h).clamp(-12.0, 12.0)
        return log_norm_mean, zero_logits


class ZINBVAE(nn.Module):
    def __init__(self, n_genes: int):
        super().__init__()
        self.encoder = Encoder(n_genes)
        self.decoder = ZINBDecoder(n_genes)
        self.log_theta = nn.Parameter(torch.full((n_genes,), math.log(2.0)))

    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        log_norm_mean, zero_logits = self.decoder(z)
        log_theta = self.log_theta.clamp(-4.0, 5.0)
        return log_norm_mean, log_theta, zero_logits, mu, logvar

    @torch.no_grad()
    def latent(self, x):
        return self.encoder(x)[0]


def zinb_nll(
    x: torch.Tensor,
    library: torch.Tensor,
    log_norm_mean: torch.Tensor,
    log_theta: torch.Tensor,
    zero_logits: torch.Tensor,
) -> torch.Tensor:
    """Per-gene mean ZINB negative log likelihood."""
    mean = torch.exp(log_norm_mean) * library[:, None]
    mean = mean.clamp_min(1e-6)
    theta = torch.exp(log_theta).clamp_min(1e-4).unsqueeze(0).expand_as(mean)
    log_theta_ratio = torch.log(theta) - torch.log(theta + mean)
    log_mean_ratio = torch.log(mean) - torch.log(theta + mean)
    log_nb = (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * log_theta_ratio
        + x * log_mean_ratio
    )
    log_pi = torch.nn.functional.logsigmoid(zero_logits)
    log_one_minus_pi = torch.nn.functional.logsigmoid(-zero_logits)
    zero_mask = x.eq(0.0)
    log_zero = torch.logaddexp(
        log_pi,
        log_one_minus_pi + theta * log_theta_ratio,
    )
    log_positive = log_one_minus_pi + log_nb
    return -torch.where(zero_mask, log_zero, log_positive).mean(dim=0)


def parameter_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def effective_rank(cov: np.ndarray) -> float:
    eig = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    denom = float(np.square(eig).sum())
    return float(eig.sum() ** 2 / denom) if denom > 0 else float("nan")


def ridge_per_gene_r2(z: np.ndarray, x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    z_mean, z_sd = z.mean(0), z.std(0)
    z_sd[z_sd < 1e-8] = 1.0
    z = (z - z_mean) / z_sd
    gram = z.T @ z
    gram.flat[:: gram.shape[0] + 1] += alpha
    coef = np.linalg.solve(
        gram.astype(np.float64),
        z.T.astype(np.float64) @ x.astype(np.float64),
    )
    sse = np.zeros(x.shape[1], dtype=np.float64)
    for start in range(0, x.shape[0], 2048):
        stop = min(start + 2048, x.shape[0])
        pred = z[start:stop].astype(np.float64) @ coef
        sse += np.square(pred - x[start:stop]).sum(axis=0)
    sst = np.square(x).sum(axis=0, dtype=np.float64)
    r2 = np.zeros(x.shape[1], dtype=np.float64)
    varying = sst > 1e-12
    r2[varying] = 1.0 - sse[varying] / sst[varying]
    return r2.astype(np.float32)


def ridge_per_gene_r2_cv(
    z: np.ndarray,
    x: np.ndarray,
    n_splits: int = 5,
    seed: int = 20261121,
    alpha: float = 1.0,
) -> np.ndarray:
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
        z_mean, z_sd = z_train.mean(0), z_train.std(0)
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
        sse += np.square(pred - x_test).sum(0)
        sst += np.square(x_test).sum(0)
    r2 = np.zeros(x.shape[1], dtype=np.float64)
    varying = sst > 1e-12
    r2[varying] = 1 - sse[varying] / sst[varying]
    return r2.astype(np.float32)


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = ad.read_h5ad(H5AD)
    log_expr = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    log_expr = np.asarray(log_expr, dtype=np.float32)
    library = a.obs["total_counts"].to_numpy(dtype=np.float32)
    raw = np.rint(np.expm1(log_expr) * library[:, None] / 10000.0)
    raw = np.clip(raw, 0.0, None).astype(np.float32)
    mean = log_expr.mean(0)
    sd = log_expr.std(0)
    sd[sd < 1e-8] = 1.0
    log_expr = ((log_expr - mean) / sd).astype(np.float32)
    real = np.load(REAL_SUPPORT, allow_pickle=True)
    random = np.load(RANDOM_SUPPORT, allow_pickle=True)
    real_idx = np.asarray(real["idx"], dtype=np.int64)
    random_idx = np.asarray(random["idx"], dtype=np.int64)
    return log_expr, raw, library, real_idx, random_idx


def make_weight(n_genes: int, idx: np.ndarray) -> np.ndarray:
    weight = np.ones(n_genes, dtype=np.float32)
    weight[idx] = 1.0 + BOOST
    return (weight / weight.mean()).astype(np.float32)


def train_arm(
    log_expr: np.ndarray,
    raw: np.ndarray,
    library: np.ndarray,
    arm: str,
    seed: int,
    weight: np.ndarray | None,
    output: Path,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ZINBVAE(log_expr.shape[1]).to(device)
    initial_digest = parameter_digest(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(
        torch.from_numpy(log_expr),
        torch.from_numpy(raw),
        torch.from_numpy(library),
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH,
        shuffle=True,
        drop_last=True,
        generator=generator,
        num_workers=0,
    )
    weight_t = (
        torch.as_tensor(weight, dtype=torch.float32, device=device)
        if weight is not None
        else None
    )

    start = time.time()
    history = []
    for epoch in range(EPOCHS):
        model.train()
        total_loss = total_nll = total_kl = 0.0
        n_batches = 0
        for x_log, x_raw, lib in loader:
            x_log = x_log.to(device, non_blocking=True)
            x_raw = x_raw.to(device, non_blocking=True)
            lib = lib.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            log_mean, log_theta, zero_logits, mu, logvar = model(x_log)
            nll = zinb_nll(x_raw, lib, log_mean, log_theta, zero_logits)
            recon = (nll * weight_t).mean() if weight_t is not None else nll.mean()
            kl = -0.5 * torch.mean(1 + logvar - mu.square() - logvar.exp())
            loss = recon + BETA * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
            total_nll += float(recon.detach())
            total_kl += float(kl.detach())
            n_batches += 1
        scheduler.step()
        row = {
            "epoch": epoch,
            "loss": total_loss / n_batches,
            "weighted_nll": total_nll / n_batches,
            "kl": total_kl / n_batches,
        }
        history.append(row)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == EPOCHS:
            print(
                f"[{arm} seed={seed}] epoch {epoch+1:03d}/{EPOCHS} "
                f"loss={row['loss']:.4f} nll={row['weighted_nll']:.4f} "
                f"kl={row['kl']:.4f}",
                flush=True,
            )

    model.eval()
    zs = []
    log_tensor = torch.from_numpy(log_expr)
    with torch.no_grad():
        for start_row in range(0, log_expr.shape[0], 2048):
            zs.append(
                model.latent(
                    log_tensor[start_row : start_row + 2048].to(device)
                ).cpu().numpy()
            )
    z = np.concatenate(zs, axis=0).astype(np.float32)
    cov = np.cov(z.astype(np.float64), rowvar=False)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arm": arm,
            "seed": seed,
            "weight": weight,
            "architecture": "ZINB VAE (MLP encoder/decoder, LayerNorm, latent 64)",
        },
        output / "model.pt",
    )
    np.savez_compressed(
        output / "latent.npz",
        Z=z,
        weight=weight if weight is not None else np.ones(log_expr.shape[1], dtype=np.float32),
    )
    metrics = {
        "arm": arm,
        "seed": seed,
        "architecture": "ZINB VAE",
        "n_cells": int(log_expr.shape[0]),
        "n_genes": int(log_expr.shape[1]),
        "epochs": EPOCHS,
        "beta": BETA,
        "boost": BOOST if weight is not None else None,
        "weight_min": float(weight.min()) if weight is not None else 1.0,
        "weight_max": float(weight.max()) if weight is not None else 1.0,
        "weight_mean": float(weight.mean()) if weight is not None else 1.0,
        "initial_parameter_sha256": initial_digest,
        "trace_Z": float(np.trace(cov)),
        "eff_rank_Z": effective_rank(cov),
        "latent_sd": float(z.std()),
        "final_loss": history[-1]["loss"],
        "final_weighted_nll": history[-1]["weighted_nll"],
        "final_kl": history[-1]["kl"],
        "epoch_history": history,
        "seconds": float(time.time() - start),
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    del model, optimizer, scheduler, loader, dataset, log_tensor
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def summarize(log_expr, real_idx, random_idx):
    overlap = np.intersect1d(real_idx, random_idx)
    real = np.setdiff1d(real_idx, overlap)
    random = np.setdiff1d(random_idx, overlap)
    other = np.setdiff1d(np.arange(log_expr.shape[1]), np.union1d(real_idx, random_idx))
    per_arm = []
    r2s = {}
    r2s_cv = {}
    for seed in SEEDS:
        for arm in ARMS:
            out = ROOT / "runs" / f"seed_{seed}" / arm
            metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
            z = np.load(out / "latent.npz")["Z"]
            r2 = ridge_per_gene_r2(z, log_expr)
            r2_cv = ridge_per_gene_r2_cv(z, log_expr)
            np.save(out / "pergene_r2.npy", r2)
            np.save(out / "pergene_r2_cv.npy", r2_cv)
            r2s[(arm, seed)] = r2
            r2s_cv[(arm, seed)] = r2_cv
            per_arm.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "sum_R2": float(r2.sum()),
                    "sum_R2_cv": float(r2_cv.sum()),
                    "mean_R2_real_targets": float(r2[real].mean()),
                    "mean_R2_random_targets": float(r2[random].mean()),
                    "mean_R2_other": float(r2[other].mean()),
                    "trace_Z": metrics["trace_Z"],
                    "eff_rank_Z": metrics["eff_rank_Z"],
                    "healthy": bool(
                        metrics["eff_rank_Z"] >= HEALTH_MIN_EFF_RANK
                    and float(r2.sum()) >= HEALTH_MIN_SUM_R2
                    and metrics["latent_sd"] > HEALTH_MIN_LATENT_SD
                    and metrics["final_kl"] > HEALTH_MIN_KL
                        and math.isfinite(metrics["final_weighted_nll"])
                    ),
                }
            )

    paired = []
    for seed in SEEDS:
        empty, real_r2, random_r2 = [r2s[(a, seed)] for a in ARMS]
        empty_cv, real_cv, random_cv = [r2s_cv[(a, seed)] for a in ARMS]
        dr = real_r2 - empty
        dc = random_r2 - empty
        dr_cv = real_cv - empty_cv
        dc_cv = random_cv - empty_cv
        interaction = (
            dr[real].mean() - dr[random].mean()
            - dc[real].mean() + dc[random].mean()
        )
        interaction_cv = (
            dr_cv[real].mean() - dr_cv[random].mean()
            - dc_cv[real].mean() + dc_cv[random].mean()
        )
        paired.append(
            {
                "seed": seed,
                "real_prior_on_real_targets": float(dr[real].mean()),
                "real_prior_on_random_targets": float(dr[random].mean()),
                "random_prior_on_real_targets": float(dc[real].mean()),
                "random_prior_on_random_targets": float(dc[random].mean()),
                "real_vs_random_on_real_targets": float(
                    (real_r2[real] - random_r2[real]).mean()
                ),
                "real_vs_random_on_random_targets": float(
                    (real_r2[random] - random_r2[random]).mean()
                ),
                "interaction": float(interaction),
                "cv_real_vs_random_on_real_targets": float(
                    (real_cv[real] - random_cv[real]).mean()
                ),
                "cv_interaction": float(interaction_cv),
                "real_gt_random_on_real_genes": int(
                    (real_r2[real] > random_r2[real]).sum()
                ),
                "random_gt_real_on_random_genes": int(
                    (random_r2[random] > real_r2[random]).sum()
                ),
                "kappa_C_W_real": float(real_r2.sum() / empty.sum()),
                "kappa_C_W_random": float(random_r2.sum() / empty.sum()),
            }
        )
    return per_arm, paired, real, random


def aggregate(per_arm, paired):
    keys = (
        "real_vs_random_on_real_targets",
        "interaction",
        "cv_real_vs_random_on_real_targets",
        "cv_interaction",
        "kappa_C_W_real",
        "kappa_C_W_random",
    )
    stats = {}
    for key in keys:
        values = np.array([row[key] for row in paired], dtype=float)
        target = 1.0 if key.startswith("kappa") else 0.0
        test = scipy_stats.ttest_1samp(values, target)
        se = values.std(ddof=1) / math.sqrt(values.size)
        crit = scipy_stats.t.ppf(0.975, values.size - 1)
        stats[key] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
            "positive_seeds_vs_null": int(
                ((values > target) if key.startswith("kappa") else (values > 0)).sum()
            ),
            "paired_t": float(test.statistic),
            "two_sided_p": float(test.pvalue),
            "ci95": [
                float(values.mean() - crit * se),
                float(values.mean() + crit * se),
            ],
        }
    healthy = all(row["healthy"] for row in per_arm)
    passed = bool(
        healthy
        and stats["interaction"]["mean"] > 0
        and stats["real_vs_random_on_real_targets"]["positive_seeds_vs_null"] >= 2
        and stats["cv_interaction"]["mean"] > 0
        and abs(stats["kappa_C_W_real"]["mean"] - 1) < 0.10
        and abs(stats["kappa_C_W_random"]["mean"] - 1) < 0.10
    )
    return {
        "architecture": "ZINB VAE",
        "likelihood": "zero-inflated negative binomial on reconstructed raw counts",
        "dataset": "GSE150774 delivery matrix, 12,000 cells x 20,527 genes",
        "boost": BOOST,
        "epochs": EPOCHS,
        "beta": BETA,
        "n_real_targets": 202,
        "n_random_targets": 202,
        "n_other_genes": 20122,
        "per_arm": per_arm,
        "paired": paired,
        "stats": stats,
        "all_healthy": healthy,
        "cross_architecture_delivery_pass": passed,
        "decision_rule": (
            "PASS iff all arms pass latent-health checks; the content interaction "
            "is positive; the real-minus-random target effect is positive in >=2/3 "
            "seeds; fixed-latent cross-validated interaction is positive; and both "
            "aggregate decoder ratios remain within [0.90, 1.10]."
        ),
        "health_gate": (
            "ZINB-specific gate fixed after a no-prior calibration: effective rank "
            ">=2.5, sum R2 >=1000, latent sd >0.2, KL >0.1, finite weighted NLL."
        ),
    }


def main():
    global EPOCHS, SEEDS, BETA
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    EPOCHS = args.epochs
    BETA = args.beta
    SEEDS = tuple(args.seeds)
    torch.set_float32_matmul_precision("high")
    print(f"[data] {H5AD}", flush=True)
    log_expr, raw, library, real_idx, random_idx = load_data()
    real_w = make_weight(log_expr.shape[1], real_idx)
    random_w = make_weight(log_expr.shape[1], random_idx)
    print(
        f"[data] X={log_expr.shape}; raw_max={float(raw.max()):.0f}; "
        f"targets={real_idx.size}; random={random_idx.size}; "
        f"overlap={np.intersect1d(real_idx, random_idx).size}",
        flush=True,
    )
    print(
        f"[prior] target={real_w[real_idx[0]]:.3f}; background={real_w[0]:.3f}",
        flush=True,
    )
    for seed in SEEDS:
        for arm, weight in [
            ("empty", None),
            ("real", real_w),
            ("random", random_w),
        ]:
            if arm not in args.arms:
                continue
            out = ROOT / "runs" / f"seed_{seed}" / arm
            if not args.force and (out / "metrics.json").exists():
                print(f"[skip] seed={seed} arm={arm} complete", flush=True)
                continue
            train_arm(log_expr, raw, library, arm, seed, weight, out)
    if set(args.arms) != set(ARMS):
        print("[pilot] incomplete arm set; skipping aggregate decision", flush=True)
        return
    per_arm, paired, real, random = summarize(log_expr, real_idx, random_idx)
    result = aggregate(per_arm, paired)
    result["n_real_targets"] = int(real.size)
    result["n_random_targets"] = int(random.size)
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
