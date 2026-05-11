"""Generate the paper-style MUAD figures from the project code."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from einops import rearrange
from sklearn.metrics import roc_auc_score, roc_curve
from torch_uncertainty.metrics import CalibrationError

from src.scoring import HybridNECOEnergyOOD, ensemble_mean_probs, predictive_entropy
from src.utils.config import load_config
from src.utils.data import build_muad_dataloaders
from src.utils.device import get_device, seed_everything
from .benchmark import _load_model, collect_ensemble_entropy_scores, collect_scores_from_map, load_or_train_deep_ensemble
from .metrics import fpr_at_tpr, prepare_segmentation_labels


ID_COLOR = "#1f77b4"
OOD_COLOR = "#d62728"
ALT_COLOR = "#33a02c"


def _squeeze_labels(labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim == 4 and labels.shape[1] == 1:
        labels = labels.squeeze(1)
    return labels


def _squeeze_sample_label(labels: torch.Tensor) -> torch.Tensor:
    labels = _squeeze_labels(labels)
    if labels.ndim == 3 and labels.shape[0] == 1:
        labels = labels.squeeze(0)
    return labels


def _to_numpy_1d(values: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().float().numpy().reshape(-1)
    return np.asarray(values, dtype=np.float32).reshape(-1)


def _sample_for_plot(values: torch.Tensor | np.ndarray, max_points: int = 200_000, seed: int = 0) -> np.ndarray:
    arr = _to_numpy_1d(values)
    if arr.size <= max_points:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.size, size=max_points, replace=False)
    return arr[idx]


def _separation(id_arr: np.ndarray, ood_arr: np.ndarray) -> float:
    mu_id, mu_ood = float(id_arr.mean()), float(ood_arr.mean())
    std_id, std_ood = float(id_arr.std() + 1e-8), float(ood_arr.std() + 1e-8)
    return abs(mu_ood - mu_id) / (0.5 * (std_id + std_ood) + 1e-8)


def _robust_minmax(values: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(values, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _to_display_rgb(
    image: torch.Tensor,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    mean_t = torch.tensor(mean, device=image.device).view(3, 1, 1)
    std_t = torch.tensor(std, device=image.device).view(3, 1, 1)
    rgb = (image * std_t + mean_t).clamp(0, 1)
    return rgb.permute(1, 2, 0).detach().cpu().numpy()


def _save_figure(fig: plt.Figure, output_dir: Path, stems: list[str]) -> list[str]:
    saved: list[str] = []
    for stem in stems:
        for suffix, kwargs in [
            ("pdf", {"format": "pdf"}),
            ("png", {"format": "png"}),
        ]:
            path = output_dir / f"{stem}.{suffix}"
            fig.savefig(path, dpi=300, bbox_inches="tight", **kwargs)
            saved.append(str(path))
    plt.close(fig)
    return saved


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _binary_summary(name: str, id_scores: torch.Tensor, ood_scores: torch.Tensor) -> dict[str, float | int | str]:
    if id_scores.numel() == 0 or ood_scores.numel() == 0:
        return {
            "method": name,
            "auroc": math.nan,
            "id_mean": math.nan,
            "ood_mean": math.nan,
            "id_count": int(id_scores.numel()),
            "ood_count": int(ood_scores.numel()),
        }
    id_arr = _to_numpy_1d(id_scores)
    ood_arr = _to_numpy_1d(ood_scores)
    y_true = np.concatenate([np.zeros_like(id_arr, dtype=np.int64), np.ones_like(ood_arr, dtype=np.int64)])
    y_score = np.concatenate([id_arr, ood_arr])
    return {
        "method": name,
        "auroc": float(roc_auc_score(y_true, y_score)),
        "id_mean": float(id_arr.mean()),
        "ood_mean": float(ood_arr.mean()),
        "id_count": int(id_arr.size),
        "ood_count": int(ood_arr.size),
    }


def _collect_paper_scores(
    loaders: dict[str, Any],
    detector: HybridNECOEnergyOOD,
    ensemble: torch.nn.Module,
    num_estimators: int,
    num_classes: int,
    device: torch.device,
    alpha: float,
    max_batches: int | None,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    stats = detector.stats
    if stats is None:
        raise RuntimeError("Hybrid detector must be fitted before collecting paper scores.")

    id_hybrid = collect_scores_from_map(loaders["test"], detector, "ood_score", num_classes, alpha, True, max_batches)
    ood_hybrid = collect_scores_from_map(loaders["ood"], detector, "ood_score", num_classes, alpha, False, max_batches)

    id_neco = collect_scores_from_map(loaders["test"], detector, "neco", num_classes, alpha, True, max_batches)
    ood_neco = collect_scores_from_map(loaders["ood"], detector, "neco", num_classes, alpha, False, max_batches)
    id_neco = -((id_neco - stats.neco_mean.detach().cpu()) / (stats.neco_std.detach().cpu() + 1e-8))
    ood_neco = -((ood_neco - stats.neco_mean.detach().cpu()) / (stats.neco_std.detach().cpu() + 1e-8))

    id_energy = collect_scores_from_map(loaders["test"], detector, "energy", num_classes, alpha, True, max_batches)
    ood_energy = collect_scores_from_map(loaders["ood"], detector, "energy", num_classes, alpha, False, max_batches)
    id_energy = -((id_energy - stats.energy_mean.detach().cpu()) / (stats.energy_std.detach().cpu() + 1e-8))
    ood_energy = -((ood_energy - stats.energy_mean.detach().cpu()) / (stats.energy_std.detach().cpu() + 1e-8))

    id_ensemble = collect_ensemble_entropy_scores(
        loaders["test"], ensemble, num_estimators, num_classes, device, True, max_batches
    )
    ood_ensemble = collect_ensemble_entropy_scores(
        loaders["ood"], ensemble, num_estimators, num_classes, device, False, max_batches
    )

    return {
        "Ensemble": (id_ensemble, ood_ensemble),
        "NECO-only": (id_neco, ood_neco),
        "Energy-only": (id_energy, ood_energy),
        "Hybrid": (id_hybrid, ood_hybrid),
    }


@torch.no_grad()
def _plot_calibration(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    num_bins: int,
    output_dir: Path,
    max_batches: int | None,
) -> tuple[float, list[str]]:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    metric = CalibrationError(task="multiclass", num_classes=num_classes, num_bins=num_bins, norm="l1").to(device)
    model.eval()
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)
        labels = prepare_segmentation_labels(labels.to(device), num_classes, ignore_index)
        logits = model(images)
        probs = logits.softmax(dim=1)
        flat_probs = rearrange(probs, "b c h w -> (b h w) c")
        flat_labels = labels.flatten()
        valid = flat_labels != ignore_index
        if valid.any():
            metric.update(flat_probs[valid], flat_labels[valid])

    ece = float(metric.compute().detach().cpu())
    fig, _ = metric.plot()
    saved = _save_figure(fig, output_dir, ["Baseline_Calibration"])
    return ece, saved


@torch.no_grad()
def _plot_ensemble_calibration(
    ensemble: torch.nn.Module,
    loader: Any,
    device: torch.device,
    num_estimators: int,
    num_classes: int,
    ignore_index: int,
    num_bins: int,
    output_dir: Path,
    max_batches: int | None,
) -> tuple[float, list[str]]:
    metric = CalibrationError(task="multiclass", num_classes=num_classes, num_bins=num_bins, norm="l1").to(device)
    ensemble.eval()
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)
        labels = prepare_segmentation_labels(labels.to(device), num_classes, ignore_index)
        outputs = ensemble(images)
        mean_probs = ensemble_mean_probs(outputs, batch_size=images.shape[0], num_estimators=num_estimators)
        flat_probs = rearrange(mean_probs, "b c h w -> (b h w) c")
        flat_labels = labels.flatten()
        valid = flat_labels != ignore_index
        if valid.any():
            metric.update(flat_probs[valid], flat_labels[valid])

    ece = float(metric.compute().detach().cpu())
    fig, _ = metric.plot()
    saved = _save_figure(fig, output_dir, ["Deep_Ensembles_ECE"])
    return ece, saved


@torch.no_grad()
def _pick_random_ood_indices(dataset: Any, num_classes: int, n: int = 3, seed: int = 41) -> list[int]:
    valid_idx: list[int] = []
    for index in range(len(dataset)):
        _, labels = dataset[index]
        labels = _squeeze_sample_label(labels)
        valid = labels != 255
        if valid.any() and ((labels >= num_classes) & valid).any():
            valid_idx.append(index)

    if len(valid_idx) <= n:
        return valid_idx
    rng = np.random.RandomState(seed)
    return rng.choice(valid_idx, size=n, replace=False).tolist()


@torch.no_grad()
def _ensemble_entropy_map(
    ensemble: torch.nn.Module,
    images: torch.Tensor,
    num_estimators: int,
) -> torch.Tensor:
    outputs = ensemble(images)
    probs = ensemble_mean_probs(outputs, batch_size=images.shape[0], num_estimators=num_estimators)
    return predictive_entropy(probs)


def _pick_ood_sample(dataset: Any, num_classes: int) -> int:
    fallback_idx = 0
    for index in range(len(dataset)):
        _, labels = dataset[index]
        labels = labels if torch.is_tensor(labels) else torch.as_tensor(labels)
        labels = _squeeze_sample_label(labels)
        valid = labels != 255
        if ((labels >= num_classes) & valid).any():
            return index
        if valid.any():
            fallback_idx = index
    return fallback_idx


@torch.no_grad()
def _plot_single_scene_qualitative(
    dataset: Any,
    detector: HybridNECOEnergyOOD,
    ensemble: torch.nn.Module,
    num_estimators: int,
    num_classes: int,
    device: torch.device,
    output_dir: Path,
    alpha: float,
) -> tuple[int, list[str]]:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.autolayout": True,
        }
    )
    sample_idx = _pick_ood_sample(dataset, num_classes)
    image, labels = dataset[sample_idx]
    labels = _squeeze_sample_label(labels)
    batch = image.unsqueeze(0).to(device)

    ensemble.eval()
    detector.model.eval()
    entropy = _ensemble_entropy_map(ensemble, batch, num_estimators).squeeze(0)
    hybrid = detector.score_maps(batch, alpha=alpha)["ood_score"].squeeze(0)

    labels = labels.to(device)
    valid = labels != 255
    true_ood_mask = ((labels >= num_classes) & valid).float().detach().cpu().numpy()
    entropy_vis = _robust_minmax(entropy.detach().cpu().numpy())
    hybrid_vis = _robust_minmax(hybrid.detach().cpu().numpy())

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)
    axes[0].imshow(_to_display_rgb(image.to(device)))
    axes[0].set_title("Input Image", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(true_ood_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("True OOD Mask", fontweight="bold")
    axes[1].axis("off")

    im2 = axes[2].imshow(entropy_vis, cmap="inferno", vmin=0, vmax=1)
    axes[2].set_title("Ensemble Predictive Entropy", fontweight="bold")
    axes[2].axis("off")
    cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=12)

    im3 = axes[3].imshow(hybrid_vis, cmap="magma", vmin=0, vmax=1)
    axes[3].set_title("Hybrid OOD Heatmap", fontweight="bold")
    axes[3].axis("off")
    cbar3 = plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    cbar3.ax.tick_params(labelsize=12)

    saved = _save_figure(fig, output_dir, ["OOD_Qualitative_Comparison"])
    return sample_idx, saved


@torch.no_grad()
def _plot_multiscene(
    dataset: Any,
    detector: HybridNECOEnergyOOD,
    ensemble: torch.nn.Module,
    num_estimators: int,
    num_classes: int,
    device: torch.device,
    output_dir: Path,
    num_scenes: int,
    seed: int,
    alpha: float,
) -> tuple[list[int], list[str]]:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    indices = _pick_random_ood_indices(dataset, num_classes=num_classes, n=num_scenes, seed=seed)
    if not indices:
        raise RuntimeError("No OOD samples were available for the qualitative figure.")

    rows = len(indices)
    fig, axes = plt.subplots(rows, 4, figsize=(16, 3.0 * rows), constrained_layout=True)
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    ensemble.eval()
    detector.model.eval()
    for row, sample_idx in enumerate(indices):
        image, labels = dataset[sample_idx]
        labels = _squeeze_sample_label(labels)
        batch = image.unsqueeze(0).to(device)

        entropy = _ensemble_entropy_map(ensemble, batch, num_estimators).squeeze(0)
        hybrid = detector.score_maps(batch, alpha=alpha)["ood_score"].squeeze(0)

        valid = labels != 255
        ood_mask = ((labels >= num_classes) & valid).float().cpu().numpy()
        entropy_vis = _robust_minmax(entropy.detach().cpu().numpy())
        hybrid_vis = _robust_minmax(hybrid.detach().cpu().numpy())

        axes[row, 0].imshow(_to_display_rgb(image.to(device)))
        axes[row, 0].set_title("Input" if row == 0 else "")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(ood_mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("GT OOD mask" if row == 0 else "")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(entropy_vis, cmap="inferno", vmin=0, vmax=1)
        axes[row, 2].set_title("Ensemble entropy" if row == 0 else "")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(hybrid_vis, cmap="magma", vmin=0, vmax=1)
        axes[row, 3].set_title("Hybrid OOD score" if row == 0 else "")
        axes[row, 3].axis("off")

    fig.suptitle("Multi-scene qualitative OOD comparison", y=1.003, fontsize=15, fontweight="bold")
    saved = _save_figure(fig, output_dir, ["H1_MultiScene_Qualitative"])
    return indices, saved


def _plot_score_distributions(
    scores: dict[str, tuple[torch.Tensor, torch.Tensor]],
    output_dir: Path,
) -> tuple[list[dict[str, float | str]], list[str]]:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.2), constrained_layout=True)
    sep_summary: list[dict[str, float | str]] = []

    for idx, (name, (id_scores, ood_scores)) in enumerate(scores.items()):
        ax = axes[idx]
        id_np = _sample_for_plot(id_scores, max_points=200_000, seed=10 + idx)
        ood_np = _sample_for_plot(ood_scores, max_points=200_000, seed=20 + idx)
        if id_np.size == 0 or ood_np.size == 0:
            ax.set_title(f"{name}\n(N/A)", fontweight="bold")
            ax.axis("off")
            sep_summary.append({"method": name, "separation": math.nan})
            continue

        lo = min(np.percentile(id_np, 0.5), np.percentile(ood_np, 0.5))
        hi = max(np.percentile(id_np, 99.5), np.percentile(ood_np, 99.5))
        if hi <= lo:
            hi = lo + 1e-6
        bins = np.linspace(lo, hi, 80)

        ax.hist(id_np, bins=bins, density=True, alpha=0.55, color=ID_COLOR, edgecolor="none")
        ax.hist(ood_np, bins=bins, density=True, alpha=0.55, color=OOD_COLOR, edgecolor="none")
        sep = _separation(id_np, ood_np)
        sep_summary.append({"method": name, "separation": sep})

        ax.set_title(f"{name} (Sep: {sep:.3f})", fontweight="bold")
        ax.set_xlabel("OOD Score")
        if idx == 0:
            ax.set_ylabel("Density")
        else:
            ax.set_yticklabels([])
        ax.grid(alpha=0.3, linestyle="--")

    id_patch = mpatches.Patch(color=ID_COLOR, alpha=0.55, label="In-Distribution (ID)")
    ood_patch = mpatches.Patch(color=OOD_COLOR, alpha=0.55, label="Out-of-Distribution (OOD)")
    fig.legend(
        handles=[id_patch, ood_patch],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        frameon=False,
    )
    saved = _save_figure(fig, output_dir, ["OOD_Scores_Distribution"])
    return sorted(sep_summary, key=lambda row: np.nan_to_num(float(row["separation"]), nan=-1.0), reverse=True), saved


def _curve_from_scores(id_scores: torch.Tensor, ood_scores: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    id_arr = _to_numpy_1d(id_scores)
    ood_arr = _to_numpy_1d(ood_scores)
    y_true = np.concatenate([np.zeros_like(id_arr, dtype=np.int64), np.ones_like(ood_arr, dtype=np.int64)])
    y_score = np.concatenate([id_arr, ood_arr])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return y_true, y_score, fpr, tpr


def _plot_roc_tail(scores: dict[str, tuple[torch.Tensor, torch.Tensor]], output_dir: Path) -> tuple[dict[str, float], list[str]]:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    y_e, s_e, fpr_e, tpr_e = _curve_from_scores(*scores["Ensemble"])
    y_h, s_h, fpr_h, tpr_h = _curve_from_scores(*scores["Hybrid"])
    fpr95_e = fpr_at_tpr(y_e, s_e, 0.95)
    fpr98_e = fpr_at_tpr(y_e, s_e, 0.98)
    fpr95_h = fpr_at_tpr(y_h, s_h, 0.95)
    fpr98_h = fpr_at_tpr(y_h, s_h, 0.98)

    fig, axes = plt.subplots(2, 1, figsize=(6, 8), constrained_layout=True)
    axes[0].plot(fpr_e, tpr_e, color=ID_COLOR, lw=2, label="Ensemble")
    axes[0].plot(fpr_h, tpr_h, color=OOD_COLOR, lw=2, label="Hybrid")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_title("(a) Full ROC")
    axes[0].set_xlabel("False Positive Rate (FPR)")
    axes[0].set_ylabel("True Positive Rate (TPR)")
    axes[0].grid(alpha=0.3, linestyle="--")
    axes[0].legend(loc="lower right")

    x_max = max(v for v in [fpr98_e, fpr98_h, 1e-6] if not math.isnan(v))
    axes[1].plot(fpr_e, tpr_e, color=ID_COLOR, lw=2, label="Ensemble")
    axes[1].plot(fpr_h, tpr_h, color=OOD_COLOR, lw=2, label="Hybrid")
    axes[1].set_xlim(0.0, min(1.0, x_max * 1.35 + 1e-6))
    axes[1].set_ylim(0.94, 1.001)
    axes[1].axhline(0.95, color="gray", ls="--", lw=1)
    axes[1].axhline(0.98, color="gray", ls=":", lw=1)
    axes[1].scatter([fpr95_e, fpr98_e], [0.95, 0.98], color=ID_COLOR, s=40)
    axes[1].scatter([fpr95_h, fpr98_h], [0.95, 0.98], color=OOD_COLOR, s=40)
    axes[1].set_title("(b) High-TPR tail zoom (TPR > 0.95)")
    axes[1].set_xlabel("False Positive Rate (FPR)")
    axes[1].set_ylabel("True Positive Rate (TPR)")
    axes[1].grid(alpha=0.3, linestyle="--")

    fig.suptitle("ROC and operating-point behavior", y=1.03, fontsize=15, fontweight="bold")
    saved = _save_figure(fig, output_dir, ["H3_ROC_TailZoom"])
    metrics = {
        "ensemble_fpr95": float(fpr95_e),
        "ensemble_fpr98": float(fpr98_e),
        "hybrid_fpr95": float(fpr95_h),
        "hybrid_fpr98": float(fpr98_h),
    }
    return metrics, saved


def _scene_features(
    image: torch.Tensor,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> tuple[float, float]:
    rgb = torch.as_tensor(_to_display_rgb(image, mean=mean, std=std))
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return float(gray.mean().item()), float(gray.std().item())


def _condition_thresholds(dataset: Any, max_scenes: int | None) -> tuple[float, float]:
    br_list: list[float] = []
    ct_list: list[float] = []
    limit = len(dataset) if max_scenes is None else min(len(dataset), max_scenes)
    for index in range(limit):
        image, _ = dataset[index]
        br, ct = _scene_features(image)
        br_list.append(br)
        ct_list.append(ct)
    if not br_list:
        return 0.5, 0.0
    return float(np.median(br_list)), float(np.median(ct_list))


def _assign_conditions(image: torch.Tensor, br_thr: float, ct_thr: float) -> list[str]:
    br, ct = _scene_features(image)
    return [
        "low_light" if br < br_thr else "high_light",
        "low_contrast" if ct < ct_thr else "high_contrast",
    ]


@torch.no_grad()
def _condition_scores(
    method: str,
    loaders: dict[str, Any],
    detector: HybridNECOEnergyOOD,
    ensemble: torch.nn.Module,
    num_estimators: int,
    num_classes: int,
    device: torch.device,
    alpha: float,
    br_thr: float,
    ct_thr: float,
    max_batches: int | None,
    max_ood_scenes: int | None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    cond_names = ["low_light", "high_light", "low_contrast", "high_contrast"]
    buckets = {name: {"id": [], "ood": []} for name in cond_names}

    for batch_idx, (images, labels) in enumerate(loaders["test"]):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)
        labels = _squeeze_labels(labels.to(device))
        if method == "hybrid":
            score_map = detector.score_maps(images, alpha=alpha)["ood_score"]
        else:
            score_map = _ensemble_entropy_map(ensemble, images, num_estimators)

        for b_idx in range(images.shape[0]):
            flat_labels = labels[b_idx].flatten()
            flat_scores = score_map[b_idx].flatten()
            keep = (flat_labels != 255) & (flat_labels < num_classes)
            if not keep.any():
                continue
            arr = flat_scores[keep].detach().cpu().numpy()
            for cond in _assign_conditions(images[b_idx].detach().cpu(), br_thr, ct_thr):
                buckets[cond]["id"].append(arr)

    dataset = loaders["ood"].dataset
    limit = len(dataset) if max_ood_scenes is None else min(len(dataset), max_ood_scenes)
    for index in range(limit):
        image, labels = dataset[index]
        labels = _squeeze_sample_label(labels).to(device)
        batch = image.unsqueeze(0).to(device)
        if method == "hybrid":
            score_map = detector.score_maps(batch, alpha=alpha)["ood_score"].squeeze(0)
        else:
            score_map = _ensemble_entropy_map(ensemble, batch, num_estimators).squeeze(0)

        flat_labels = labels.flatten()
        flat_scores = score_map.flatten()
        valid = flat_labels != 255
        ood = valid & (flat_labels >= num_classes)
        keep = ood if ood.any() else valid
        if not keep.any():
            continue
        arr = flat_scores[keep].detach().cpu().numpy()
        for cond in _assign_conditions(image.detach().cpu(), br_thr, ct_thr):
            buckets[cond]["ood"].append(arr)

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cond in cond_names:
        id_arr = np.concatenate(buckets[cond]["id"]) if buckets[cond]["id"] else np.array([], dtype=np.float32)
        ood_arr = np.concatenate(buckets[cond]["ood"]) if buckets[cond]["ood"] else np.array([], dtype=np.float32)
        out[cond] = (id_arr, ood_arr)
    return out


def _condition_metrics(id_arr: np.ndarray, ood_arr: np.ndarray) -> tuple[float, float, float]:
    if id_arr.size == 0 or ood_arr.size == 0:
        return math.nan, math.nan, math.nan
    y_true = np.concatenate([np.zeros_like(id_arr, dtype=np.int64), np.ones_like(ood_arr, dtype=np.int64)])
    y_score = np.concatenate([id_arr, ood_arr])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.trapezoid(tpr, fpr)), fpr_at_tpr(y_true, y_score, 0.95), fpr_at_tpr(y_true, y_score, 0.98)


def _plot_condition_bars(
    loaders: dict[str, Any],
    detector: HybridNECOEnergyOOD,
    ensemble: torch.nn.Module,
    num_estimators: int,
    num_classes: int,
    device: torch.device,
    alpha: float,
    output_dir: Path,
    max_batches: int | None,
    max_ood_scenes: int | None,
) -> tuple[dict[str, dict[str, dict[str, float]]], list[str]]:
    br_thr, ct_thr = _condition_thresholds(loaders["ood"].dataset, max_ood_scenes)
    cond_names = ["low_light", "high_light", "low_contrast", "high_contrast"]
    cond_hybrid = _condition_scores(
        "hybrid", loaders, detector, ensemble, num_estimators, num_classes, device, alpha, br_thr, ct_thr, max_batches, max_ood_scenes
    )
    cond_ensemble = _condition_scores(
        "ensemble", loaders, detector, ensemble, num_estimators, num_classes, device, alpha, br_thr, ct_thr, max_batches, max_ood_scenes
    )

    metrics = ["AUROC", "FPR95", "FPR98"]
    values = {
        "Hybrid": {metric: [] for metric in metrics},
        "Ensemble": {metric: [] for metric in metrics},
    }
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for cond in cond_names:
        au_h, f95_h, f98_h = _condition_metrics(*cond_hybrid[cond])
        au_e, f95_e, f98_e = _condition_metrics(*cond_ensemble[cond])
        values["Hybrid"]["AUROC"].append(au_h)
        values["Hybrid"]["FPR95"].append(f95_h)
        values["Hybrid"]["FPR98"].append(f98_h)
        values["Ensemble"]["AUROC"].append(au_e)
        values["Ensemble"]["FPR95"].append(f95_e)
        values["Ensemble"]["FPR98"].append(f98_e)
        summary[cond] = {
            "Hybrid": {"AUROC": au_h, "FPR95": f95_h, "FPR98": f98_h},
            "Ensemble": {"AUROC": au_e, "FPR95": f95_e, "FPR98": f98_e},
        }

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    x = np.arange(len(cond_names))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        y_e = np.asarray(values["Ensemble"][metric], dtype=float)
        y_h = np.asarray(values["Hybrid"][metric], dtype=float)
        if metric.startswith("FPR"):
            y_e = 100.0 * y_e
            y_h = 100.0 * y_h
        ax.bar(x - width / 2, y_e, width=width, label="Ensemble", color=ID_COLOR, alpha=0.9)
        ax.bar(x + width / 2, y_h, width=width, label="Hybrid", color=OOD_COLOR, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(cond_names, rotation=25, ha="right")
        ax.set_title(metric)
        ax.grid(alpha=0.25, linestyle="--")
        if metric == "AUROC":
            ax.set_ylim(0.0, 1.0)
        else:
            ax.set_ylabel("%")
    axes[0].legend(loc="best")
    fig.suptitle("Condition-wise robustness (proxy conditions)", y=1.03, fontsize=15, fontweight="bold")
    saved = _save_figure(fig, output_dir, ["H2_ConditionWise_Bars"])
    return summary, saved


@torch.no_grad()
def _collect_feature_points(
    detector: HybridNECOEnergyOOD,
    loader: Any,
    num_classes: int,
    device: torch.device,
    per_group: int,
    id_split: bool,
    max_batches: int | None,
) -> torch.Tensor:
    points: list[torch.Tensor] = []
    total = 0
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)
        labels = _squeeze_labels(labels.to(device))
        _, feats = detector._forward_logits_and_feats(images)
        flat_feats = rearrange(feats, "b d h w -> (b h w) d")
        flat_labels = labels.flatten()
        valid = flat_labels != 255
        keep = valid & (flat_labels < num_classes) if id_split else valid & (flat_labels >= num_classes)
        if (not id_split) and (not keep.any()):
            keep = valid
        selected = flat_feats[keep]
        if selected.numel() == 0:
            continue
        remaining = per_group - total
        take = min(remaining, selected.shape[0])
        perm = torch.randperm(selected.shape[0], device=selected.device)[:take]
        points.append(selected[perm].detach().cpu())
        total += take
        if total >= per_group:
            break
    if not points:
        return torch.empty(0, detector.stats.mu.shape[1] if detector.stats else 0)
    return torch.cat(points, dim=0)


@torch.no_grad()
def _fit_class_centers(
    detector: HybridNECOEnergyOOD,
    loader: Any,
    num_classes: int,
    device: torch.device,
    max_pixels: int,
    max_batches: int | None,
) -> torch.Tensor:
    sums: torch.Tensor | None = None
    counts = torch.zeros(num_classes, dtype=torch.float64)
    seen = 0
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)
        labels = prepare_segmentation_labels(labels.to(device), num_classes, 255)
        _, feats = detector._forward_logits_and_feats(images)
        flat_feats = rearrange(feats, "b d h w -> (b h w) d").detach().cpu().double()
        flat_labels = labels.flatten().detach().cpu()
        valid = flat_labels != 255
        if not valid.any():
            continue
        if sums is None:
            sums = torch.zeros((num_classes, flat_feats.shape[1]), dtype=torch.float64)
        for cls in range(num_classes):
            mask = valid & (flat_labels == cls)
            if mask.any():
                sums[cls] += flat_feats[mask].sum(dim=0)
                counts[cls] += int(mask.sum())
        seen += int(valid.sum())
        if seen >= max_pixels:
            break
    if sums is None or not (counts > 0).any():
        raise RuntimeError("No ID features were available for feature-geometry centers.")
    return (sums[counts > 0] / counts[counts > 0].unsqueeze(1)).float()


def _embed_2d(features: np.ndarray, perplexity: float = 30.0) -> np.ndarray:
    if features.shape[0] < 5:
        centered = features - features.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        emb = centered @ vh[: min(2, vh.shape[0])].T
        if emb.shape[1] == 1:
            emb = np.concatenate([emb, np.zeros_like(emb)], axis=1)
        return emb
    try:
        from sklearn.manifold import TSNE

        perplexity = min(perplexity, max(2.0, (features.shape[0] - 1) / 3.0))
        return TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=42,
        ).fit_transform(features)
    except Exception:
        centered = features - features.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        return centered @ vh[:2].T


def _balanced_numpy_sample(features: torch.Tensor, n_vis: int, rng: np.random.RandomState) -> np.ndarray:
    arr = features.detach().cpu().float().numpy()
    if arr.shape[0] == 0:
        return arr
    indices = rng.choice(arr.shape[0], size=min(n_vis, arr.shape[0]), replace=False)
    return arr[indices]


def _nearest_center_distances(features: np.ndarray, centers: torch.Tensor) -> np.ndarray:
    if features.size == 0:
        return np.array([], dtype=np.float32)
    feats_t = torch.from_numpy(features).float()
    centers_t = centers.detach().cpu().float()
    return torch.cdist(feats_t, centers_t).min(dim=1).values.numpy()


def _safe_id_auroc(id_distances: np.ndarray, ood_distances: np.ndarray) -> float:
    if id_distances.size == 0 or ood_distances.size == 0:
        return math.nan
    y_true = np.concatenate([np.ones(id_distances.size, dtype=np.int64), np.zeros(ood_distances.size, dtype=np.int64)])
    y_score = np.concatenate([-id_distances, -ood_distances])
    return float(roc_auc_score(y_true, y_score))


def _plot_density(ax: plt.Axes, values: np.ndarray, color: str, label: str) -> None:
    if values.size >= 2 and float(np.std(values)) > 0:
        sns.kdeplot(values, ax=ax, color=color, label=label, fill=True, alpha=0.20, linewidth=1.0)
    elif values.size:
        ax.hist(values, bins=1, density=True, alpha=0.20, color=color, label=label)


def _plot_feature_geometry(
    loaders: dict[str, Any],
    detector: HybridNECOEnergyOOD,
    num_classes: int,
    device: torch.device,
    output_dir: Path,
    per_group: int,
    max_batches: int | None,
) -> tuple[list[str], dict[str, int], dict[str, float]]:
    if detector.stats is None:
        raise RuntimeError("Detector must be fitted before plotting feature geometry.")

    id_feats = _collect_feature_points(detector, loaders["test"], num_classes, device, per_group, True, max_batches)
    ood_feats = _collect_feature_points(detector, loaders["ood"], num_classes, device, per_group * 2, False, max_batches)
    if id_feats.numel() == 0 or ood_feats.numel() == 0:
        return [], {"ID": int(id_feats.shape[0]), "OOD-low-NECO": 0, "OOD-high-NECO": 0}, {}

    stats = detector.stats
    ood_dev = ood_feats.to(device)
    centered = ood_dev - stats.mu
    proj = centered @ stats.basis
    neco = torch.norm(proj, dim=1) / (torch.norm(centered, dim=1) + detector.eps)
    neco_ood_score = -((neco - stats.neco_mean) / (stats.neco_std + 1e-8))
    neco_ood_score_cpu = neco_ood_score.detach().cpu()
    threshold = torch.median(neco_ood_score_cpu)
    low = ood_feats[neco_ood_score_cpu <= threshold]
    high = ood_feats[neco_ood_score_cpu > threshold]
    low = low[:per_group] if low.shape[0] else ood_feats[:per_group]
    high = high[:per_group] if high.shape[0] else ood_feats[-per_group:]
    id_feats = id_feats[:per_group]

    centers = _fit_class_centers(detector, loaders["val"], num_classes, device, max_pixels=100_000, max_batches=max_batches)

    rng = np.random.RandomState(42)
    n_vis = per_group
    id_vis = _balanced_numpy_sample(id_feats, n_vis=n_vis, rng=rng)
    low_vis = _balanced_numpy_sample(low, n_vis=n_vis, rng=rng)
    high_vis = _balanced_numpy_sample(high, n_vis=n_vis, rng=rng)
    combined = np.vstack([id_vis, low_vis, high_vis])
    labels = np.concatenate(
        [
            np.zeros(len(id_vis), dtype=int),
            np.ones(len(low_vis), dtype=int),
            np.full(len(high_vis), 2, dtype=int),
        ]
    )
    emb = _embed_2d(combined, perplexity=30.0)

    id_distances = _nearest_center_distances(id_vis, centers)
    low_distances = _nearest_center_distances(low_vis, centers)
    high_distances = _nearest_center_distances(high_vis, centers)
    id_mean = float(np.mean(id_distances)) if id_distances.size else math.nan
    low_mean = float(np.mean(low_distances)) if low_distances.size else math.nan
    high_mean = float(np.mean(high_distances)) if high_distances.size else math.nan
    denom = id_mean if np.isfinite(id_mean) and abs(id_mean) > 1e-12 else math.nan
    low_ratio = float(low_mean / denom) if np.isfinite(denom) and np.isfinite(low_mean) else math.nan
    high_ratio = float(high_mean / denom) if np.isfinite(denom) and np.isfinite(high_mean) else math.nan
    low_auroc = _safe_id_auroc(id_distances, low_distances)
    high_auroc = _safe_id_auroc(id_distances, high_distances)

    id_norms = np.linalg.norm(id_vis, axis=1)
    low_norms = np.linalg.norm(low_vis, axis=1)
    high_norms = np.linalg.norm(high_vis, axis=1)

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    col_id = "#1f77b4"
    col_low = "#ff7f0e"
    col_high = "#2ca02c"
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35), constrained_layout=True)

    mask_id = labels == 0
    mask_low = labels == 1
    mask_high = labels == 2
    axes[0].scatter(emb[mask_id, 0], emb[mask_id, 1], s=3, alpha=0.35, c=col_id, label="ID")
    axes[0].scatter(emb[mask_low, 0], emb[mask_low, 1], s=3, alpha=0.35, c=col_low, label="OOD-low-NECO")
    axes[0].scatter(emb[mask_high, 0], emb[mask_high, 1], s=3, alpha=0.35, c=col_high, label="OOD-high-NECO")
    axes[0].set_title("(A) Feature Geometry (t-SNE)")
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")
    axes[0].legend(frameon=True, loc="best", ncol=3, columnspacing=0.7, handletextpad=0.3)

    _plot_density(axes[1], id_distances, col_id, "ID")
    _plot_density(axes[1], low_distances, col_low, "OOD-low-NECO")
    _plot_density(axes[1], high_distances, col_high, "OOD-high-NECO")
    axes[1].axvline(id_mean, color=col_id, linestyle="--", linewidth=0.8)
    axes[1].axvline(low_mean, color=col_low, linestyle="--", linewidth=0.8)
    axes[1].axvline(high_mean, color=col_high, linestyle="--", linewidth=0.8)
    axes[1].set_title("(B) Distance to Nearest Center")
    axes[1].set_xlabel("Distance")
    axes[1].set_ylabel("Density")
    axes[1].legend(frameon=True, loc="best")

    _plot_density(axes[2], id_norms, col_id, "ID")
    _plot_density(axes[2], low_norms, col_low, "OOD-low-NECO")
    _plot_density(axes[2], high_norms, col_high, "OOD-high-NECO")
    axes[2].axvline(float(np.mean(id_norms)), color=col_id, linestyle="--", linewidth=0.8)
    axes[2].axvline(float(np.mean(low_norms)), color=col_low, linestyle="--", linewidth=0.8)
    axes[2].axvline(float(np.mean(high_norms)), color=col_high, linestyle="--", linewidth=0.8)
    axes[2].set_title("(C) Feature Norm Distribution")
    axes[2].set_xlabel("Feature norm (L2)")
    axes[2].set_ylabel("Density")
    axes[2].legend(frameon=True, loc="best")

    fig.suptitle("NC5 Analysis", fontsize=9.5, fontweight="bold")

    saved = []
    for suffix, kwargs in [
        ("pdf", {"format": "pdf"}),
        ("png", {"format": "png", "dpi": 600}),
    ]:
        path = output_dir / f"NC5_Feature_Geometry.{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.01, **kwargs)
        saved.append(str(path))
    plt.close(fig)

    counts = {
        "ID": int(id_vis.shape[0]),
        "OOD-low-NECO": int(low_vis.shape[0]),
        "OOD-high-NECO": int(high_vis.shape[0]),
    }
    metrics = {
        "id_mean_distance": id_mean,
        "ood_low_neco_mean_distance": low_mean,
        "ood_high_neco_mean_distance": high_mean,
        "ood_low_neco_distance_ratio": low_ratio,
        "ood_high_neco_distance_ratio": high_ratio,
        "ood_low_neco_auroc": low_auroc,
        "ood_high_neco_auroc": high_auroc,
        "id_mean_feature_norm": float(np.mean(id_norms)),
        "ood_low_neco_mean_feature_norm": float(np.mean(low_norms)),
        "ood_high_neco_mean_feature_norm": float(np.mean(high_norms)),
    }
    return saved, counts, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the figures and table used by the Energy-Aware NECO paper.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-condition-scenes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-scenes", type=int, default=3)
    parser.add_argument("--scene-seed", type=int, default=41)
    parser.add_argument("--feature-points-per-group", type=int, default=1200)
    parser.add_argument("--skip-feature-geometry", action="store_true")
    parser.add_argument("--train-missing-ensemble", action="store_true")
    parser.add_argument("--ensemble-epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    if args.device is not None:
        config["device"] = args.device

    seed_everything(int(config.get("seed", 42)))
    device = get_device(config.get("device", "auto"))
    output_dir = Path(args.output_dir or config.get("paths", {}).get("paper_figures", "results/paper_figures"))
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders = build_muad_dataloaders(config)
    model = _load_model(config, device)
    num_classes = int(config["model"]["num_classes"])
    ignore_index = int(config["training"].get("ignore_index", 255))
    alpha = float(config["ood"].get("alpha", 0.6))

    ensemble, num_estimators, reason = load_or_train_deep_ensemble(
        config,
        loaders,
        device,
        train_missing=args.train_missing_ensemble,
        epochs=args.ensemble_epochs,
    )
    if ensemble is None:
        raise FileNotFoundError(
            "The paper figures require the three Deep Ensemble checkpoints. "
            f"{reason}. Re-run with --train-missing-ensemble if you want this script to create them."
        )

    detector = HybridNECOEnergyOOD(
        model=model,
        num_classes=num_classes,
        num_components=int(config["ood"]["num_components"]),
        device=device,
        temperature=float(config["ood"]["temperature"]),
    )
    detector.fit(loaders["val"], max_pixels=int(config["ood"]["max_fit_pixels"]))
    detector_path = Path(config["paths"]["detector"])
    detector_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(detector.state_dict(), detector_path)

    saved_files: list[str] = []
    ece, paths = _plot_calibration(
        model,
        loaders["test"],
        device,
        num_classes,
        ignore_index,
        int(config.get("uncertainty", {}).get("ece_num_bins", 15)),
        output_dir,
        args.max_batches,
    )
    saved_files.extend(paths)
    ensemble_ece, paths = _plot_ensemble_calibration(
        ensemble,
        loaders["test"],
        device,
        num_estimators,
        num_classes,
        ignore_index,
        int(config.get("uncertainty", {}).get("ece_num_bins", 15)),
        output_dir,
        args.max_batches,
    )
    saved_files.extend(paths)

    scores = _collect_paper_scores(loaders, detector, ensemble, num_estimators, num_classes, device, alpha, args.max_batches)
    table = [
        _binary_summary("Ensemble (predictive entropy)", *scores["Ensemble"]),
        _binary_summary("NECO-only", *scores["NECO-only"]),
        _binary_summary("Energy-only", *scores["Energy-only"]),
        _binary_summary("Hybrid (NECO + Energy)", *scores["Hybrid"]),
    ]

    qualitative_idx, paths = _plot_single_scene_qualitative(
        loaders["ood"].dataset,
        detector,
        ensemble,
        num_estimators,
        num_classes,
        device,
        output_dir,
        alpha,
    )
    saved_files.extend(paths)

    scene_indices, paths = _plot_multiscene(
        loaders["ood"].dataset,
        detector,
        ensemble,
        num_estimators,
        num_classes,
        device,
        output_dir,
        args.num_scenes,
        args.scene_seed,
        alpha,
    )
    saved_files.extend(paths)

    separation, paths = _plot_score_distributions(scores, output_dir)
    saved_files.extend(paths)

    feature_counts: dict[str, int] = {}
    feature_metrics: dict[str, float] = {}
    if not args.skip_feature_geometry:
        paths, feature_counts, feature_metrics = _plot_feature_geometry(
            loaders,
            detector,
            num_classes,
            device,
            output_dir,
            per_group=int(args.feature_points_per_group),
            max_batches=args.max_batches,
        )
        saved_files.extend(paths)

    roc_metrics, paths = _plot_roc_tail(scores, output_dir)
    saved_files.extend(paths)

    condition_summary, paths = _plot_condition_bars(
        loaders,
        detector,
        ensemble,
        num_estimators,
        num_classes,
        device,
        alpha,
        output_dir,
        args.max_batches,
        args.max_condition_scenes,
    )
    saved_files.extend(paths)

    payload: dict[str, Any] = {
        "ece": ece,
        "ensemble_ece": ensemble_ece,
        "table_i": table,
        "separation": separation,
        "qualitative_sample_index": qualitative_idx,
        "qualitative_scene_indices": scene_indices,
        "roc_operating_points": roc_metrics,
        "condition_wise": condition_summary,
        "feature_geometry_counts": feature_counts,
        "feature_geometry_metrics": feature_metrics,
        "saved_files": saved_files,
    }
    summary_path = output_dir / "paper_results.json"
    payload = _json_safe(payload)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))
    detector.cleanup()


if __name__ == "__main__":
    main()
