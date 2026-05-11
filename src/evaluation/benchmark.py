"""OOD and uncertainty benchmark entrypoint."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from einops import rearrange
from sklearn.metrics import roc_curve
from torch_uncertainty.metrics import CalibrationError

from src.models import UNet
from src.scoring import (
    HybridNECOEnergyOOD,
    ensemble_mean_probs,
    fit_temperature,
    max_class_probability,
    predictive_entropy,
)
from src.utils.config import load_config
from src.utils.data import build_muad_dataloaders, compute_training_class_weights
from src.utils.device import get_device, seed_everything
from .metrics import binary_auroc_score, fpr_at_tpr, prepare_segmentation_labels
from .trainer import evaluate, train_one_epoch


ScoreFn = Callable[[torch.Tensor], torch.Tensor]


def _as_int_or_none(value: Any) -> int | None:
    return None if value is None else int(value)


def _squeeze_labels(labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim == 4 and labels.shape[1] == 1:
        labels = labels.squeeze(1)
    return labels


def _select_pixel_scores(
    score_map: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    id_split: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    flat_score = score_map.flatten()
    flat_labels = _squeeze_labels(labels).flatten()
    valid = flat_labels != 255
    fallback = flat_score[valid].detach().cpu() if valid.any() else None

    if id_split:
        keep = valid & (flat_labels < num_classes)
    else:
        keep = valid & (flat_labels >= num_classes)

    if keep.any():
        return flat_score[keep].detach().cpu(), fallback
    return None, fallback


def fpr95_discrete(y_true: np.ndarray, y_score: np.ndarray, target_tpr: float = 0.95) -> float:
    """Classic FPR95: first FPR where TPR reaches the requested target."""

    fpr, tpr, _ = roc_curve(y_true, y_score)
    indices = np.where(tpr >= target_tpr)[0]
    if len(indices) == 0:
        return math.nan
    return float(fpr[indices[0]])


def robust_fpr_summary(
    id_scores: torch.Tensor,
    ood_scores: torch.Tensor,
    max_points: int = 200_000,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> dict[str, float | int]:
    """Repeated-subsampling FPR summary for the pixel OOD protocol."""

    if id_scores.numel() == 0 or ood_scores.numel() == 0:
        return {
            "n_total": 0,
            "n_sampled": 0,
            "fpr95_discrete_full": math.nan,
            "fpr95_interp_full": math.nan,
            "fpr95_interp_mean": math.nan,
            "fpr95_interp_std": math.nan,
            "fpr90_interp_mean": math.nan,
            "fpr90_interp_std": math.nan,
            "fpr98_interp_mean": math.nan,
            "fpr98_interp_std": math.nan,
        }

    id_arr = id_scores.detach().cpu().float().numpy().reshape(-1)
    ood_arr = ood_scores.detach().cpu().float().numpy().reshape(-1)
    y_true_full = np.concatenate([np.zeros_like(id_arr, dtype=np.int64), np.ones_like(ood_arr, dtype=np.int64)])
    y_score_full = np.concatenate([id_arr, ood_arr])

    use_full = len(y_true_full) <= 600_000
    if use_full:
        fpr95_d = fpr95_discrete(y_true_full, y_score_full, target_tpr=0.95)
        fpr95_i = fpr_at_tpr(y_true_full, y_score_full, target_tpr=0.95)
    else:
        fpr95_d = math.nan
        fpr95_i = math.nan

    n = len(y_true_full)
    k = min(max_points, n)
    vals95: list[float] = []
    vals90: list[float] = []
    vals98: list[float] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        if k < n:
            indices = rng.choice(n, size=k, replace=False)
            y_true = y_true_full[indices]
            y_score = y_score_full[indices]
        else:
            y_true = y_true_full
            y_score = y_score_full
        vals95.append(fpr_at_tpr(y_true, y_score, target_tpr=0.95))
        vals90.append(fpr_at_tpr(y_true, y_score, target_tpr=0.90))
        vals98.append(fpr_at_tpr(y_true, y_score, target_tpr=0.98))

    vals95_np = np.asarray(vals95, dtype=np.float64)
    vals90_np = np.asarray(vals90, dtype=np.float64)
    vals98_np = np.asarray(vals98, dtype=np.float64)
    return {
        "n_total": int(n),
        "n_sampled": int(k),
        "fpr95_discrete_full": fpr95_d,
        "fpr95_interp_full": fpr95_i,
        "fpr95_interp_mean": float(vals95_np.mean()),
        "fpr95_interp_std": float(vals95_np.std()),
        "fpr90_interp_mean": float(vals90_np.mean()),
        "fpr90_interp_std": float(vals90_np.std()),
        "fpr98_interp_mean": float(vals98_np.mean()),
        "fpr98_interp_std": float(vals98_np.std()),
    }


def summarize_scores(name: str, id_scores: torch.Tensor, ood_scores: torch.Tensor) -> dict[str, float | int | str]:
    """Summarize one OOD method where larger scores indicate OOD."""

    summary: dict[str, float | int | str] = {
        "method": name,
        "auroc": binary_auroc_score(id_scores, ood_scores),
        "id_count": int(id_scores.numel()),
        "ood_count": int(ood_scores.numel()),
        "id_mean": float(id_scores.mean()) if id_scores.numel() else math.nan,
        "ood_mean": float(ood_scores.mean()) if ood_scores.numel() else math.nan,
    }
    summary.update(robust_fpr_summary(id_scores, ood_scores))
    return summary


@torch.no_grad()
def collect_scores_from_logits(
    loader: Any,
    model: torch.nn.Module,
    score_fn: ScoreFn,
    num_classes: int,
    device: str | torch.device,
    id_split: bool,
    max_batches: int | None = None,
) -> torch.Tensor:
    """Collect pixel scores from a single-model logit score map."""

    device = torch.device(device)
    scores: list[torch.Tensor] = []
    fallback_scores: list[torch.Tensor] = []
    model.to(device)
    model.eval()

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)
        labels = _squeeze_labels(labels.to(device))
        score_map = score_fn(model(images))
        selected, fallback = _select_pixel_scores(score_map, labels, num_classes, id_split)
        if selected is not None:
            scores.append(selected)
        if fallback is not None:
            fallback_scores.append(fallback)

    if scores:
        return torch.cat(scores)
    if not id_split and fallback_scores:
        return torch.cat(fallback_scores)
    return torch.empty(0)


def collect_scores_from_map(
    loader: Any,
    detector: HybridNECOEnergyOOD,
    map_key: str,
    num_classes: int,
    alpha: float = 0.6,
    id_split: bool = True,
    max_batches: int | None = None,
) -> torch.Tensor:
    """Collect ID or OOD scores from one detector map."""

    scores: list[torch.Tensor] = []
    fallback_scores: list[torch.Tensor] = []

    for batch_idx, batch_data in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = batch_data[0].to(detector.device)
        labels = _squeeze_labels(batch_data[1].to(detector.device))
        score_map = detector.score_maps(images, alpha=alpha)[map_key]
        selected, fallback = _select_pixel_scores(score_map, labels, num_classes, id_split)
        if selected is not None:
            scores.append(selected)
        if fallback is not None:
            fallback_scores.append(fallback)

    if scores:
        return torch.cat(scores)
    if not id_split and fallback_scores:
        return torch.cat(fallback_scores)
    return torch.empty(0)


@torch.no_grad()
def collect_ensemble_entropy_scores(
    loader: Any,
    ensemble_model: torch.nn.Module,
    num_estimators: int,
    num_classes: int,
    device: str | torch.device,
    id_split: bool = True,
    max_batches: int | None = None,
) -> torch.Tensor:
    """Collect predictive entropy scores from ensemble-style outputs."""

    device = torch.device(device)
    scores: list[torch.Tensor] = []
    fallback_scores: list[torch.Tensor] = []
    ensemble_model.to(device)
    ensemble_model.eval()

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)
        labels = _squeeze_labels(labels.to(device))

        outputs = ensemble_model(images)
        probs = ensemble_mean_probs(outputs, batch_size=images.shape[0], num_estimators=num_estimators)
        entropy_map = predictive_entropy(probs)
        selected, fallback = _select_pixel_scores(entropy_map, labels, num_classes, id_split)
        if selected is not None:
            scores.append(selected)
        if fallback is not None:
            fallback_scores.append(fallback)

    if scores:
        return torch.cat(scores)
    if not id_split and fallback_scores:
        return torch.cat(fallback_scores)
    return torch.empty(0)


def compare_hybrid_scores(
    id_loader: Any,
    ood_loader: Any,
    detector: HybridNECOEnergyOOD,
    num_classes: int,
    alpha: float = 0.6,
    max_batches: int | None = None,
) -> list[dict[str, float | int | str]]:
    """Compare NECO-only, energy-only, and hybrid OOD scores."""

    id_hybrid = collect_scores_from_map(id_loader, detector, "ood_score", num_classes, alpha, True, max_batches)
    ood_hybrid = collect_scores_from_map(ood_loader, detector, "ood_score", num_classes, alpha, False, max_batches)
    id_neco = collect_scores_from_map(id_loader, detector, "neco", num_classes, alpha, True, max_batches)
    ood_neco = collect_scores_from_map(ood_loader, detector, "neco", num_classes, alpha, False, max_batches)
    id_energy = collect_scores_from_map(id_loader, detector, "energy", num_classes, alpha, True, max_batches)
    ood_energy = collect_scores_from_map(ood_loader, detector, "energy", num_classes, alpha, False, max_batches)

    stats = detector.stats
    if stats is None:
        raise RuntimeError("Detector must be fitted before comparison.")

    id_neco = -((id_neco - stats.neco_mean.cpu()) / (stats.neco_std.cpu() + 1e-8))
    ood_neco = -((ood_neco - stats.neco_mean.cpu()) / (stats.neco_std.cpu() + 1e-8))
    id_energy = -((id_energy - stats.energy_mean.cpu()) / (stats.energy_std.cpu() + 1e-8))
    ood_energy = -((ood_energy - stats.energy_mean.cpu()) / (stats.energy_std.cpu() + 1e-8))

    return [
        summarize_scores("NECO-only", id_neco, ood_neco),
        summarize_scores("Energy-only", id_energy, ood_energy),
        summarize_scores("Hybrid NECO+Energy", id_hybrid, ood_hybrid),
    ]


def _build_unet(config: dict[str, Any]) -> UNet:
    model_cfg = config["model"]
    return UNet(
        classes=int(model_cfg["num_classes"]),
        in_channels=int(model_cfg.get("in_channels", 3)),
        base_channels=int(model_cfg.get("base_channels", 32)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )


def _load_model(config: dict[str, Any], device: torch.device) -> UNet:
    model = _build_unet(config)
    checkpoint = Path(config["paths"]["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device)


def _make_training_criterion(config: dict[str, Any], loaders: dict[str, Any], device: torch.device) -> torch.nn.Module:
    train_cfg = config["training"]
    weights = compute_training_class_weights(
        loaders["train"],
        train_num_classes=int(config["model"]["num_classes"]),
        weight_num_classes=int(train_cfg.get("class_weight_num_classes", config["model"]["num_classes"])),
        ignore_index=int(train_cfg.get("ignore_index", 255)),
        c=float(train_cfg.get("enet_c", 1.02)),
        max_batches=train_cfg.get("class_weight_batches"),
    ).to(device)
    return torch.nn.CrossEntropyLoss(weight=weights, ignore_index=int(train_cfg.get("ignore_index", 255)))


def _train_unet_checkpoint(
    model: UNet,
    checkpoint_path: Path,
    config: dict[str, Any],
    loaders: dict[str, Any],
    criterion: torch.nn.Module,
    device: torch.device,
    epochs: int,
) -> None:
    train_cfg = config["training"]
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        momentum=float(train_cfg.get("momentum", 0.9)),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(train_cfg["lr_decay_epochs"]),
        gamma=float(train_cfg["lr_decay"]),
    )
    num_classes = int(config["model"]["num_classes"])
    ignore_index = int(train_cfg.get("ignore_index", 255))
    model.to(device)

    for epoch in range(epochs):
        train_metrics = train_one_epoch(model, loaders["train"], optimizer, criterion, device, num_classes, ignore_index)
        val_metrics = evaluate(model, loaders["val"], criterion, device, num_classes, ignore_index)
        scheduler.step()
        print(
            f"ensemble_epoch={epoch + 1}/{epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_miou={train_metrics['miou']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_miou={val_metrics['miou']:.4f}"
        )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)


def load_or_train_deep_ensemble(
    config: dict[str, Any],
    loaders: dict[str, Any],
    device: torch.device,
    train_missing: bool,
    epochs: int | None,
) -> tuple[torch.nn.Module | None, int, str | None]:
    """Load or train the three-model Deep Ensemble."""

    from torch_uncertainty.models import deep_ensembles

    ens_cfg = config.get("uncertainty", {}).get("ensemble", {})
    num_estimators = int(ens_cfg.get("num_estimators", 3))
    template = ens_cfg.get("checkpoint_template", "checkpoints/unet_ens_{index}.pth")
    paths = [Path(str(template).format(index=index)) for index in range(num_estimators)]
    missing = [path for path in paths if not path.exists()]

    if missing and not train_missing:
        return None, num_estimators, f"missing checkpoints: {', '.join(str(path) for path in missing)}"

    criterion: torch.nn.Module | None = None
    if missing:
        criterion = _make_training_criterion(config, loaders, device)
        train_epochs = epochs if epochs is not None else int(config["training"]["epochs"])
        for index, path in enumerate(paths):
            if path.exists():
                continue
            print(f"Training ensemble model {index + 1}/{num_estimators}")
            model = _build_unet(config)
            _train_unet_checkpoint(model, path, config, loaders, criterion, device, train_epochs)

    models: list[UNet] = []
    for path in paths:
        model = _build_unet(config)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)

    return deep_ensembles(models), num_estimators, None


@torch.no_grad()
def collect_validation_logits(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    max_batches: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect flattened validation logits and labels for temperature scaling."""

    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    model.eval()

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)
        labels = prepare_segmentation_labels(labels.to(device), num_classes, ignore_index)
        logits = model(images)
        flat_logits = rearrange(logits, "b c h w -> (b h w) c")
        flat_labels = labels.flatten()
        valid = flat_labels != ignore_index
        if valid.any():
            logits_list.append(flat_logits[valid].detach())
            labels_list.append(flat_labels[valid].detach())

    if not logits_list:
        raise RuntimeError("No valid validation pixels found for temperature scaling.")
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


@torch.no_grad()
def compute_ece(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    num_bins: int,
    temperature: float = 1.0,
    max_batches: int | None = None,
) -> float:
    """Compute multiclass ECE on flattened segmentation pixels."""

    metric = CalibrationError(
        task="multiclass",
        num_classes=num_classes,
        num_bins=num_bins,
        norm="l1",
        ignore_index=ignore_index,
    ).to(device)
    model.eval()

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)
        labels = prepare_segmentation_labels(labels.to(device), num_classes, ignore_index)
        logits = model(images) / temperature
        probs = logits.softmax(dim=1)
        flat_probs = rearrange(probs, "b c h w -> (b h w) c")
        flat_labels = labels.flatten()
        valid = flat_labels != ignore_index
        if valid.any():
            metric.update(flat_probs[valid], flat_labels[valid])

    return float(metric.compute().detach().cpu())


def measure_runtime_ms(
    func: Callable[[torch.Tensor], torch.Tensor],
    sample: torch.Tensor,
    device: torch.device,
    runs: int = 30,
    warmup: int = 8,
) -> float:
    """Time the complete inference and score-map path for one input."""

    sample = sample.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = func(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            _ = func(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / runs


def _enabled_methods(methods: list[str]) -> set[str]:
    if "paper" in methods:
        return {"ensemble", "hybrid"}
    if "all" in methods:
        return {"mcp", "temperature", "mc_dropout", "ensemble", "hybrid"}
    return set(methods)


def _first_sample(loader: Any) -> torch.Tensor:
    images, _ = next(iter(loader))
    return images[:1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MUAD uncertainty and OOD benchmarks.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["paper"],
        choices=["paper", "all", "mcp", "temperature", "mc_dropout", "ensemble", "hybrid"],
    )
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--temperature-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-missing", action="store_true")
    parser.add_argument("--ensemble-epochs", type=int, default=None)
    parser.add_argument("--skip-runtime", action="store_true")
    parser.add_argument("--runtime-runs", type=int, default=30)
    parser.add_argument("--runtime-warmup", type=int, default=8)
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
    loaders = build_muad_dataloaders(config)
    model = _load_model(config, device)

    num_classes = int(config["model"]["num_classes"])
    ignore_index = int(config["training"].get("ignore_index", 255))
    max_batches = args.max_batches if args.max_batches is not None else _as_int_or_none(config["ood"].get("max_eval_batches"))
    methods = _enabled_methods(args.methods)
    results: list[dict[str, float | int | str]] = []
    skipped: dict[str, str] = {}
    runtime: dict[str, dict[str, float | int]] = {}
    sample = _first_sample(loaders["test"]).to(device)

    if "mcp" in methods:
        id_scores = collect_scores_from_logits(
            loaders["test"],
            model,
            lambda logits: 1.0 - max_class_probability(logits),
            num_classes,
            device,
            id_split=True,
            max_batches=max_batches,
        )
        ood_scores = collect_scores_from_logits(
            loaders["ood"],
            model,
            lambda logits: 1.0 - max_class_probability(logits),
            num_classes,
            device,
            id_split=False,
            max_batches=max_batches,
        )
        results.append(summarize_scores("MCP uncertainty", id_scores, ood_scores))
        if not args.skip_runtime:
            runtime["MCP uncertainty"] = {
                "latency_ms": measure_runtime_ms(
                    lambda x: 1.0 - max_class_probability(model(x)),
                    sample,
                    device,
                    runs=args.runtime_runs,
                    warmup=args.runtime_warmup,
                ),
                "runs": args.runtime_runs,
            }

    temperature_info: dict[str, float] = {}
    if "temperature" in methods:
        unc_cfg = config.get("uncertainty", {})
        flat_logits, flat_labels = collect_validation_logits(
            model,
            loaders["val"],
            device,
            num_classes,
            ignore_index,
            max_batches=args.temperature_batches,
        )
        temperature = fit_temperature(
            flat_logits,
            flat_labels,
            ignore_index=ignore_index,
            init_temperature=float(unc_cfg.get("temperature_init", 1.5)),
            max_iter=int(unc_cfg.get("temperature_max_iter", 50)),
        )
        ece_before = compute_ece(
            model,
            loaders["test"],
            device,
            num_classes,
            ignore_index,
            num_bins=int(unc_cfg.get("ece_num_bins", 15)),
            temperature=1.0,
            max_batches=max_batches,
        )
        ece_after = compute_ece(
            model,
            loaders["test"],
            device,
            num_classes,
            ignore_index,
            num_bins=int(unc_cfg.get("ece_num_bins", 15)),
            temperature=temperature,
            max_batches=max_batches,
        )
        temperature_info = {"temperature": temperature, "ece_before": ece_before, "ece_after": ece_after}
        id_scores = collect_scores_from_logits(
            loaders["test"],
            model,
            lambda logits: 1.0 - max_class_probability(logits, temperature=temperature),
            num_classes,
            device,
            id_split=True,
            max_batches=max_batches,
        )
        ood_scores = collect_scores_from_logits(
            loaders["ood"],
            model,
            lambda logits: 1.0 - max_class_probability(logits, temperature=temperature),
            num_classes,
            device,
            id_split=False,
            max_batches=max_batches,
        )
        summary = summarize_scores("Temperature-scaled MCP uncertainty", id_scores, ood_scores)
        summary.update(temperature_info)
        results.append(summary)
        if not args.skip_runtime:
            runtime["Temperature-scaled MCP uncertainty"] = {
                "latency_ms": measure_runtime_ms(
                    lambda x: 1.0 - max_class_probability(model(x), temperature=temperature),
                    sample,
                    device,
                    runs=args.runtime_runs,
                    warmup=args.runtime_warmup,
                ),
                "runs": args.runtime_runs,
            }

    if "mc_dropout" in methods:
        from torch_uncertainty.models.wrappers.mc_dropout import mc_dropout

        for num_estimators in config.get("uncertainty", {}).get("mc_dropout_estimators", [3, 20]):
            num_estimators = int(num_estimators)
            mc_model = mc_dropout(model, num_estimators=num_estimators)
            id_scores = collect_ensemble_entropy_scores(
                loaders["test"],
                mc_model,
                num_estimators,
                num_classes,
                device,
                id_split=True,
                max_batches=max_batches,
            )
            ood_scores = collect_ensemble_entropy_scores(
                loaders["ood"],
                mc_model,
                num_estimators,
                num_classes,
                device,
                id_split=False,
                max_batches=max_batches,
            )
            name = f"MC Dropout entropy T={num_estimators}"
            results.append(summarize_scores(name, id_scores, ood_scores))
            if not args.skip_runtime:
                runtime[name] = {
                    "latency_ms": measure_runtime_ms(
                        lambda x, m=mc_model, n=num_estimators: predictive_entropy(
                            ensemble_mean_probs(m(x), batch_size=x.shape[0], num_estimators=n)
                        ),
                        sample,
                        device,
                        runs=args.runtime_runs,
                        warmup=args.runtime_warmup,
                    ),
                    "runs": args.runtime_runs,
                }

    if "ensemble" in methods:
        train_ensemble = args.train_missing or bool(config.get("uncertainty", {}).get("ensemble", {}).get("train_missing", False))
        ensemble, num_estimators, reason = load_or_train_deep_ensemble(
            config,
            loaders,
            device,
            train_missing=train_ensemble,
            epochs=args.ensemble_epochs,
        )
        if ensemble is None:
            skipped["Deep Ensemble entropy"] = reason or "not available"
        else:
            id_scores = collect_ensemble_entropy_scores(
                loaders["test"],
                ensemble,
                num_estimators,
                num_classes,
                device,
                id_split=True,
                max_batches=max_batches,
            )
            ood_scores = collect_ensemble_entropy_scores(
                loaders["ood"],
                ensemble,
                num_estimators,
                num_classes,
                device,
                id_split=False,
                max_batches=max_batches,
            )
            results.append(summarize_scores("Deep Ensemble entropy", id_scores, ood_scores))
            if not args.skip_runtime:
                runtime["Deep Ensemble entropy"] = {
                    "latency_ms": measure_runtime_ms(
                        lambda x: predictive_entropy(
                            ensemble_mean_probs(ensemble(x), batch_size=x.shape[0], num_estimators=num_estimators)
                        ),
                        sample,
                        device,
                        runs=args.runtime_runs,
                        warmup=args.runtime_warmup,
                    ),
                    "runs": args.runtime_runs,
                }

    if "hybrid" in methods:
        ood_cfg = config["ood"]
        detector = HybridNECOEnergyOOD(
            model=model,
            num_classes=num_classes,
            num_components=int(ood_cfg["num_components"]),
            device=device,
            temperature=float(ood_cfg["temperature"]),
        )
        detector.fit(loaders["val"], max_pixels=int(ood_cfg["max_fit_pixels"]))
        detector_path = Path(config["paths"]["detector"])
        detector_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(detector.state_dict(), detector_path)
        results.extend(
            compare_hybrid_scores(
                loaders["test"],
                loaders["ood"],
                detector,
                num_classes=num_classes,
                alpha=float(ood_cfg["alpha"]),
                max_batches=max_batches,
            )
        )
        if not args.skip_runtime:
            runtime["Hybrid NECO+Energy"] = {
                "latency_ms": measure_runtime_ms(
                    lambda x: detector.score_maps(x, alpha=float(ood_cfg["alpha"]))["ood_score"],
                    sample,
                    device,
                    runs=args.runtime_runs,
                    warmup=args.runtime_warmup,
                ),
                "runs": args.runtime_runs,
            }
        detector.cleanup()

    payload: dict[str, Any] = {
        "methods": results,
        "temperature": temperature_info,
        "runtime": runtime,
        "skipped": skipped,
    }
    out_path = Path(config["paths"]["benchmark"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
