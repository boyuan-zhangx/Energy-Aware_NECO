"""Training and evaluation loops."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json

import torch

from .metrics import MeanIoUAccumulator, prepare_segmentation_labels


NOTEBOOK_HISTORY_KEYS = ("train_losses", "train_IoU", "val_losses", "val_IoU")
HISTORY_KEY_ALIASES = {
    "train_loss": "train_losses",
    "train_miou": "train_IoU",
    "val_loss": "val_losses",
    "val_miou": "val_IoU",
}


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Any,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: str | torch.device,
    num_classes: int,
    ignore_index: int = 255,
    log_interval: int = 0,
) -> dict[str, Any]:
    """Train one epoch and return loss plus IoU metrics."""

    model.train()
    device = torch.device(device)
    meter = MeanIoUAccumulator(num_classes=num_classes, ignore_index=ignore_index, device=device)
    total_loss = 0.0

    for step, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = prepare_segmentation_labels(labels.to(device), num_classes, ignore_index)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        meter.update(logits.detach(), labels.detach())

        if log_interval and step % log_interval == 0:
            print(f"step={step} loss={loss.item():.4f}")

    per_class_iou = meter.per_class_iou()
    return {
        "loss": total_loss / max(len(data_loader), 1),
        "miou": float(meter.compute().detach().cpu()),
        "iou_per_class": per_class_iou.detach().cpu(),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: Any,
    criterion: torch.nn.Module,
    device: str | torch.device,
    num_classes: int,
    ignore_index: int = 255,
) -> dict[str, Any]:
    """Evaluate one split and return loss plus IoU metrics."""

    model.eval()
    device = torch.device(device)
    meter = MeanIoUAccumulator(num_classes=num_classes, ignore_index=ignore_index, device=device)
    total_loss = 0.0

    for images, labels in data_loader:
        images = images.to(device)
        labels = prepare_segmentation_labels(labels.to(device), num_classes, ignore_index)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += float(loss.item())
        meter.update(logits, labels)

    per_class_iou = meter.per_class_iou()
    return {
        "loss": total_loss / max(len(data_loader), 1),
        "miou": float(meter.compute().detach().cpu()),
        "iou_per_class": per_class_iou.detach().cpu(),
    }


def save_history(history: dict[str, list[float]], path: str | Path) -> None:
    """Save training history as JSON."""

    history = normalize_history(history)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


def load_history(path: str | Path) -> dict[str, list[float]]:
    """Load training history from JSON."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return normalize_history(json.load(handle))


def normalize_history(history: dict[str, list[float]]) -> dict[str, list[float]]:
    """Normalize translated-project history keys to notebook key names."""

    normalized = dict(history)
    for old_key, notebook_key in HISTORY_KEY_ALIASES.items():
        if notebook_key not in normalized and old_key in normalized:
            normalized[notebook_key] = normalized[old_key]
    return {key: list(normalized.get(key, [])) for key in NOTEBOOK_HISTORY_KEYS}


def plot_training_curves(history: dict[str, list[float]], path: str | Path = "training_curves_unet.png") -> None:
    """Save the notebook training-curve figure with matching plot settings."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = normalize_history(history)
    colors = [[31, 120, 180], [51, 160, 44]]
    colors = [(r / 255, g / 255, b / 255) for (r, g, b) in colors]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_losses"], label="Train Loss", color=colors[0])
    plt.plot(history["val_losses"], label="Val Loss", color=colors[1])
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_IoU"], label="Train IoU", color=colors[0])
    plt.plot(history["val_IoU"], label="Val IoU", color=colors[1])
    plt.title("mIoU over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Mean IoU")
    plt.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
