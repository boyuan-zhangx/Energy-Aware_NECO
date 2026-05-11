"""MUAD dataset and dataloader utilities."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2


def build_transforms(
    image_size: list[int] | tuple[int, int] = (256, 512),
    mean: list[float] | tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: list[float] | tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> tuple[v2.Compose, v2.Compose]:
    """Build train and evaluation transforms."""

    dtype_transform = v2.ToDtype(
        dtype={
            tv_tensors.Image: torch.float32,
            tv_tensors.Mask: torch.int64,
            "others": None,
        },
        scale=True,
    )
    train_transform = v2.Compose(
        [
            v2.Resize(size=tuple(image_size), antialias=True),
            v2.RandomHorizontalFlip(),
            dtype_transform,
            v2.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = v2.Compose(
        [
            v2.Resize(size=tuple(image_size), antialias=True),
            dtype_transform,
            v2.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def build_muad_datasets(config: dict[str, Any]) -> dict[str, Any]:
    """Create MUAD train, val, test, and OOD datasets."""

    from torch_uncertainty.datasets import MUAD

    data_cfg = config["data"]
    train_transform, eval_transform = build_transforms(
        image_size=data_cfg.get("image_size", (256, 512)),
        mean=data_cfg.get("mean", (0.485, 0.456, 0.406)),
        std=data_cfg.get("std", (0.229, 0.224, 0.225)),
    )
    common = {
        "root": data_cfg["root"],
        "target_type": data_cfg.get("target_type", "semantic"),
        "version": data_cfg.get("version", "small"),
        "download": bool(data_cfg.get("download", True)),
    }
    return {
        "train": MUAD(split="train", transforms=train_transform, **common),
        "val": MUAD(split="val", transforms=eval_transform, **common),
        "test": MUAD(split="test", transforms=eval_transform, **common),
        "ood": MUAD(split="ood", transforms=eval_transform, **common),
    }


def build_dataloader(
    dataset: Any,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool = False,
) -> DataLoader:
    """Create one DataLoader."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def build_muad_dataloaders(config: dict[str, Any]) -> dict[str, DataLoader]:
    """Create MUAD dataloaders from config."""

    datasets = build_muad_datasets(config)
    data_cfg = config["data"]
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", False))
    ood_pin_memory = bool(data_cfg.get("ood_pin_memory", pin_memory))
    return {
        "train": build_dataloader(datasets["train"], batch_size, True, num_workers, pin_memory),
        "val": build_dataloader(datasets["val"], batch_size, False, num_workers, pin_memory),
        "test": build_dataloader(datasets["test"], batch_size, False, num_workers, pin_memory),
        "ood": build_dataloader(datasets["ood"], batch_size, False, num_workers, ood_pin_memory),
    }


@torch.no_grad()
def compute_enet_class_weights(
    dataloader: DataLoader,
    num_classes: int,
    ignore_index: int = 255,
    c: float = 1.02,
    max_batches: int | None = None,
) -> torch.Tensor:
    """Compute ENet class weights from segmentation labels."""

    class_count = torch.zeros(num_classes, dtype=torch.float64)
    total = 0

    for batch_idx, (_, labels) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        labels = labels.cpu().long().flatten()
        labels = labels[(labels != ignore_index) & (labels < num_classes)]
        if labels.numel() == 0:
            continue
        class_count += torch.bincount(labels, minlength=num_classes).to(torch.float64)
        total += int(labels.numel())

    if total == 0:
        raise RuntimeError("No valid labels found for class-weight computation.")
    propensity = class_count / total
    return (1.0 / torch.log(c + propensity)).float()


def compute_training_class_weights(
    dataloader: DataLoader,
    train_num_classes: int,
    weight_num_classes: int | None = None,
    ignore_index: int = 255,
    c: float = 1.02,
    max_batches: int | None = None,
) -> torch.Tensor:
    """Compute ENet weights and slice them to the trained class set."""

    weight_num_classes = weight_num_classes or train_num_classes
    weights = compute_enet_class_weights(
        dataloader,
        num_classes=weight_num_classes,
        ignore_index=ignore_index,
        c=c,
        max_batches=max_batches,
    )
    if weights.numel() < train_num_classes:
        raise ValueError(
            f"weight_num_classes={weight_num_classes} is smaller than train_num_classes={train_num_classes}"
        )
    return weights[:train_num_classes]
