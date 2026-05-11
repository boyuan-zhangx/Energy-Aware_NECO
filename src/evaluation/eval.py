"""Command-line model evaluation entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.models import UNet
from src.utils.config import load_config
from src.utils.data import build_muad_dataloaders, compute_training_class_weights
from src.utils.device import get_device
from .trainer import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation model.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config.get("device", "auto"))
    loaders = build_muad_dataloaders(config)
    num_classes = int(config["model"]["num_classes"])

    model = UNet(
        classes=num_classes,
        in_channels=int(config["model"].get("in_channels", 3)),
        base_channels=int(config["model"].get("base_channels", 32)),
        dropout=float(config["model"].get("dropout", 0.1)),
    ).to(device)
    checkpoint = Path(config["paths"]["checkpoint"])
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    train_cfg = config["training"]
    weights = compute_training_class_weights(
        loaders["train"],
        train_num_classes=num_classes,
        weight_num_classes=int(train_cfg.get("class_weight_num_classes", num_classes)),
        ignore_index=int(train_cfg.get("ignore_index", 255)),
        c=float(train_cfg.get("enet_c", 1.02)),
        max_batches=train_cfg.get("class_weight_batches"),
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=int(train_cfg.get("ignore_index", 255)))
    metrics = evaluate(model, loaders["test"], criterion, device, num_classes)
    print(f"test_loss={metrics['loss']:.4f} test_miou={metrics['miou']:.4f}")
    print("per_class_iou=", metrics["iou_per_class"].tolist())


if __name__ == "__main__":
    main()
