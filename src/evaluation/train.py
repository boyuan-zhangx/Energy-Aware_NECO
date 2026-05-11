"""Command-line training entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.models import UNet
from src.utils.config import load_config
from src.utils.data import build_muad_dataloaders, compute_training_class_weights
from src.utils.device import get_device, seed_everything
from .trainer import evaluate, load_history, plot_training_curves, save_history, train_one_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MUAD segmentation model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-if-checkpoint-exists", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    device = get_device(config.get("device", "auto"))

    checkpoint_path = Path(config["paths"]["checkpoint"])
    if args.skip_if_checkpoint_exists and checkpoint_path.exists():
        print(f"Checkpoint already exists: {checkpoint_path}")
        history_path = Path(config["paths"]["history"])
        if history_path.exists():
            plot_training_curves(load_history(history_path), config["paths"].get("training_curves", "training_curves_unet.png"))
        return

    loaders = build_muad_dataloaders(config)
    model_cfg = config["model"]
    train_cfg = config["training"]
    num_classes = int(model_cfg["num_classes"])

    model = UNet(
        classes=num_classes,
        in_channels=int(model_cfg.get("in_channels", 3)),
        base_channels=int(model_cfg.get("base_channels", 32)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)

    weights = compute_training_class_weights(
        loaders["train"],
        train_num_classes=num_classes,
        weight_num_classes=int(train_cfg.get("class_weight_num_classes", num_classes)),
        ignore_index=int(train_cfg.get("ignore_index", 255)),
        c=float(train_cfg.get("enet_c", 1.02)),
        max_batches=train_cfg.get("class_weight_batches"),
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=int(train_cfg.get("ignore_index", 255)))
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

    epochs = args.epochs if args.epochs is not None else int(train_cfg["epochs"])
    history: dict[str, list[float]] = {
        "train_losses": [],
        "train_IoU": [],
        "val_losses": [],
        "val_IoU": [],
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
            num_classes,
            ignore_index=int(train_cfg.get("ignore_index", 255)),
            log_interval=int(train_cfg.get("log_interval", 0)),
        )
        val_metrics = evaluate(
            model,
            loaders["val"],
            criterion,
            device,
            num_classes,
            ignore_index=int(train_cfg.get("ignore_index", 255)),
        )
        scheduler.step()

        history["train_losses"].append(train_metrics["loss"])
        history["train_IoU"].append(train_metrics["miou"])
        history["val_losses"].append(val_metrics["loss"])
        history["val_IoU"].append(val_metrics["miou"])

        print(
            f"epoch={epoch + 1}/{epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_miou={train_metrics['miou']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_miou={val_metrics['miou']:.4f}"
        )

    torch.save(model.state_dict(), checkpoint_path)
    save_history(history, config["paths"]["history"])
    plot_training_curves(history, config["paths"].get("training_curves", "training_curves_unet.png"))


if __name__ == "__main__":
    main()
