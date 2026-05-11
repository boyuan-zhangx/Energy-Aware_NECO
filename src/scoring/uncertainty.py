"""Uncertainty score helpers for segmentation logits."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange


def temperature_scale(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply scalar temperature scaling to logits."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return logits / temperature


def max_class_probability(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Return the maximum softmax probability map."""

    probs = F.softmax(temperature_scale(logits, temperature), dim=1)
    return probs.max(dim=1).values


def predictive_entropy(probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute predictive entropy for probabilities shaped [B, C, H, W]."""

    return -(probs * torch.log(probs.clamp_min(eps))).sum(dim=1)


def entropy_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute entropy from logits shaped [B, C, H, W]."""

    probs = F.softmax(temperature_scale(logits, temperature), dim=1)
    return predictive_entropy(probs)


def ensemble_mean_probs(
    outputs: torch.Tensor,
    batch_size: int,
    num_estimators: int,
) -> torch.Tensor:
    """Normalize common ensemble output layouts to mean probabilities."""

    if outputs.dim() == 4:
        if outputs.shape[0] == batch_size:
            return outputs.softmax(dim=1)
        if outputs.shape[0] == batch_size * num_estimators:
            outputs = rearrange(outputs, "(m b) c h w -> m b c h w", m=num_estimators, b=batch_size)
            return outputs.softmax(dim=2).mean(dim=0)
        raise ValueError(f"Unexpected 4D ensemble output shape: {tuple(outputs.shape)}")

    if outputs.dim() == 5:
        if outputs.shape[0] == num_estimators:
            return outputs.softmax(dim=2).mean(dim=0)
        if outputs.shape[1] == num_estimators:
            return outputs.softmax(dim=2).mean(dim=1)
        return outputs.softmax(dim=2).mean(dim=0)

    raise ValueError(f"Unexpected ensemble output dims: {outputs.dim()}")


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = 255,
    init_temperature: float = 1.5,
    max_iter: int = 50,
) -> float:
    """Optimize one scalar temperature on validation logits."""

    if logits.dim() == 2:
        flat_logits = logits
    elif logits.dim() == 4:
        flat_logits = rearrange(logits, "b c h w -> (b h w) c")
    else:
        raise ValueError(f"Expected 2D or 4D logits, got shape {tuple(logits.shape)}")

    flat_labels = labels.flatten()
    valid = flat_labels != ignore_index
    if not valid.any():
        raise ValueError("No valid labels available for temperature fitting.")

    temperature = torch.nn.Parameter(torch.ones(1, device=logits.device) * init_temperature)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = criterion(flat_logits[valid] / temperature.clamp_min(1e-6), flat_labels[valid])
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp_min(1e-6).item())
