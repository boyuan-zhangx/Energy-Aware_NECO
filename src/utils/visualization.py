"""Visualization helpers for normalized tensors and score maps."""

from __future__ import annotations

import torch
from torchvision.transforms.v2 import functional as F


def denormalize_image(
    image: torch.Tensor,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Convert a normalized CHW image tensor back to [0, 1]."""

    device = image.device
    mean_t = torch.tensor(mean, device=device).view(3, 1, 1)
    std_t = torch.tensor(std, device=device).view(3, 1, 1)
    return (image * std_t + mean_t).clamp(0, 1)


def normalize_map(score_map: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Min-max normalize a score map for display."""

    score_map = score_map.detach().float().cpu()
    return (score_map - score_map.min()) / (score_map.max() - score_map.min() + eps)


def tensor_to_pil(image: torch.Tensor):
    """Convert a CHW image tensor to a PIL image."""

    return F.to_pil_image(image.detach().cpu())

