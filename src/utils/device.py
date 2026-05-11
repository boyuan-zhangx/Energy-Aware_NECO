"""Device and reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def get_device(requested: str | None = "auto") -> torch.device:
    """Return cuda when available and requested is auto."""

    if requested in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

