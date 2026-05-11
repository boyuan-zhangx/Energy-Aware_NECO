"""Configuration loading helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive merge of base and override."""

    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Read a dotted key from a nested config dictionary."""

    value: Any = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value

