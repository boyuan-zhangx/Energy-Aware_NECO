"""Shared utilities.

Keep this package initializer lightweight so config-only tools do not import
the deep learning stack.
"""

from .config import deep_update, load_config

__all__ = ["deep_update", "load_config"]
