"""Resolve CellScoper configuration and annotation resources."""

from __future__ import annotations

import os


_PACKAGE_DIR = os.path.dirname(__file__)
_SOURCE_ROOT = os.path.dirname(_PACKAGE_DIR)


def bundled_resource_path(name: str) -> str:
    """Return a packaged reference path, falling back to a source checkout."""
    packaged = os.path.join(_PACKAGE_DIR, "data", name)
    if os.path.isfile(packaged):
        return packaged
    return os.path.join(_SOURCE_ROOT, "src", name)


def default_config_path() -> str:
    """Return the packaged params file, falling back to a source checkout."""
    packaged = os.path.join(_PACKAGE_DIR, "data", "params.yaml")
    if os.path.isfile(packaged):
        return packaged
    return os.path.join(_SOURCE_ROOT, "config", "params.yaml")
