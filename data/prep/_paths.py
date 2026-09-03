"""Shared path helpers. DATA_BASE overrides the data root, which defaults to
<repo>/data.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["DATA_ROOT", "out_path"]

#: <repo>/data, or $DATA_BASE when set.
DATA_ROOT = Path(os.environ.get("DATA_BASE") or Path(__file__).resolve().parents[1])


def out_path(*parts: str) -> Path:
    """``out_path("code", "TACO", "train.parquet")`` -> ``<repo>/data/code/TACO/train.parquet``."""
    return DATA_ROOT.joinpath(*parts)
