"""Config access to main_egemaps_baseline_deviation_cbm.py"""

from __future__ import annotations

import os
import sys
from typing import Any

NC_ROOT = os.path.dirname(os.path.abspath(__file__))
if NC_ROOT not in sys.path:
    sys.path.insert(0, NC_ROOT)

import main_egemaps_baseline_deviation_cbm as nc


def fresh_config(dataset: str, **overrides: Any) -> "nc.Config":
    cfg = nc.Config()
    cfg.DATASET = nc.normalize_dataset_name(dataset)
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"Config has no field {key!r}")
        setattr(cfg, key, value)
    return cfg
