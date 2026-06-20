"""YAML config loading with `_base_` inheritance and deep merge."""
from __future__ import annotations

import copy
import os
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML config; if it has a top-level `_base_: <relpath>`, merge onto it."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base_ref = cfg.pop("_base_", None)
    if base_ref:
        base_path = os.path.join(os.path.dirname(os.path.abspath(path)), base_ref)
        base = load_config(base_path)
        cfg = _deep_merge(base, cfg)
    return cfg


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply CLI dotted overrides like `loca.eta=0.1 train.n_train=2000`."""
    cfg = copy.deepcopy(cfg)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got: {ov}")
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)  # parse int/float/bool/str
    return cfg
