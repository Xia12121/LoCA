"""Reproducibility helpers: global seeding and deterministic torch config."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python / numpy / torch. Call once at the start of every run."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Keep CPU/GPU math reproducible; harmless on CPU-only boxes.
        torch.use_deterministic_algorithms(False)  # some HF ops lack det kernels
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_generator(seed: int) -> torch.Generator:
    """A standalone generator for reproducible frozen random matrices (A, F)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
