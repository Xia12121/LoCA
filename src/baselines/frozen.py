"""Frozen baseline (§6): the untrained base model — the lower bound for recovery."""
from __future__ import annotations


def train_frozen(model, *args, **kwargs):
    """No-op: frozen base is evaluated as-is."""
    model.eval()
    return model, []
