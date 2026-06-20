"""Full-parameter SFT baseline (§6). Upper bound, small models only."""
from __future__ import annotations

from .sft_common import SFTConfig, train_sft


def train_full_sft(model, batches, full_cfg: dict, device="cpu"):
    for p in model.parameters():
        p.requires_grad_(True)
    sft = SFTConfig(lr=full_cfg.get("lr", 1e-5), epochs=full_cfg.get("epochs", 3),
                    grad_accum=full_cfg.get("grad_accum", 1))
    return train_sft(model.to(device), batches, sft, device=device)
