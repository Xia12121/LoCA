"""LoRA baseline via PEFT + standard backprop SFT (§6). Quality upper bound.

Requires `peft` (installed on the GPU box). This is the reference every method's
`recovery` is measured against.
"""
from __future__ import annotations

import torch

from .sft_common import SFTConfig, train_sft


def build_lora_model(model, lora_cfg: dict):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=lora_cfg.get("r", 32),
        lora_alpha=lora_cfg.get("alpha", 64),
        lora_dropout=lora_cfg.get("dropout", 0.0),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, cfg)


def train_lora(model, batches, lora_cfg: dict, device="cpu"):
    peft_model = build_lora_model(model, lora_cfg).to(device)
    sft = SFTConfig(lr=lora_cfg.get("lr", 2e-4), epochs=lora_cfg.get("epochs", 3),
                    grad_accum=lora_cfg.get("grad_accum", 1))
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    return train_sft(peft_model, batches, sft, device=device, trainable_params=trainable)
