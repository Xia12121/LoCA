"""Shared backprop SFT loop for the LoRA and full-SFT baselines (§6).

Plain PyTorch (no TRL). Same data/collation/dtype as LoCA for fairness; the only
difference is these use standard autograd through the whole backbone — which is
exactly the global backward chain LoCA avoids.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SFTConfig:
    lr: float = 2e-4
    epochs: int = 3
    grad_accum: int = 1
    log_every: int = 50
    max_grad_norm: float = 1.0


def _ce_loss(model, batch, device):
    ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    out = model(input_ids=ids, attention_mask=attn, use_cache=False)
    logits = out.logits[:, :-1, :]
    tgt = labels[:, 1:]
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1),
        ignore_index=-100, reduction="mean")


def train_sft(model, batches, cfg: SFTConfig, device="cpu", trainable_params=None):
    """Standard SFT. `trainable_params` defaults to all params requiring grad."""
    model.train()
    params = trainable_params if trainable_params is not None else [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    history = []
    step = 0
    for epoch in range(cfg.epochs):
        for i, batch in enumerate(batches):
            loss = _ce_loss(model, batch, device) / cfg.grad_accum
            loss.backward()
            if (i + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if step % cfg.log_every == 0:
                history.append({"epoch": epoch, "step": step, "loss": loss.detach().item() * cfg.grad_accum})
            step += 1
    model.eval()
    return model, history
