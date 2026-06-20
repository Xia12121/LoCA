"""MeZO — Memory-efficient Zeroth-Order SGD (Malladi et al. 2023).

The direct forward-only competitor to LoCA (§6). For fairness we perturb the SAME
ResidualLoRA B matrices that LoCA solves in closed form, with identical dtype /
threads / data. MeZO estimates the gradient with n-SPSA: two forward passes per
step using the seed trick to regenerate the perturbation without storing it.

Why it is the foil: ZO gradient variance scales with the number of perturbed
parameters, so it converges 10-20x slower than a deterministic local solve.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..adapters.residual_lora import attach_adapters
from ..utils.seed import set_seed


@dataclass
class MeZOConfig:
    eps: float = 1e-3
    lr: float = 1e-6
    n_perturb: int = 1
    steps: int = 20000
    log_every: int = 500


@torch.no_grad()
def _loss_on_batch(model, batch, device) -> float:
    ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    out = model(input_ids=ids, attention_mask=attn, use_cache=False)
    logits = out.logits[:, :-1, :]
    tgt = labels[:, 1:]
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1),
        ignore_index=-100, reduction="mean")
    return float(ce)


@torch.no_grad()
def _perturb(params, scale: float, seed: int) -> None:
    g = torch.Generator(device=params[0].device)
    g.manual_seed(seed)
    for p in params:
        z = torch.randn(p.shape, generator=g, device=p.device, dtype=p.dtype)
        p.add_(z, alpha=scale)


def train_mezo(model, tokenizer, batches, cfg: MeZOConfig, seed: int, device="cpu"):
    """Optimize ResidualLoRA B matrices with MeZO. Returns (model, history)."""
    set_seed(seed)
    blocks = attach_adapters(model, r=32, seed=seed)
    params = [b.adapter.B for b in blocks]
    for p in params:
        p.requires_grad_(False)

    history = []
    rng = torch.Generator().manual_seed(seed)
    nb = len(batches)
    for step in range(cfg.steps):
        batch = batches[int(torch.randint(0, nb, (1,), generator=rng))]
        # average over n_perturb SPSA samples
        for _ in range(cfg.n_perturb):
            z_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=rng))
            _perturb(params, +cfg.eps, z_seed)
            loss_pos = _loss_on_batch(model, batch, device)
            _perturb(params, -2 * cfg.eps, z_seed)
            loss_neg = _loss_on_batch(model, batch, device)
            _perturb(params, +cfg.eps, z_seed)             # restore
            grad_scalar = (loss_pos - loss_neg) / (2 * cfg.eps)
            # theta -= lr * grad_scalar * z  (regenerate z from the same seed)
            _perturb(params, -cfg.lr * grad_scalar / cfg.n_perturb, z_seed)

        if step % cfg.log_every == 0:
            history.append({"step": step, "loss": loss_pos})
    return model, blocks, history
