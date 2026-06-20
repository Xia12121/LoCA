"""Diagnostics for the science claims C4 (§4.8).

These use the TRUE gradient g_l = dL/dh_l (autograd) for MEASUREMENT ONLY — never
to train. They are how we show the fixed feedback F_l e aligns with g_l in the
small-change regime, and how the first-order approximation degrades as O(eta^2).
"""
from __future__ import annotations

import torch
from torch import Tensor

from ..adapters.model_utils import get_handles
from .top_error import top_layer_error


@torch.no_grad()
def _cos(a: Tensor, b: Tensor) -> float:
    a = a.flatten().double()
    b = b.flatten().double()
    na, nb = a.norm(), b.norm()
    if na == 0 or nb == 0:
        return 0.0
    return float((a @ b) / (na * nb))


def true_block_grads(model, blocks, batch: dict) -> tuple[list[Tensor], Tensor]:
    """Return (g_l for each block at label positions, e at label positions).

    g_l = dL/dh_l from a single autograd backward on the frozen base model. For
    DIAGNOSTICS only. h_l here is the block output (pre final-norm except last).
    """
    h = get_handles(model)
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    predict_mask = batch["predict_mask"].to(device)
    targets = batch["targets"].to(device)

    was_training = model.training
    model.eval()
    with torch.enable_grad():
        # All params are frozen, so we inject a differentiable leaf via inputs_embeds
        # to obtain dL/dh_l for measurement only (never used to train).
        embeds = h.embed_tokens(input_ids).detach().requires_grad_(True)
        out = model(inputs_embeds=embeds, attention_mask=attn,
                    output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        for t in hs:
            if t.requires_grad:
                t.retain_grad()
        # standard shifted causal-LM loss (hidden at t predicts token t+1)
        loss = torch.nn.functional.cross_entropy(
            out.logits[:, :-1, :].reshape(-1, out.logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100, reduction="sum",
        )
        loss.backward()

    # g_l and e live at prediction positions (t such that t+1 is a label)
    g_list = [hs[l + 1].grad[predict_mask].detach().to(torch.float32) for l in range(len(blocks))]
    h_L = hs[-1].detach()[predict_mask].to(torch.float32)
    e = top_layer_error(h_L, targets[predict_mask], h.lm_head, reduction="sum")
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return g_list, e


def alignment_angles(model, blocks, feedback: list[Tensor], batch: dict) -> list[float]:
    """cos(F_l e, g_l) per block (C4(a) headline curve). Expected > 0 and rising."""
    g_list, e = true_block_grads(model, blocks, batch)
    cosines = []
    for l, g in enumerate(g_list):
        Fe = e @ feedback[l].to(device=e.device, dtype=e.dtype).t()
        cosines.append(_cos(Fe, g))
    return cosines


def linearization_gap(model, blocks, feedback: list[Tensor], batch: dict, eta: float) -> dict:
    """Compare predicted first-order dL = -eta * sum<g_l, F_l e> against truth.

    Returns the predicted dL and the per-block alignment dot products. The actual
    dL must be measured by the caller by re-evaluating loss after applying the
    -eta F_l e move (kept separate so this stays autograd-light).
    """
    g_list, e = true_block_grads(model, blocks, batch)
    dot = 0.0
    per_block = []
    for l, g in enumerate(g_list):
        Fe = e @ feedback[l].to(device=e.device, dtype=e.dtype).t()
        d = float((g.double().flatten() @ Fe.double().flatten()))
        per_block.append(d)
        dot += d
    return {"pred_dL": -eta * dot, "per_block_dot": per_block, "sum_dot": dot}
