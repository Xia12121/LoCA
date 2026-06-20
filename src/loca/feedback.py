"""Fixed feedback operators F_l (§4.5) — the DFA connection.

Two variants (these form the C4 / risk-A ablation):

  random : F_l ~ N(0, 1/d), frozen. Cheapest. Pure feedback-alignment.

  sketch : a one-time, init-only low-rank approximation of the TRUE backward
           operator (d L / d h_l) seen as a fixed linear map from the top-error
           space to block l's space. We fit, on a small probe batch, the rank-k
           map F_l minimizing || F_l e - g_l || (ridge), where g_l = dL/dh_l is
           obtained by a SINGLE backbone backward on the frozen base model.
           This backward is a one-time INIT cost (explicitly allowed, §4.5); it
           is NOT part of the per-epoch training and must be reported separately.

Both return a list {F_l}_{l=1..L}, each (d, d) float32, frozen.
"""
from __future__ import annotations

import torch
from torch import Tensor

from ..utils.seed import seed_generator


def build_random_feedback(d: int, L: int, seed: int) -> list[Tensor]:
    g = seed_generator(seed)
    return [
        (torch.randn(d, d, generator=g, dtype=torch.float32) / (d ** 0.5))
        for _ in range(L)
    ]


@torch.no_grad()
def _ridge_lowrank_map(G: Tensor, E: Tensor, rank: int, beta: float = 1e-3) -> Tensor:
    """Fit F minimizing ||F E - G||^2 + beta||F||^2, then truncate to `rank`.

    E: (d_top, n) top-error columns ; G: (d, n) true block-gradient columns.
    F = G E^T (E E^T + beta I)^{-1} ; SVD-truncate to `rank`.
    """
    Ef = E.to(torch.float64)
    Gf = G.to(torch.float64)
    d_top = Ef.shape[0]
    M = Ef @ Ef.t() + beta * torch.eye(d_top, dtype=torch.float64, device=Ef.device)
    F = Gf @ Ef.t() @ torch.linalg.inv(M)            # (d, d_top)
    if rank and rank < min(F.shape):
        U, S, Vh = torch.linalg.svd(F, full_matrices=False)
        F = (U[:, :rank] * S[:rank]) @ Vh[:rank]
    return F.to(torch.float32)


def build_sketch_feedback(
    model,
    blocks,
    probe_batch: dict,
    rank_sketch: int = 8,
) -> list[Tensor]:
    """One-time sketch of the base backward operator on a probe batch.

    probe_batch: dict with input_ids, attention_mask, labels, label_mask.
    Returns {F_l} each (d, d). Requires a single grad-enabled forward+backward on
    the FROZEN base model (adapters are B=0 here, so model == base).
    """
    from ..adapters.model_utils import get_handles

    h = get_handles(model)
    L = len(blocks)
    device = next(model.parameters()).device

    input_ids = probe_batch["input_ids"].to(device)
    attn = probe_batch["attention_mask"].to(device)
    labels = probe_batch["labels"].to(device)
    predict_mask = probe_batch["predict_mask"].to(device)
    targets = probe_batch["targets"].to(device)

    was_training = model.training
    model.eval()
    # Gradient checkpointing bounds the one-time sketch backward's peak memory
    # (trades a little compute for memory) so the sketch-init does not dominate
    # LoCA's footprint on memory-constrained boxes.
    ckpt_enabled = False
    if getattr(model, "supports_gradient_checkpointing", False):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            ckpt_enabled = True
        except Exception:
            pass
    with torch.enable_grad():
        # frozen params -> inject a differentiable leaf via inputs_embeds (one-time init).
        embeds = h.embed_tokens(input_ids).detach().requires_grad_(True)
        out = model(inputs_embeds=embeds, attention_mask=attn,
                    output_hidden_states=True, use_cache=False)
        hs = out.hidden_states                       # len L+1; hs[l] = block l output
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

    # top-error e (post-final-norm space) at prediction positions
    from .top_error import top_layer_error
    h_L = hs[-1].detach()[predict_mask].to(torch.float32)          # (n_tok, d)
    E = top_layer_error(h_L, targets[predict_mask], h.lm_head, reduction="sum").t()   # (d, n_tok)

    F_list: list[Tensor] = []
    for l in range(L):
        g_full = hs[l + 1].grad                                  # (B, T, d), block l output grad
        g = g_full[predict_mask].to(torch.float32).t()           # (d, n_tok)
        F_l = _ridge_lowrank_map(g, E, rank=rank_sketch)
        F_list.append(F_l)

    model.zero_grad(set_to_none=True)
    if ckpt_enabled:
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
    if was_training:
        model.train()
    return F_list


def build_feedback(
    d: int,
    L: int,
    kind: str,
    seed: int,
    model=None,
    blocks=None,
    probe_batch=None,
    rank_sketch: int = 8,
) -> list[Tensor]:
    """kind in {'random','sketch'}; returns {F_l}_{l=1..L}, all frozen."""
    if kind == "random":
        return build_random_feedback(d, L, seed)
    if kind == "sketch":
        if model is None or probe_batch is None or blocks is None:
            raise ValueError("sketch feedback needs model, blocks, probe_batch")
        return build_sketch_feedback(model, blocks, probe_batch, rank_sketch)
    raise ValueError(f"unknown feedback kind: {kind}")
