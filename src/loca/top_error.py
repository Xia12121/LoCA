"""Top-layer error e (§4.4) — the ONLY place an error signal is computed, and it
passes through the LM head ONLY (never reverse through the backbone).

For CE loss with an LM head z_t = W_unembed h_{L,t}:

    e_t = W_unembed^T ( softmax(z_t) - onehot(y_t) ) = (p_t - onehot(y_t)) @ W_unembed

This is exactly d(CE_sum)/d(h_L) at the label positions; prompt/pad get 0.
We compute it in closed form (no autograd, no backward graph).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def top_layer_error(
    h_L: Tensor,
    labels: Tensor,
    lm_head: torch.nn.Module,
    reduction: str = "sum",
    vocab_chunk: int = 0,
) -> Tensor:
    """Per-token top error e_t.

    Args:
        h_L:    (n_tok, d) hidden states the LM head consumes (post final-norm),
                already restricted to the label/valid positions.
        labels: (n_tok,) target token ids for those same positions (no -100 here).
        lm_head: the unembedding Linear (weight (V, d), maybe tied, maybe bias).
        reduction: 'sum' -> e is grad of summed CE; 'mean' -> divide by n_tok.
        vocab_chunk: if >0, chunk the (p-onehot)@W matmul over vocab to cap memory.

    Returns:
        e: (n_tok, d) float32 error vectors. Does NOT call backward.
    """
    W = lm_head.weight                      # (V, d)
    b = getattr(lm_head, "bias", None)
    h = h_L.to(W.dtype)
    z = h @ W.t()
    if b is not None:
        z = z + b
    # stable softmax
    p = F.softmax(z.float(), dim=-1)        # (n_tok, V)
    n_tok, V = p.shape
    p[torch.arange(n_tok, device=p.device), labels] -= 1.0   # p - onehot

    Wf = W.float()
    if vocab_chunk and V > vocab_chunk:
        e = torch.zeros(n_tok, h.shape[-1], dtype=torch.float32, device=h.device)
        for start in range(0, V, vocab_chunk):
            end = min(start + vocab_chunk, V)
            e += p[:, start:end] @ Wf[start:end]
    else:
        e = p @ Wf                          # (n_tok, d)

    if reduction == "mean":
        e = e / max(n_tok, 1)
    return e
