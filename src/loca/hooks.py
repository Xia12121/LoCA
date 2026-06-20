"""Single forward pass that caches every block's local quantities (§4.3).

One forward through {W_l, B_l A_l} caches, per block and PER LABEL TOKEN only:
    s_l  = h_{l-1}   (block input)
    h0_l = h_l^base  (block output before adapter)
and returns the top hidden state h_L (for the top-layer error e).

Memory: only label positions are kept (mask), and the caller is expected to
fold them into the streaming Gram immediately (§4.6) rather than hoard across
batches. That streaming is what keeps CPU RAM flat in token count.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor

from ..adapters.residual_lora import AdaptedBlock
from ..adapters.model_utils import get_handles


@dataclass
class BlockCache:
    s: Tensor      # (n_tok, d) block input  = h_{l-1}
    h0: Tensor     # (n_tok, d) base output  = h_l^base


@contextmanager
def caching(blocks: list[AdaptedBlock], label_mask: Tensor | None):
    for b in blocks:
        b.enable_cache(label_mask)
    try:
        yield
    finally:
        for b in blocks:
            b.disable_cache()


@torch.no_grad()
def cached_forward(
    model,
    blocks: list[AdaptedBlock],
    input_ids: Tensor,
    attention_mask: Tensor,
    label_mask: Tensor,
) -> tuple[list[BlockCache], Tensor]:
    """Run ONE forward; return (per-block caches, h_L at label positions).

    h_L is the final hidden state *after* the model's final norm — i.e. exactly
    the tensor the LM head consumes — restricted to label positions.
    """
    h = get_handles(model)
    with caching(blocks, label_mask):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    # last hidden state from the model already has final_norm applied by HF.
    h_last = out.hidden_states[-1]                     # (B, T, d)
    m = label_mask.to(h_last.device)
    h_L = h_last[m].to(torch.float32)                  # (n_tok, d)

    caches = [BlockCache(s=b.cache_s, h0=b.cache_h0) for b in blocks]
    for b in blocks:
        b.clear_cache()
    return caches, h_L
