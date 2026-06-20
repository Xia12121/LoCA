"""ResidualLoRA: additive low-rank correction to a transformer block's OUTPUT
residual stream (§4.2). This placement is the key design decision that makes the
closed-form solve (eq. 8) EXACT rather than approximate:

    h_l = f_l^base(h_{l-1})  +  B_l A_l s_l ,   with  s_l = h_{l-1}

    - A_l : (r, d) frozen random ~ N(0, 1/d)
    - B_l : (d, r) trainable, init 0   (closed-form solved)
    - B_l = 0  =>  h_l == h_l^base  (exact frozen base)

Because the correction is a linear, additive, DIRECT edit of h_l (it does NOT pass
through any attention/MLP nonlinearity), B_l A_l s_l == rho_l holds exactly and the
ridge closed form is precise.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..utils.seed import seed_generator


class ResidualLoRA(nn.Module):
    """Additive low-rank correction on a block's output residual stream."""

    def __init__(self, d_model: int, r: int, seed: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_model = d_model
        self.r = r
        g = seed_generator(seed)
        # A: (r, d) frozen ~ N(0, 1/d). Fixed random projection.
        A = torch.randn(r, d_model, generator=g, dtype=torch.float32) / (d_model ** 0.5)
        self.register_buffer("A", A.to(dtype))
        # B: (d, r) trainable, init 0 -> starts as exact base.
        self.B = nn.Parameter(torch.zeros(d_model, r, dtype=dtype))
        self.B.requires_grad_(False)  # LoCA solves B in closed form, not by autograd

    def project(self, s: Tensor) -> Tensor:
        """p = A s  ; s: (..., d) -> p: (..., r)."""
        return s.to(self.A.dtype) @ self.A.t()

    def correction(self, block_in: Tensor) -> Tensor:
        """B A s = (s @ A^T) @ B^T ; block_in is s_l = h_{l-1}."""
        p = self.project(block_in)                  # (..., r)
        return p @ self.B.t()                       # (..., d)

    def forward(self, block_out: Tensor, block_in: Tensor) -> Tensor:
        """Return h_l = h_l^base + B A s."""
        return block_out + self.correction(block_in).to(block_out.dtype)

    @torch.no_grad()
    def set_B(self, B_new: Tensor) -> None:
        """Write back a closed-form solution. B_new: (d, r)."""
        assert B_new.shape == self.B.shape, f"{B_new.shape} != {self.B.shape}"
        self.B.copy_(B_new.to(self.B.dtype))

    @torch.no_grad()
    def reset(self) -> None:
        self.B.zero_()


class AdaptedBlock(nn.Module):
    """Wraps one HF decoder block, applying a ResidualLoRA to its output and
    (optionally) caching the per-token local quantities for one forward pass.

    Cache (label positions only, see set_cache_mask):
        s   = block_in   (h_{l-1})          -> (n_tok, d)
        h0  = block_out  (h_l^base)         -> (n_tok, d)
    h_l (adapted) is recoverable as h0 + adapter.correction(s); we don't store it.
    """

    def __init__(self, block: nn.Module, adapter: ResidualLoRA, layer_idx: int):
        super().__init__()
        self.block = block
        self.adapter = adapter
        self.layer_idx = layer_idx
        self._caching = False
        self._cache_mask: Tensor | None = None   # (B, T) bool, label positions
        self.cache_s: Tensor | None = None
        self.cache_h0: Tensor | None = None

    # -- caching control ---------------------------------------------------- #
    def enable_cache(self, mask: Tensor | None) -> None:
        self._caching = True
        self._cache_mask = mask
        self.cache_s = None
        self.cache_h0 = None

    def disable_cache(self) -> None:
        self._caching = False
        self._cache_mask = None

    def clear_cache(self) -> None:
        self.cache_s = None
        self.cache_h0 = None

    # -- forward ------------------------------------------------------------ #
    def _extract_hidden(self, out):
        if isinstance(out, tuple):
            return out[0], out
        return out, None

    def forward(self, *args, **kwargs):
        # block_in (h_{l-1}) is the first positional arg or the `hidden_states` kwarg.
        if args:
            block_in = args[0]
        else:
            block_in = kwargs["hidden_states"]

        out = self.block(*args, **kwargs)
        block_out, full = self._extract_hidden(out)        # h_l^base

        if self._caching:
            self._store(block_in, block_out)

        h = self.adapter(block_out, block_in)              # h_l = base + correction
        if full is None:
            return h
        return (h,) + tuple(full[1:])

    @torch.no_grad()
    def _store(self, block_in: Tensor, block_out: Tensor) -> None:
        if self._cache_mask is not None:
            m = self._cache_mask.to(block_in.device)
            s = block_in[m]                                # (n_tok, d)
            h0 = block_out[m]
        else:
            s = block_in.reshape(-1, block_in.shape[-1])
            h0 = block_out.reshape(-1, block_out.shape[-1])
        self.cache_s = s.detach().to(torch.float32)
        self.cache_h0 = h0.detach().to(torch.float32)


def attach_adapters(model, r: int, seed: int, dtype: torch.dtype = torch.float32) -> list[AdaptedBlock]:
    """Wrap every decoder block of `model` with an AdaptedBlock. Returns the list
    of AdaptedBlocks in layer order. The base model weights are left frozen.
    """
    from .model_utils import get_handles

    for p in model.parameters():
        p.requires_grad_(False)

    h = get_handles(model)
    d = h.hidden_size
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    adapted: list[AdaptedBlock] = []
    for i in range(len(h.layers)):
        adapter = ResidualLoRA(d, r=r, seed=seed * 100003 + i, dtype=dtype).to(device)
        ab = AdaptedBlock(h.layers[i], adapter, layer_idx=i)
        h.layers[i] = ab
        adapted.append(ab)
    return adapted
