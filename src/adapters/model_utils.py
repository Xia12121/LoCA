"""Locate the decoder-layer list, hidden size, final norm and LM head across
common HF causal-LM architectures (GPT2, GPT-NeoX/Pythia, Llama, Qwen2).

We keep this purely structural so the rest of LoCA never special-cases an arch.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


@dataclass
class ModelHandles:
    layers: nn.ModuleList   # the decoder blocks, in order
    hidden_size: int
    final_norm: nn.Module   # norm applied to h_L before the LM head (Identity if none)
    lm_head: nn.Module      # the unembedding Linear (weight may be tied to embeddings)
    embed_tokens: nn.Module


def get_handles(model) -> ModelHandles:
    cfg = model.config
    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    if hidden is None:
        raise ValueError("could not infer hidden size from config")

    # Llama / Qwen2 / Mistral style: model.model.{layers, norm, embed_tokens}
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        inner = model.model
        return ModelHandles(
            layers=inner.layers,
            hidden_size=hidden,
            final_norm=getattr(inner, "norm", nn.Identity()),
            lm_head=model.lm_head,
            embed_tokens=inner.embed_tokens,
        )
    # GPT-NeoX / Pythia: model.gpt_neox.{layers, final_layer_norm, embed_in}
    if hasattr(model, "gpt_neox"):
        inner = model.gpt_neox
        return ModelHandles(
            layers=inner.layers,
            hidden_size=hidden,
            final_norm=getattr(inner, "final_layer_norm", nn.Identity()),
            lm_head=model.embed_out,
            embed_tokens=inner.embed_in,
        )
    # GPT2: model.transformer.{h, ln_f, wte}
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        inner = model.transformer
        return ModelHandles(
            layers=inner.h,
            hidden_size=hidden,
            final_norm=getattr(inner, "ln_f", nn.Identity()),
            lm_head=model.lm_head,
            embed_tokens=inner.wte,
        )
    raise NotImplementedError(f"unsupported architecture: {type(model).__name__}")
