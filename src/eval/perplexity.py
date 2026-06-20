"""Held-out cross-entropy / perplexity on label tokens (§7)."""
from __future__ import annotations

import math

import torch

from ..data.loaders import make_collate_fn


@torch.no_grad()
def eval_perplexity(model, tokenizer, examples, batch_size: int = 8, device: str = "cpu") -> dict:
    """Mean per-token CE and perplexity over label positions of `examples`."""
    model.eval()
    collate = make_collate_fn(tokenizer.pad_token_id)
    total_ce, total_tok = 0.0, 0
    for i in range(0, len(examples), batch_size):
        batch = collate(examples[i:i + batch_size])
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=ids, attention_mask=attn, use_cache=False)
        logits = out.logits[:, :-1, :]
        tgt = labels[:, 1:]
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1),
            ignore_index=-100, reduction="sum")
        ntok = int((tgt != -100).sum())
        total_ce += float(ce)
        total_tok += ntok
    mean_ce = total_ce / max(total_tok, 1)
    return {"ce": mean_ce, "perplexity": math.exp(min(mean_ce, 50)), "n_tokens": total_tok}
