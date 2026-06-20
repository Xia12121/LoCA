"""Rank-classification accuracy (generation-free).

For each example with a candidate set `choices`, score every candidate by its
length-normalized log-likelihood under the (trained) model and pick the argmax.
This is the standard multiple-choice metric (acc_norm) used by lm-eval-harness;
it avoids the noise of free-form generation matching. Works on the tokenized
dicts produced by load_task (which now carry prompt/completion/choices).
"""
from __future__ import annotations
import torch


@torch.no_grad()
def _completion_logprob(model, tok, prompt: str, completion: str, device: str):
    p_ids = tok(prompt, add_special_tokens=True)["input_ids"]
    c_ids = tok(completion, add_special_tokens=False)["input_ids"]
    if len(c_ids) == 0:
        return -1e9, 1
    ids = torch.tensor([p_ids + c_ids], dtype=torch.long, device=device)
    logits = model(input_ids=ids, use_cache=False).logits[0]        # (T, V)
    logp = torch.log_softmax(logits.float(), dim=-1)
    n_p = len(p_ids)
    total = 0.0
    for i, tid in enumerate(c_ids):
        total += logp[n_p - 1 + i, tid].item()                      # hidden at n_p-1+i predicts c_ids[i]
    return total, len(c_ids)


@torch.no_grad()
def eval_rank_accuracy(model, tok, examples, device: str = "cpu", **_):
    """examples: list of dicts (or Examples) carrying prompt/completion/choices.
    Returns {acc_norm, acc_raw, n} or None if no example has choices."""
    model.eval()
    def _get(ex, k):
        return ex.get(k) if isinstance(ex, dict) else getattr(ex, k, None)
    correct = correct_norm = total = 0
    for ex in examples:
        ch = _get(ex, "choices")
        if not ch:
            continue
        prompt = _get(ex, "prompt"); gold = _get(ex, "completion")
        raw, norm = [], []
        for c in ch:
            lp, n = _completion_logprob(model, tok, prompt, c, device)
            raw.append(lp); norm.append(lp / max(n, 1))
        if ch[max(range(len(ch)), key=lambda i: raw[i])] == gold:
            correct += 1
        if ch[max(range(len(ch)), key=lambda i: norm[i])] == gold:
            correct_norm += 1
        total += 1
    if total == 0:
        return None
    return {"acc_norm": correct_norm / total, "acc_raw": correct / total, "n": total}
