"""Generation-based accuracy (§7): classification labels & format-following.

For clean scalar tasks (SST-2/BoolQ as generation, fixed-JSON format) we greedily
decode the completion and check an exact/regex criterion. Kept simple and
deterministic so it is a stable training signal during sweeps.
"""
from __future__ import annotations

import json
import re

import torch


@torch.no_grad()
def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int = 16, device: str = "cpu") -> str:
    model.eval()
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id)
    gen = out[0, ids.shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)


@torch.no_grad()
def eval_classification(model, tokenizer, raw_examples, label_words, device="cpu", max_new_tokens=4) -> dict:
    """raw_examples: list[(prompt, gold_word)]; label_words: set of valid words.

    Accuracy = first matched label word in the generation equals gold.
    """
    correct, total = 0, 0
    for prompt, gold in raw_examples:
        gen = generate_completion(model, tokenizer, prompt, max_new_tokens, device).lower()
        pred = None
        for w in label_words:
            if w.lower() in gen:
                pred = w.lower()
                break
        correct += int(pred == gold.strip().lower())
        total += 1
    return {"accuracy": correct / max(total, 1), "n": total}


@torch.no_grad()
def eval_json_format(model, tokenizer, raw_examples, device="cpu", max_new_tokens=24) -> dict:
    """Fraction of generations that are valid JSON with an 'answer' key (format-following)."""
    valid, correct, total = 0, 0, 0
    for prompt, gold in raw_examples:
        gen = generate_completion(model, tokenizer, prompt, max_new_tokens, device).strip()
        m = re.search(r"\{.*\}", gen, re.DOTALL)
        ok_fmt = False
        ok_ans = False
        if m:
            try:
                obj = json.loads(m.group(0))
                ok_fmt = "answer" in obj
                try:
                    gold_obj = json.loads(gold)
                    ok_ans = ok_fmt and str(obj.get("answer")).strip() == str(gold_obj.get("answer")).strip()
                except Exception:
                    ok_ans = ok_fmt
            except Exception:
                ok_fmt = False
        valid += int(ok_fmt)
        correct += int(ok_ans)
        total += 1
    return {"format_acc": valid / max(total, 1), "answer_acc": correct / max(total, 1), "n": total}
