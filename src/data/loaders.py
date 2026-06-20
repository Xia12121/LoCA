"""Dataset loaders → tokenized (input_ids, labels, attention_mask) examples.

Design rules (shared by ALL methods for fairness, §6):
- Same tokenization, same prompt template, same train/eval split per task.
- `labels` masks prompt + pad positions with -100 so loss / top-error e are computed
  ONLY on completion (label) tokens. LoCA's hooks rely on this same mask.
- Returned examples are plain dicts of 1-D LongTensors (un-padded). The collate_fn
  pads a batch and builds the label_mask.

Tasks (§3.2):
  - instruction:  tatsu-lab/alpaca / dolly        (C1/C2 main)
  - domain:       a small domain corpus slice      (C1, C4(b) shift sweep)
  - format:       fixed-format SFT (JSON / tone)    (C1, clean)
  - classify:     SST-2 / BoolQ as generation       (clean scalar signal)
  - gsm8k:        negative control (should fail)     (C4(b))
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch.nn.utils.rnn import pad_sequence


# --------------------------------------------------------------------------- #
# Prompt templates
# --------------------------------------------------------------------------- #
ALPACA_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the "
    "request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
ALPACA_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
)


@dataclass
class Example:
    prompt: str
    completion: str
    choices: list | None = None


# --------------------------------------------------------------------------- #
# Raw text builders: dataset name -> list[Example]
# --------------------------------------------------------------------------- #
def _build_alpaca(n: int, split: str) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = _slice(ds, n, split)
    out = []
    for r in ds:
        if r.get("input", "").strip():
            prompt = ALPACA_WITH_INPUT.format(instruction=r["instruction"], input=r["input"])
        else:
            prompt = ALPACA_NO_INPUT.format(instruction=r["instruction"])
        out.append(Example(prompt=prompt, completion=r["output"].strip()))
    return out


def _build_dolly(n: int, split: str) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    ds = _slice(ds, n, split)
    out = []
    for r in ds:
        ctx = r.get("context", "").strip()
        if ctx:
            prompt = ALPACA_WITH_INPUT.format(instruction=r["instruction"], input=ctx)
        else:
            prompt = ALPACA_NO_INPUT.format(instruction=r["instruction"])
        out.append(Example(prompt=prompt, completion=r["response"].strip()))
    return out


def _build_sst2(n: int, split: str) -> list[Example]:
    from datasets import load_dataset

    hf_split = "train" if split == "train" else "validation"
    try:
        ds = load_dataset("nyu-mll/glue", "sst2", split=hf_split)   # canonical namespace (newer datasets)
    except Exception:
        ds = load_dataset("glue", "sst2", split=hf_split)            # legacy fallback
    ds = _slice(ds, n, split, already_split=True)
    label_word = {0: "negative", 1: "positive"}
    out = []
    for r in ds:
        prompt = f"Review: {r['sentence'].strip()}\nSentiment (positive or negative):"
        out.append(Example(prompt=prompt, completion=" " + label_word[r["label"]], choices=[" " + label_word[0], " " + label_word[1]]))
    return out


def _build_boolq(n: int, split: str) -> list[Example]:
    from datasets import load_dataset

    hf_split = "train" if split == "train" else "validation"
    ds = load_dataset("google/boolq", split=hf_split)
    ds = _slice(ds, n, split, already_split=True)
    out = []
    for r in ds:
        prompt = f"Passage: {r['passage'].strip()}\nQuestion: {r['question'].strip()}\nAnswer (yes or no):"
        out.append(Example(prompt=prompt, completion=" " + ("yes" if r["answer"] else "no"), choices=[" yes", " no"]))
    return out


def _build_hellaswag(n: int, split: str) -> list[Example]:
    """HellaSwag commonsense completion: ctx -> correct ending (CE on gold ending)."""
    from datasets import load_dataset

    hf_split = "train" if split == "train" else "validation"
    ds = load_dataset("Rowan/hellaswag", split=hf_split)
    ds = _slice(ds, n, split, already_split=True)
    out = []
    for r in ds:
        label = r["label"]
        if label == "":
            continue
        prompt = r["ctx"].strip()
        out.append(Example(prompt=prompt, completion=" " + r["endings"][int(label)].strip()))
    return out


def _build_arc(n: int, split: str, subset: str = "ARC-Easy") -> list[Example]:
    """ARC science QA: question -> correct choice text (CE on gold answer)."""
    from datasets import load_dataset

    hf_split = "train" if split == "train" else "validation"
    ds = load_dataset("allenai/ai2_arc", subset, split=hf_split)
    ds = _slice(ds, n, split, already_split=True)
    out = []
    for r in ds:
        ch = r["choices"]
        ak = r["answerKey"]
        if ak not in ch["label"]:
            continue
        text = ch["text"][ch["label"].index(ak)]
        prompt = f"Question: {r['question'].strip()}\nAnswer:"
        out.append(Example(prompt=prompt, completion=" " + text.strip(), choices=[" " + t.strip() for t in ch["text"]]))
    return out


def _build_arc_challenge(n: int, split: str) -> list[Example]:
    return _build_arc(n, split, subset="ARC-Challenge")


def _build_piqa(n: int, split: str) -> list[Example]:
    """PIQA physical commonsense: goal -> correct solution (CE on gold solution)."""
    from datasets import load_dataset

    hf_split = "train" if split == "train" else "validation"
    ds = load_dataset("piqa", split=hf_split, trust_remote_code=True)
    ds = _slice(ds, n, split, already_split=True)
    out = []
    for r in ds:
        sols = [r["sol1"], r["sol2"]]
        prompt = f"Goal: {r['goal'].strip()}\nSolution:"
        out.append(Example(prompt=prompt, completion=" " + sols[int(r["label"])].strip()))
    return out


def _build_gsm8k(n: int, split: str) -> list[Example]:
    from datasets import load_dataset

    hf_split = "train" if split == "train" else "test"
    ds = load_dataset("gsm8k", "main", split=hf_split)
    ds = _slice(ds, n, split, already_split=True)
    out = []
    for r in ds:
        prompt = f"Question: {r['question'].strip()}\nAnswer:"
        out.append(Example(prompt=prompt, completion=" " + r["answer"].strip()))
    return out


def _build_format_json(n: int, split: str) -> list[Example]:
    """Synthetic fixed-JSON-format SFT: clean, auto-judgeable (§3.2 style/format).

    Teaches the model to answer every query as {"answer": "..."} on one line.
    Deterministic, no download — good for Phase 0/1 smoke + format-accuracy eval.
    """
    import json

    topics = [
        ("capital of France", "Paris"), ("largest planet", "Jupiter"),
        ("speed of light unit", "m/s"), ("author of Hamlet", "Shakespeare"),
        ("chemical symbol for gold", "Au"), ("number of continents", "7"),
        ("opposite of hot", "cold"), ("color of the sky", "blue"),
        ("square root of 81", "9"), ("first US president", "Washington"),
    ]
    rng = list(range(len(topics)))
    out = []
    k = 0
    while len(out) < n:
        q, a = topics[rng[k % len(topics)]]
        prompt = f"Question: What is the {q}?\nRespond in JSON.\n"
        completion = json.dumps({"answer": a})
        out.append(Example(prompt=prompt, completion=completion))
        k += 1
    # deterministic train/eval split by stride
    return out[:n] if split == "train" else out[: max(1, n)]


def _build_domain(n: int, split: str, hf_name: str = "wikitext", subset: str = "wikitext-2-raw-v1") -> list[Example]:
    """Light domain adaptation: language-model continuation on a domain corpus.

    Prompt is empty (pure LM); completion is the text. Used for perplexity-drop
    measurement and the C4(b) domain-shift sweep (swap hf_name for the far domain).
    """
    from datasets import load_dataset

    ds = load_dataset(hf_name, subset, split="train" if split == "train" else "validation")
    out = []
    for r in ds:
        t = r["text"].strip()
        if len(t) < 32:
            continue
        out.append(Example(prompt="", completion=t))
        if len(out) >= n:
            break
    return out


_BUILDERS: dict[str, Callable[..., list[Example]]] = {
    "tatsu-lab/alpaca": _build_alpaca,
    "alpaca": _build_alpaca,
    "databricks/databricks-dolly-15k": _build_dolly,
    "dolly": _build_dolly,
    "sst2": _build_sst2,
    "glue/sst2": _build_sst2,
    "boolq": _build_boolq,
    "hellaswag": _build_hellaswag,
    "arc": _build_arc,
    "arc_easy": _build_arc,
    "arc_challenge": _build_arc_challenge,
    "piqa": _build_piqa,
    "gsm8k": _build_gsm8k,
    "format_json": _build_format_json,
    "domain": _build_domain,
    "wikitext": _build_domain,
}


def _slice(ds, n: int, split: str, already_split: bool = False):
    """Take a deterministic slice. For single-split datasets, carve eval off the tail."""
    if already_split:
        return ds.select(range(min(n, len(ds))))
    total = len(ds)
    if split == "train":
        end = min(n, int(total * 0.9))
        return ds.select(range(end))
    else:  # eval from the tail, disjoint from train
        start = int(total * 0.9)
        end = min(start + n, total)
        return ds.select(range(start, end))


# --------------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------------- #
def tokenize_example(ex: Example, tokenizer, max_len: int) -> dict[str, torch.Tensor]:
    """Tokenize prompt+completion; mask prompt and pad in labels with -100."""
    prompt_ids = tokenizer(ex.prompt, add_special_tokens=True)["input_ids"] if ex.prompt else (
        [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    )
    comp_ids = tokenizer(ex.completion, add_special_tokens=False)["input_ids"]
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    input_ids = (prompt_ids + comp_ids + eos)[:max_len]
    n_prompt = min(len(prompt_ids), len(input_ids))
    labels = list(input_ids)
    for i in range(n_prompt):
        labels[i] = -100  # mask prompt
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        "prompt": ex.prompt,
        "completion": ex.completion,
        "choices": ex.choices,
    }


def load_task(
    name: str,
    tokenizer,
    n: int,
    max_len: int,
    split: str = "train",
    **builder_kwargs: Any,
) -> list[dict[str, torch.Tensor]]:
    """Load + tokenize a task into a list of example tensors."""
    if name not in _BUILDERS:
        raise KeyError(f"unknown dataset '{name}'. known: {sorted(_BUILDERS)}")
    raw = _BUILDERS[name](n, split, **builder_kwargs)
    return [tokenize_example(ex, tokenizer, max_len) for ex in raw]


# --------------------------------------------------------------------------- #
# Collation
# --------------------------------------------------------------------------- #
def make_collate_fn(pad_token_id: int) -> Callable:
    """Pad a batch (right-pad) and add the causal-LM prediction alignment.

    Returns, besides input_ids/labels/attention_mask/label_mask:
      predict_mask : (B,T) bool — position t is "active" if it PREDICTS a label,
                     i.e. labels[t+1] != -100. The hidden state at t predicts t+1,
                     so LoCA's features s_t, top-error e_t, and target token all
                     live at these prediction positions (NOT the label positions).
      targets      : (B,T) long — the next token input_ids[t+1] at predict
                     positions, -100 elsewhere. targets[predict_mask] gives the
                     1-D per-prediction target ids.
    """

    def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_token_id)
        labels = pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100)
        attn = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)

        label_mask = labels.ne(-100)
        predict_mask = torch.zeros_like(label_mask)
        predict_mask[:, :-1] = label_mask[:, 1:]                 # t active iff t+1 is a label
        targets = torch.full_like(input_ids, -100)
        targets[:, :-1] = input_ids[:, 1:]
        targets = targets.masked_fill(~predict_mask, -100)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attn,
            "label_mask": label_mask,
            "predict_mask": predict_mask,
            "targets": targets,
        }

    return collate
