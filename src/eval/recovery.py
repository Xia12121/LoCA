"""The recovery metric (C1): how much of LoRA's gain a method recovers.

    recovery = (method - frozen) / (LoRA - frozen)

Higher-is-better metrics (accuracy) and lower-is-better (CE/perplexity) are both
supported via `higher_better`. C1 acceptance: recovery >= 0.85.
"""
from __future__ import annotations


def recovery(method_val: float, frozen_val: float, lora_val: float, higher_better: bool = True) -> float:
    denom = (lora_val - frozen_val) if higher_better else (frozen_val - lora_val)
    numer = (method_val - frozen_val) if higher_better else (frozen_val - method_val)
    if abs(denom) < 1e-12:
        return float("nan")
    return numer / denom


def run_lm_eval_stub(*args, **kwargs):
    """Placeholder for lm-evaluation-harness (MMLU/IFEval). Wired on the GPU box
    where lm-eval is installed; raises if called without it."""
    raise NotImplementedError(
        "lm-eval harness not wired in this environment. Install lm-eval>=0.4.4 and "
        "call lm_eval.simple_evaluate on the adapted model. See scripts/run_lm_eval.py."
    )
