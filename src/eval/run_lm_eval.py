"""MMLU / IFEval via lm-evaluation-harness (§7). Runs on the GPU box where
lm-eval>=0.4.4 is installed. Wraps the (adapter-applied) HF model.

Usage (programmatic):
    from src.eval.run_lm_eval import run_lm_eval
    res = run_lm_eval(model, tokenizer, tasks=["mmlu","ifeval"], device="cuda")
"""
from __future__ import annotations


def run_lm_eval(model, tokenizer, tasks, device="cuda", batch_size="auto", limit=None) -> dict:
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError as e:
        raise ImportError("install lm-eval>=0.4.4 on the GPU box to run MMLU/IFEval") from e

    lm = HFLM(pretrained=model, tokenizer=tokenizer, device=device, batch_size=batch_size)
    out = lm_eval.simple_evaluate(model=lm, tasks=list(tasks), limit=limit)
    # flatten the metrics we care about
    flat = {}
    for task, metrics in out.get("results", {}).items():
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                flat[f"{task}/{k}"] = v
    return flat
