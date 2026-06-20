"""End-to-end LoCA training run: model -> adapters -> feedback -> solve -> eval.

Wraps Algorithm 1 with config loading, batching, profiling and CSV logging so a
whole experiment is one function call (used by scripts/run_loca.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..adapters.residual_lora import attach_adapters
from ..adapters.model_utils import get_handles
from ..data.loaders import load_task, make_collate_fn
from ..utils.profiling import profile_block
from ..utils.seed import set_seed
from .feedback import build_feedback
from .solver import LoCASolver, LoCAConfig, KillSignal


def _resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def _make_batches(examples, collate, batch_size):
    return [collate(examples[i:i + batch_size]) for i in range(0, len(examples), batch_size)]


@dataclass
class LoCARunResult:
    history: list
    wall_clock_s: float
    peak_mem_mb: float
    killed: str | None
    blocks: Any
    feedback: Any
    best_t: int = -1
    best_score: float = float("nan")


def run_loca(model, tokenizer, cfg: dict, seed: int, eval_fn=None) -> LoCARunResult:
    set_seed(seed)
    device = _resolve_device(cfg["model"].get("device", "auto"))
    model = model.to(device).eval()

    if device == "cpu":
        nthreads = cfg.get("runtime", {}).get("cpu_threads", -1)
        if nthreads and nthreads > 0:
            torch.set_num_threads(nthreads)

    lc = cfg["loca"]
    r = cfg["adapter"]["r"]
    dtype = getattr(torch, cfg["model"].get("dtype", "float32"))

    blocks = attach_adapters(model, r=r, seed=seed, dtype=dtype)
    h = get_handles(model)

    # data
    collate = make_collate_fn(tokenizer.pad_token_id)
    exs = load_task(cfg["train"]["dataset"], tokenizer,
                    n=cfg["train"]["n_train"], max_len=cfg["train"]["max_len"], split="train")
    batches = _make_batches(exs, collate, cfg["train"]["batch_size"])

    # feedback (sketch needs a probe batch)
    fb_kind = lc.get("feedback", "random")
    probe = batches[0] if fb_kind == "sketch" else None
    if probe is not None:
        probe = {k: v.to(device) for k, v in probe.items()}
    feedback = build_feedback(h.hidden_size, len(blocks), kind=fb_kind, seed=seed,
                              model=model, blocks=blocks, probe_batch=probe,
                              rank_sketch=lc.get("sketch_rank", 8))

    solver_cfg = LoCAConfig(
        eta=lc["eta"], lam=lc["lam"], T=lc["T"],
        mode=lc.get("mode", "jacobi"), variant=lc.get("variant", "F"),
        target_norm=lc.get("target_norm", "none"),
    )
    solver = LoCASolver(model, blocks, feedback, solver_cfg)

    killed = None
    with profile_block(device) as prof:
        try:
            solver.fit(batches, eval_fn=eval_fn)
        except KillSignal as ks:
            killed = ks.key
            print(f"[run_loca] KILL: {ks}")

    return LoCARunResult(
        history=solver.history,
        wall_clock_s=prof.wall_clock_s,
        peak_mem_mb=prof.peak_mem_mb if device == "cpu" else prof.cuda_peak_mb,
        killed=killed,
        blocks=blocks,
        feedback=feedback,
        best_t=getattr(solver, "best_t", -1),
        best_score=getattr(solver, "best_score", float("nan")),
    )
