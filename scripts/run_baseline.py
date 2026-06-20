"""Baseline entry point (frozen / lora / mezo / full_sft), §6.

  python scripts/run_baseline.py --config configs/baselines/lora.yaml
  python scripts/run_baseline.py --config configs/baselines/mezo.yaml --override mezo.steps=2000

Writes held-out perplexity + wall-clock + peak memory to outputs/results.csv,
on the SAME machine/dtype/threads as LoCA for a fair comparison.
"""
from __future__ import annotations

import argparse
import sys

import torch

sys.path.insert(0, ".")
from src.utils.config import load_config, apply_overrides
from src.utils.logging_csv import ResultRow, append_row, config_hash
from src.utils.profiling import profile_block
from src.utils.seed import set_seed
from src.loca.runner import _resolve_device, _make_batches
from src.data.loaders import load_task, make_collate_fn
from src.eval.perplexity import eval_perplexity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--task-name", default=None)
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    method = cfg["method"]
    chash = config_hash(cfg)
    device = _resolve_device(cfg["model"].get("device", "auto"))
    task_name = args.task_name or cfg["train"]["dataset"]
    print(f"[run_baseline] method={method} config_hash={chash} device={device}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = getattr(torch, cfg["model"].get("dtype", "float32"))
    collate = make_collate_fn(tok.pad_token_id)

    train_exs = load_task(cfg["train"]["dataset"], tok, n=cfg["train"]["n_train"],
                          max_len=cfg["train"]["max_len"], split="train")
    eval_exs = load_task(cfg["train"]["dataset"], tok, n=cfg["train"].get("n_eval", 500),
                         max_len=cfg["train"]["max_len"], split="eval")
    batches = _make_batches(train_exs, collate, cfg["train"]["batch_size"])

    for seed in cfg["runtime"]["seeds"]:
        set_seed(seed)
        model = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"], dtype=dtype).to(device)
        if device == "cpu":
            nthreads = cfg.get("runtime", {}).get("cpu_threads", -1)
            if nthreads and nthreads > 0:
                torch.set_num_threads(nthreads)

        with profile_block(device) as prof:
            if method == "frozen":
                from src.baselines.frozen import train_frozen
                model, _ = train_frozen(model)
            elif method == "lora":
                from src.baselines.lora_sft import train_lora
                model, _ = train_lora(model, batches, cfg["lora"], device=device)
            elif method == "mezo":
                from src.baselines.mezo import train_mezo, MeZOConfig
                mc = cfg["mezo"]
                model, _, _ = train_mezo(model, tok, batches,
                                         MeZOConfig(eps=mc["eps"], lr=mc["lr"],
                                                    n_perturb=mc["n_perturb"], steps=mc["steps"]),
                                         seed=seed, device=device)
            elif method == "full_sft":
                from src.baselines.full_sft import train_full_sft
                model, _ = train_full_sft(model, batches, cfg["full_sft"], device=device)
            else:
                raise ValueError(f"unknown method {method}")

        ppl = eval_perplexity(model, tok, eval_exs, batch_size=cfg["train"]["batch_size"], device=device)
        mem = prof.peak_mem_mb if device == "cpu" else prof.cuda_peak_mb
        print(f"[run_baseline] {method} seed={seed} ce={ppl['ce']:.4f} ppl={ppl['perplexity']:.2f} "
              f"wall={prof.wall_clock_s:.1f}s mem={mem:.0f}MB")
        common = dict(method=method, model=cfg["model"]["name"], task=task_name, seed=seed,
                      wall_clock_s=prof.wall_clock_s, peak_mem_mb=mem, config_hash=chash, extra={})
        append_row(cfg["log"]["csv"], ResultRow(metric="ce", value=ppl["ce"], **common))
        append_row(cfg["log"]["csv"], ResultRow(metric="perplexity", value=ppl["perplexity"], **common))


if __name__ == "__main__":
    main()
