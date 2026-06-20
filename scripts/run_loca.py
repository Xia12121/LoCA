"""Main LoCA entry point.

  python scripts/run_loca.py --config configs/loca_f.yaml \
         --override loca.eta=0.05 train.n_train=2000 model.name=Qwen/Qwen2.5-0.5B

Trains LoCA, evaluates held-out perplexity, writes one row per (seed, metric) to
outputs/results.csv with wall-clock and peak memory.
"""
from __future__ import annotations

import argparse
import sys

import torch

sys.path.insert(0, ".")
from src.utils.config import load_config, apply_overrides
from src.utils.logging_csv import ResultRow, append_row, config_hash
from src.loca.runner import run_loca, _resolve_device
from src.data.loaders import load_task
from src.eval.perplexity import eval_perplexity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--task-name", default=None, help="label for the CSV task column")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    chash = config_hash(cfg)
    device = _resolve_device(cfg["model"].get("device", "auto"))
    task_name = args.task_name or cfg["train"]["dataset"]
    print(f"[run_loca] config_hash={chash} device={device} method={cfg.get('method')}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = getattr(torch, cfg["model"].get("dtype", "float32"))

    eval_exs = load_task(cfg["train"]["dataset"], tok, n=cfg["train"].get("n_eval", 500),
                         max_len=cfg["train"]["max_len"], split="eval")

    for seed in cfg["runtime"]["seeds"]:
        model = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"], dtype=dtype)
        res = run_loca(model, tok, cfg, seed)
        ppl = eval_perplexity(model, tok, eval_exs, batch_size=cfg["train"]["batch_size"], device=device)
        print(f"[run_loca] seed={seed} ce={ppl['ce']:.4f} ppl={ppl['perplexity']:.2f} "
              f"wall={res.wall_clock_s:.1f}s mem={res.peak_mem_mb:.0f}MB killed={res.killed}")

        common = dict(method=cfg.get("method", "loca_f"), model=cfg["model"]["name"],
                      task=task_name, seed=seed, wall_clock_s=res.wall_clock_s,
                      peak_mem_mb=res.peak_mem_mb, config_hash=chash,
                      extra={"killed": res.killed, "eta": cfg["loca"]["eta"],
                             "lam": cfg["loca"]["lam"], "T": cfg["loca"]["T"],
                             "mode": cfg["loca"].get("mode"), "feedback": cfg["loca"].get("feedback")})
        append_row(cfg["log"]["csv"], ResultRow(metric="ce", value=ppl["ce"], **common))
        append_row(cfg["log"]["csv"], ResultRow(metric="perplexity", value=ppl["perplexity"], **common))
        if res.history:
            append_row(cfg["log"]["csv"], ResultRow(metric="final_train_ce",
                                                    value=res.history[-1].global_ce, **common))


if __name__ == "__main__":
    main()
