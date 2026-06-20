"""Hyperparameter sweep over eta / r / lam / T (§5 Phase 4, §10).

  python scripts/sweep.py --config configs/loca_f.yaml --param loca.eta \
         --values 0.01 0.02 0.05 0.1 --override model.name=Qwen/Qwen2.5-0.5B

Each value is run via run_loca's machinery; results land in outputs/results.csv
tagged by config_hash so make_figures.py can pivot on the swept parameter.
"""
from __future__ import annotations

import argparse
import copy
import sys

import torch

sys.path.insert(0, ".")
from src.utils.config import load_config, apply_overrides
from src.utils.logging_csv import ResultRow, append_row, config_hash
from src.loca.runner import run_loca, _resolve_device
from src.data.loaders import load_task
from src.eval.perplexity import eval_perplexity


def _set_dotted(cfg, key, val):
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node[p]
    node[parts[-1]] = val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--param", required=True, help="dotted key, e.g. loca.eta")
    ap.add_argument("--values", nargs="+", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--task-name", default=None)
    args = ap.parse_args()

    import yaml
    base = apply_overrides(load_config(args.config), args.override)
    device = _resolve_device(base["model"].get("device", "auto"))
    task_name = args.task_name or base["train"]["dataset"]

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(base["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = getattr(torch, base["model"].get("dtype", "float32"))
    eval_exs = load_task(base["train"]["dataset"], tok, n=base["train"].get("n_eval", 500),
                         max_len=base["train"]["max_len"], split="eval")

    for raw in args.values:
        val = yaml.safe_load(raw)
        cfg = copy.deepcopy(base)
        _set_dotted(cfg, args.param, val)
        chash = config_hash(cfg)
        print(f"[sweep] {args.param}={val} hash={chash}")
        for seed in cfg["runtime"]["seeds"]:
            model = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"], dtype=dtype)
            res = run_loca(model, tok, cfg, seed)
            ppl = eval_perplexity(model, tok, eval_exs, batch_size=cfg["train"]["batch_size"], device=device)
            print(f"   seed={seed} {args.param}={val} ce={ppl['ce']:.4f} killed={res.killed}")
            common = dict(method=cfg.get("method", "loca_f"), model=cfg["model"]["name"],
                          task=task_name, seed=seed, wall_clock_s=res.wall_clock_s,
                          peak_mem_mb=res.peak_mem_mb, config_hash=chash,
                          extra={"swept_param": args.param, "swept_value": val,
                                 "eta": cfg["loca"]["eta"], "lam": cfg["loca"]["lam"],
                                 "T": cfg["loca"]["T"], "r": cfg["adapter"]["r"],
                                 "mode": cfg["loca"].get("mode"), "feedback": cfg["loca"].get("feedback")})
            append_row(cfg["log"]["csv"], ResultRow(metric="ce", value=ppl["ce"], **common))


if __name__ == "__main__":
    main()
