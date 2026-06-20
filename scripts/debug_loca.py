"""Quick LoCA diagnostics: per-outer-iteration CE + final eval CE across eta/mode.

  python scripts/debug_loca.py --model Qwen/Qwen2.5-0.5B --task sst2 \
         --etas 0.2 0.05 0.01 0.005 0.002 --modes jacobi --T 8

Prints, per setting, the solver's internal global_ce trajectory (should DECREASE
monotonically and not blow up) and the held-out CE vs frozen. Use it to locate the
stable eta before launching the full Phase-1 matrix.
"""
from __future__ import annotations

import argparse
import sys

import torch

sys.path.insert(0, ".")
from src.utils.seed import set_seed
from src.adapters.residual_lora import attach_adapters
from src.adapters.model_utils import get_handles
from src.data.loaders import load_task, make_collate_fn
from src.loca.feedback import build_feedback
from src.loca.solver import LoCASolver, LoCAConfig
from src.loca.runner import _resolve_device, _make_batches
from src.eval.perplexity import eval_perplexity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--task", default="sst2")
    ap.add_argument("--etas", nargs="+", type=float, default=[0.2, 0.05, 0.01, 0.005, 0.002])
    ap.add_argument("--modes", nargs="+", default=["jacobi"])
    ap.add_argument("--feedback", default="sketch")
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--normalize", action="store_true", help="row-normalize feedback signal")
    args = ap.parse_args()

    device = _resolve_device(args.device)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    collate = make_collate_fn(tok.pad_token_id)
    train_exs = load_task(args.task, tok, n=args.n_train, max_len=args.max_len, split="train")
    eval_exs = load_task(args.task, tok, n=args.n_eval, max_len=args.max_len, split="eval")
    batches = _make_batches(train_exs, collate, args.batch_size)

    # frozen reference
    m0 = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device).eval()
    L = len(get_handles(m0).layers)
    fr = eval_perplexity(m0, tok, eval_exs, batch_size=args.batch_size, device=device)["ce"]
    print(f"[debug] model={args.model} L={L} layers  frozen_ce={fr:.4f}")
    del m0
    torch.cuda.empty_cache()

    for mode in args.modes:
        for eta in args.etas:
            set_seed(0)
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device).eval()
            blocks = attach_adapters(model, r=32, seed=0)
            h = get_handles(model)
            probe = {k: v.to(device) for k, v in batches[0].items()}
            F = build_feedback(h.hidden_size, len(blocks), kind=args.feedback, seed=0,
                               model=model, blocks=blocks, probe_batch=probe, rank_sketch=8)
            solver = LoCASolver(model, blocks, F, LoCAConfig(eta=eta, lam=0.1, T=args.T, mode=mode,
                                                            check_monotone=False))
            killed = None
            try:
                hist = solver.fit(batches)
                traj = [round(m.global_ce, 3) for m in hist]
            except Exception as e:
                traj = "KILLED: " + str(e)[:60]
                killed = True
            ce = eval_perplexity(model, tok, eval_exs, batch_size=args.batch_size, device=device)["ce"]
            rec = (fr - ce) / fr
            print(f"[debug] mode={mode} eta={eta:<6} eval_ce={ce:.4f} (frozen {fr:.3f}, rel {rec:+.2%})  traj={traj}")
            del model, F, blocks
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
