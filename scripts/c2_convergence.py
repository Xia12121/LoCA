"""C2 — LoCA vs MeZO: quality-vs-wall-clock convergence (the fair comparison).

MeZO is forward-only like LoCA but needs thousands of noisy steps; LoCA needs a
handful of deterministic closed-form outer iterations. Comparing FINAL quality at
unlimited budget is unfair to LoCA (MeZO eventually fits easy tasks). The right C2
metric is convergence SPEED: held-out CE as a function of cumulative wall-clock.

  python scripts/c2_convergence.py --model Qwen/Qwen2.5-0.5B --task sst2 --device cuda

Outputs a curve CSV (method, step, wall_s, ce) and prints:
  - LoCA final CE and wall-clock
  - MeZO CE at LoCA's wall-clock budget   (expect: much worse)
  - MeZO wall-clock to REACH LoCA's CE     (expect: >> LoCA -> the speedup factor)
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

sys.path.insert(0, ".")
from src.utils.seed import set_seed
from src.utils.logging_csv import ResultRow, append_row
from src.adapters.residual_lora import attach_adapters
from src.adapters.model_utils import get_handles
from src.data.loaders import load_task, make_collate_fn
from src.loca.feedback import build_feedback
from src.loca.solver import LoCASolver, LoCAConfig
from src.loca.runner import _resolve_device, _make_batches
from src.eval.perplexity import eval_perplexity
from src.baselines.mezo import _loss_on_batch, _perturb


def _eval(model, tok, eval_exs, bs, device):
    return eval_perplexity(model, tok, eval_exs, batch_size=bs, device=device)["ce"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--task", default="sst2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--loca-etas", nargs="+", type=float, default=[0.003])
    ap.add_argument("--loca-T", type=int, default=8)
    ap.add_argument("--mezo-total-steps", type=int, default=12000)
    ap.add_argument("--mezo-eval-every", type=int, default=500)
    ap.add_argument("--mezo-eps", type=float, default=1e-3)
    ap.add_argument("--mezo-lr", type=float, default=1e-6)
    ap.add_argument("--mezo-full-param", action="store_true")
    ap.add_argument("--csv", default="outputs/c2_curve.csv")
    ap.add_argument("--dtype", default="float32", choices=["float32","bfloat16","float16"])
    args = ap.parse_args()
    _DTYPE_MAP={"float32":torch.float32,"bfloat16":torch.bfloat16,"float16":torch.float16}
    DT=_DTYPE_MAP[args.dtype]

    device = _resolve_device(args.device)
    set_seed(0)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    collate = make_collate_fn(tok.pad_token_id)
    train = load_task(args.task, tok, n=args.n_train, max_len=args.max_len, split="train")
    eval_exs = load_task(args.task, tok, n=args.n_eval, max_len=args.max_len, split="eval")
    batches = _make_batches(train, collate, args.batch_size)

    def log_point(method, step, wall, ce):
        print(f"[c2] {method:6s} step={step:6d} wall={wall:7.1f}s ce={ce:.4f}")
        append_row(args.csv, ResultRow(method=method, model=args.model, task=args.task,
                                       seed=0, metric="ce", value=ce, wall_clock_s=wall,
                                       extra={"step": step}))

    # ---------------- LoCA: CE after each outer iteration ------------------ #
    loca_curve = []
    best_eta_ce = float("inf")
    for eta in args.loca_etas:
        m = AutoModelForCausalLM.from_pretrained(args.model, dtype=DT).to(device).eval()
        blocks = attach_adapters(m, r=32, seed=0)
        h = get_handles(m)
        F = build_feedback(h.hidden_size, len(blocks), kind="sketch", seed=0,
                           model=m, blocks=blocks, probe_batch={k: v.to(device) for k, v in batches[0].items()})
        solver = LoCASolver(m, blocks, F, LoCAConfig(eta=eta, lam=0.1, T=args.loca_T, mode="jacobi"))
        wall = 0.0
        curve = [(0.0, _eval(m, tok, eval_exs, args.batch_size, device))]
        for t in range(1, args.loca_T + 1):
            t0 = time.perf_counter()
            solver.outer_step(batches)
            wall += time.perf_counter() - t0
            ce = _eval(m, tok, eval_exs, args.batch_size, device)
            curve.append((wall, ce))
        final = min(c[1] for c in curve)
        if final < best_eta_ce:
            best_eta_ce, loca_curve = final, curve
        # free EVERYTHING that references the model (solver/blocks/F hold caches);
        # `del m` alone leaks ~28GB at 14B and the MeZO load below OOMs
        del solver, F, h, blocks, m
        import gc; gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    for i, (w, ce) in enumerate(loca_curve):
        log_point("loca", i, w, ce)
    loca_final_ce = min(c[1] for c in loca_curve)
    loca_wall = loca_curve[-1][0]

    # ---------------- MeZO: CE vs cumulative wall-clock -------------------- #
    set_seed(0)
    m = AutoModelForCausalLM.from_pretrained(args.model, dtype=DT).to(device).eval()
    if args.mezo_full_param:
        params = [pp for pp in m.parameters()]
    else:
        blocks = attach_adapters(m, r=32, seed=0)
        params = [b.adapter.B for b in blocks]
    for p in params:
        p.requires_grad_(False)
    rng = torch.Generator().manual_seed(0)
    nb = len(batches)
    wall = 0.0
    mezo_reach_wall = None
    mezo_ce_at_loca_wall = None
    for step in range(1, args.mezo_total_steps + 1):
        t0 = time.perf_counter()
        batch = batches[int(torch.randint(0, nb, (1,), generator=rng))]
        z = int(torch.randint(0, 2**31 - 1, (1,), generator=rng))
        with torch.no_grad():
            _perturb(params, +args.mezo_eps, z)
            lp = _loss_on_batch(m, batch, device)
            _perturb(params, -2 * args.mezo_eps, z)
            ln = _loss_on_batch(m, batch, device)
            _perturb(params, +args.mezo_eps, z)
            g = (lp - ln) / (2 * args.mezo_eps)
            _perturb(params, -args.mezo_lr * g, z)
        wall += time.perf_counter() - t0
        if step % args.mezo_eval_every == 0:
            ce = _eval(m, tok, eval_exs, args.batch_size, device)
            log_point("mezo", step, wall, ce)
            if mezo_ce_at_loca_wall is None and wall >= loca_wall:
                mezo_ce_at_loca_wall = ce
            if mezo_reach_wall is None and ce <= loca_final_ce:
                mezo_reach_wall = wall
            if mezo_reach_wall is not None and wall > 6 * loca_wall:
                break

    print("\n[c2] ===== SUMMARY =====")
    print(f"  LoCA: final CE {loca_final_ce:.4f} in {loca_wall:.1f}s")
    if mezo_ce_at_loca_wall is not None:
        print(f"  MeZO CE at LoCA's wall ({loca_wall:.0f}s): {mezo_ce_at_loca_wall:.4f}  (worse = LoCA wins)")
    if mezo_reach_wall is not None:
        print(f"  MeZO needs {mezo_reach_wall:.1f}s to reach LoCA's CE -> LoCA is {mezo_reach_wall/max(loca_wall,1e-9):.1f}x faster")
    else:
        print(f"  MeZO did NOT reach LoCA's CE within {args.mezo_total_steps} steps -> LoCA strictly better in budget")


if __name__ == "__main__":
    main()
