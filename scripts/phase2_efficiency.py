"""Phase 2 — CPU efficiency / headline (C3). ONE (model, method) per process so
memory is measured cleanly. Reports base (inference) RAM and the MARGINAL training
overhead separately (base is shared/unavoidable; the marginal is the method cost).

  python scripts/phase2_efficiency.py --model Qwen/Qwen2.5-0.5B --method loca_f \
         --n-train 512 --max-len 256 --threads 96

Metrics per config -> outputs/efficiency.csv:
  wall_per_pass_s : time for ONE data pass of the method's core op
                    (lora: 1 epoch fwd+bwd+step; loca: 1 outer iter fwd+solve;
                     mezo: 1 pass of SPSA steps; frozen: 1 fwd pass)
  loca_sketch_s   : one-time sketch-init backward time (loca only)
  base_ram_mb     : RSS after model load + one forward (inference footprint)
  peak_ram_mb     : process peak RSS over training
  marginal_ram_mb : peak - base  (the training overhead to compare across methods)
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

sys.path.insert(0, ".")
from src.utils.seed import set_seed
from src.utils.logging_csv import ResultRow, append_row, config_hash
from src.data.loaders import load_task, make_collate_fn
from src.loca.runner import _make_batches


def _rss_mb():
    import resource, platform
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / (1024 * 1024) if platform.system() == "Darwin" else v / 1024


def _cur_rss_mb():
    import psutil
    return psutil.Process().memory_info().rss / (1024 * 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", required=True, choices=["frozen", "lora", "mezo", "loca_f"])
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=-1)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--mezo-pass-steps", type=int, default=0, help="0 -> n_train/batch (one pass)")
    ap.add_argument("--csv", default="outputs/efficiency.csv")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    nthreads = torch.get_num_threads()
    set_seed(0)
    device = args.device
    _CUDA = str(device).startswith("cuda")
    def _cur_mem():
        if _CUDA:
            torch.cuda.synchronize(); return torch.cuda.memory_allocated()/(1024*1024)
        return _cur_rss_mb()
    def _peak_mem():
        if _CUDA:
            torch.cuda.synchronize(); return torch.cuda.max_memory_allocated()/(1024*1024)
        return _rss_mb()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to(device)

    # Synthetic fixed-length batches: efficiency is about compute/memory, not quality,
    # so random token ids of length max_len are the standard, dataset-free way to
    # measure wall-clock and RAM. First half of each sequence is "prompt" (masked).
    vocab = int(getattr(model.config, "vocab_size"))
    collate = make_collate_fn(tok.pad_token_id)
    g = torch.Generator().manual_seed(0)
    exs = []
    for _ in range(args.n_train):
        ids = torch.randint(0, vocab, (args.max_len,), generator=g)
        labels = ids.clone()
        labels[: args.max_len // 2] = -100
        exs.append({"input_ids": ids, "labels": labels,
                    "attention_mask": torch.ones(args.max_len, dtype=torch.long)})
    batches = _make_batches(exs, collate, args.batch_size)

    # base footprint: model + one inference forward
    with torch.no_grad():
        b0 = batches[0]
        model(input_ids=b0["input_ids"].to(device), attention_mask=b0["attention_mask"].to(device), use_cache=False)
    base_ram = _cur_mem()
    if _CUDA: torch.cuda.reset_peak_memory_stats()

    sketch_s = 0.0
    t0 = time.perf_counter()
    if args.method == "frozen":
        with torch.no_grad():
            for b in batches:
                model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device), use_cache=False)
    elif args.method == "lora":
        from src.baselines.lora_sft import build_lora_model
        from src.baselines.sft_common import _ce_loss
        pm = build_lora_model(model, {"r": 32, "alpha": 64,
                                      "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}).to(device)
        params = [p for p in pm.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=2e-4)
        for b in batches:                       # one epoch = fwd+bwd+step
            loss = _ce_loss(pm, b, device)
            loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True)
    elif args.method == "loca_f":
        from src.adapters.residual_lora import attach_adapters
        from src.adapters.model_utils import get_handles
        from src.loca.feedback import build_feedback
        from src.loca.solver import LoCASolver, LoCAConfig
        blocks = attach_adapters(model, r=32, seed=0)
        h = get_handles(model)
        probe = batches[0]
        ts = time.perf_counter()
        F = build_feedback(h.hidden_size, len(blocks), kind="sketch", seed=0,
                           model=model, blocks=blocks, probe_batch=probe, rank_sketch=8)
        sketch_s = time.perf_counter() - ts
        solver = LoCASolver(model, blocks, F, LoCAConfig(eta=0.003, lam=0.1, T=1, mode="jacobi"))
        t0 = time.perf_counter()                # reset: time ONE outer iter (fwd+solve), exclude sketch
        solver.outer_step(batches)
    elif args.method == "mezo":
        from src.baselines.mezo import train_mezo, MeZOConfig
        steps = args.mezo_pass_steps or max(1, len(batches))
        train_mezo(model, tok, batches, MeZOConfig(eps=1e-3, lr=1e-6, n_perturb=1,
                                                   steps=steps, log_every=steps), seed=0, device=device)

    wall = time.perf_counter() - t0
    steady_ram = _cur_mem()                       # current mem: forward-only/steady working set
    peak_ram = _peak_mem()                             # process peak (includes one-time sketch backward spike)
    marginal = max(0.0, peak_ram - base_ram)         # peak training overhead
    marginal_steady = max(0.0, steady_ram - base_ram)  # steady-state overhead (LoCA's real forward-only cost)

    print(f"[phase2] {args.method:7s} {args.model}  dtype={args.dtype} threads={nthreads}")
    print(f"  wall_per_pass={wall:.2f}s  sketch_init={sketch_s:.2f}s")
    print(f"  base_ram={base_ram:.0f}MB  peak_ram={peak_ram:.0f}MB  marginal={marginal:.0f}MB  marginal_steady={marginal_steady:.0f}MB")

    chash = config_hash({"model": args.model, "method": args.method, "n_train": args.n_train,
                         "max_len": args.max_len, "threads": nthreads, "dtype": args.dtype})
    common = dict(model=args.model, task="cpu_efficiency", seed=0, config_hash=chash)
    for metric, value in [("wall_per_pass_s", wall), ("loca_sketch_s", sketch_s),
                          ("base_ram_mb", base_ram), ("peak_ram_mb", peak_ram),
                          ("marginal_ram_mb", marginal), ("marginal_steady_ram_mb", marginal_steady)]:
        append_row(args.csv, ResultRow(method=args.method, metric=metric, value=value,
                                       wall_clock_s=wall, peak_mem_mb=peak_ram,
                                       extra={"threads": nthreads, "n_train": args.n_train,
                                              "dtype": args.dtype}, **common))


if __name__ == "__main__":
    main()
