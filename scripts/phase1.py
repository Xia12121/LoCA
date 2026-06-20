"""Phase 1 — quality / recovery on one (model, task) across all methods (C1/C2).

  python scripts/phase1.py --model Qwen/Qwen2.5-0.5B --task sst2 \
         --methods frozen lora mezo loca_f --seeds 0 --device cuda

Loads tokenizer + shared train/eval splits once (fairness), trains each method on
a FRESH model, evaluates held-out CE, writes rows to outputs/results.csv, and
prints a recovery table:  recovery = (frozen_ce - method_ce)/(frozen_ce - lora_ce).
Gate: LoCA-F recovery >= 0.85 and > MeZO.
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
from src.loca.runner import run_loca, _resolve_device, _make_batches
from src.data.loaders import load_task, make_collate_fn
from src.eval.perplexity import eval_perplexity

DEFAULT_LOCA = dict(variant="F", feedback="sketch", sketch_rank=8,
                    eta=0.05, lam=0.1, T=5, mode="jacobi")


def _load_model(name, dtype, device):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)


def _completed_cells(csv_path):
    """Map (method, model, task, seed) -> ce for rows already in the CSV (resume)."""
    import csv as _csv
    import os
    done = {}
    if not os.path.exists(csv_path):
        return done
    with open(csv_path) as f:
        for r in _csv.DictReader(f):
            if r.get("metric") == "ce":
                done[(r["method"], r["model"], r["task"], r["seed"])] = float(r["value"])
    return done


def _peak_reset(device):
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def _peak_read(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    from src.utils.profiling import _rss_mb
    return _rss_mb()


def _train_loca(method, model_name, tok, cfg, seed, device, dtype, es_exs, train_batches):
    """Run the eta auto-select. Keeps only ONE model on GPU at a time (parks the
    current winner on CPU) so the measured training peak is a single model's, not
    two. Returns (model_on_device, sel_eta, sel_t, killed)."""
    etas = cfg.get("loca_etas", [cfg.get("loca_override", {}).get("eta", 0.003)])
    best_score, best_model, best_eta, best_t = float("inf"), None, None, -1
    for eta in etas:
        if best_model is not None and device.startswith("cuda"):
            best_model.to("cpu")                       # park winner -> free VRAM
            torch.cuda.empty_cache()
        m = _load_model(model_name, dtype, device).eval()
        loca = dict(DEFAULT_LOCA)
        loca["variant"] = "F" if method == "loca_f" else "D"
        loca.update(cfg.get("loca_override", {}))
        loca["eta"] = eta
        run_cfg = {"model": {"name": model_name, "dtype": cfg["dtype"], "device": device},
                   "adapter": {"r": cfg.get("adapter_r", 32), "s_source": "block_input"},
                   "loca": loca, "train": cfg["train"], "runtime": {"cpu_threads": -1}}
        eval_fn = lambda mdl=m: eval_perplexity(mdl, tok, es_exs,
                                               batch_size=cfg["train"]["batch_size"], device=device)["ce"]
        res = run_loca(m, tok, run_cfg, seed, eval_fn=eval_fn)
        vscore = res.best_score
        print(f"           (loca eta={eta} val={vscore:.4f} best_t={res.best_t} killed={res.killed})")
        if vscore < best_score:
            if best_model is not None:
                del best_model                          # free old winner (on CPU)
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            best_score, best_model, best_eta, best_t = vscore, m, eta, res.best_t
        else:
            del m
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    if device.startswith("cuda"):
        best_model.to(device)
    print(f"           (loca SELECTED eta={best_eta} best_t={best_t} val={best_score:.4f})")
    return best_model, best_eta, best_t, None


def train_and_eval(method, model_name, tok, train_batches, eval_exs, cfg, seed, device, dtype, val_exs=None):
    """Return (ce, train_wall_s, train_mem_mb, killed).

    Memory and wall-clock are measured over TRAINING ONLY (eval excluded), because
    the eval logits tensor is identical across methods and would mask the training-
    memory difference that C3 is about. val_exs is used only for LoCA early-stop.
    For LoCA, the eta auto-select keeps a single model on GPU (see _train_loca), so
    the peak is one model's training peak, not two.
    """
    import time
    killed = None
    _peak_reset(device)
    t0 = time.perf_counter()
    if method == "frozen":
        model = _load_model(model_name, dtype, device).eval()
    elif method in ("loca_f", "loca_d"):
        es_exs = val_exs if val_exs else eval_exs[: min(len(eval_exs), 200)]
        model, _, _, killed = _train_loca(method, model_name, tok, cfg, seed, device, dtype, es_exs, train_batches)
    elif method == "lora":
        from src.baselines.lora_sft import train_lora
        model = _load_model(model_name, dtype, device)
        model, _ = train_lora(model, train_batches, cfg["lora"], device=device)
    elif method == "mezo":
        from src.baselines.mezo import train_mezo, MeZOConfig
        model = _load_model(model_name, dtype, device).eval()
        mc = cfg["mezo"]
        model, _, _ = train_mezo(model, tok, train_batches,
                                 MeZOConfig(eps=mc["eps"], lr=mc["lr"],
                                            n_perturb=mc["n_perturb"], steps=mc["steps"],
                                            log_every=max(1, mc["steps"] // 10)),
                                 seed=seed, device=device)
    elif method == "full_sft":
        from src.baselines.full_sft import train_full_sft
        model = _load_model(model_name, dtype, device)
        model, _ = train_full_sft(model, train_batches, cfg["full_sft"], device=device)
    else:
        raise ValueError(method)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    train_wall = time.perf_counter() - t0
    train_mem = _peak_read(device)                      # training-only peak

    ppl = eval_perplexity(model, tok, eval_exs, batch_size=cfg["train"]["batch_size"], device=device)
    _acc = None
    try:
        from src.eval.rank_acc import eval_rank_accuracy
        _has = eval_exs and (eval_exs[0].get("choices") if isinstance(eval_exs[0], dict) else getattr(eval_exs[0], "choices", None))
        if _has:
            _r = eval_rank_accuracy(model, tok, eval_exs, device=device)
            _acc = _r["acc_norm"] if _r else None
    except Exception as _e:
        print(f"[phase1] acc eval skipped: {_e}")
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return ppl["ce"], train_wall, train_mem, killed, _acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--task", default="sst2")
    ap.add_argument("--methods", nargs="+", default=["frozen", "lora", "mezo", "loca_f"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=0, help="CPU intra-op threads (0=leave default)")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--mezo-steps", type=int, default=10000)
    ap.add_argument("--eta", type=float, default=0.003)
    ap.add_argument("--loca-etas", nargs="+", type=float, default=None,
                    help="candidate etas; LoCA keeps the one with best val CE")
    ap.add_argument("--T", type=int, default=15)
    ap.add_argument("--feedback", default="sketch")
    ap.add_argument("--mode", default="jacobi", choices=["jacobi", "gauss_seidel"])
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--loca-target-norm", default="none", choices=["none", "rms"],
                    help="rms = scale-invariant relative-step target (one eta across sizes)")
    ap.add_argument("--task-name", default=None, help="CSV task label (defaults to --task)")
    ap.add_argument("--lora-lr", type=float, default=2e-4)
    ap.add_argument("--lora-epochs", type=int, default=3)
    ap.add_argument("--csv", default="outputs/results.csv")
    args = ap.parse_args()

    device = _resolve_device(args.device)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    collate = make_collate_fn(tok.pad_token_id)
    train_exs = load_task(args.task, tok, n=args.n_train, max_len=args.max_len, split="train")
    full_eval = load_task(args.task, tok, n=args.n_eval, max_len=args.max_len, split="eval")
    half = max(1, len(full_eval) // 2)
    val_exs, eval_exs = full_eval[:half], full_eval[half:]   # val=early-stop, test=report
    train_batches = _make_batches(train_exs, collate, args.batch_size)
    print(f"[phase1] model={args.model} task={args.task} train={len(train_exs)} val={len(val_exs)} test={len(eval_exs)} device={device}")

    cfg = {
        "dtype": args.dtype,
        "train": {"dataset": args.task, "n_train": args.n_train, "n_eval": args.n_eval,
                  "max_len": args.max_len, "batch_size": args.batch_size},
        "lora": {"r": 32, "alpha": 64, "dropout": 0.0,
                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                 "lr": args.lora_lr, "epochs": args.lora_epochs},
        "mezo": {"eps": 1e-3, "lr": 1e-6, "n_perturb": 1, "steps": args.mezo_steps},
        "full_sft": {"lr": 1e-5, "epochs": 3},
        "loca_override": {"eta": args.eta, "T": args.T, "feedback": args.feedback,
                          "mode": args.mode, "lam": args.lam,
                          "target_norm": args.loca_target_norm},
        "loca_etas": args.loca_etas or [args.eta],
        "adapter_r": args.r,
    }

    task_label = args.task_name or args.task   # CSV label (data still loaded from --task)
    done = _completed_cells(args.csv)   # resume: skip cells already in the CSV
    results = {}  # method -> list of ce across seeds
    for seed in args.seeds:
        set_seed(seed)
        for method in args.methods:
            key = (method, args.model, task_label, str(seed))
            if key in done:
                prev = done[key]
                results.setdefault(method, []).append(prev)
                print(f"[phase1] {method:9s} seed={seed} SKIP (already done, ce={prev:.4f})")
                continue
            ce, wall, mem, killed, acc = train_and_eval(method, args.model, tok, train_batches,
                                                   eval_exs, cfg, seed, device, dtype, val_exs=val_exs)
            results.setdefault(method, []).append(ce)
            _astr = f" acc={acc:.4f}" if acc is not None else ""
            print(f"[phase1] {method:9s} seed={seed} ce={ce:.4f}{_astr} wall={wall:.1f}s mem={mem:.0f}MB killed={killed}")
            append_row(args.csv, ResultRow(
                method=method, model=args.model, task=task_label, seed=seed,
                metric="ce", value=ce, wall_clock_s=wall, peak_mem_mb=mem,
                config_hash=config_hash({**cfg, "method": method, "model": args.model}),
                extra={"killed": killed, "feedback": args.feedback, "eta": args.eta, "T": args.T,
                       "mezo_steps": args.mezo_steps}))
            if acc is not None:
                append_row(args.csv, ResultRow(
                    method=method, model=args.model, task=task_label, seed=seed,
                    metric="acc", value=acc, wall_clock_s=wall, peak_mem_mb=mem,
                    config_hash=config_hash({**cfg, "method": method, "model": args.model}),
                    extra={"metric_kind": "acc_norm", "feedback": args.feedback, "eta": args.eta, "T": args.T}))

    # recovery table
    import statistics as st
    def mean(m):
        return st.mean(results[m]) if m in results and results[m] else None
    fr, lo = mean("frozen"), mean("lora")
    print("\n[phase1] ==== RECOVERY (CE, lower better) ====")
    print(f"  frozen CE = {fr}   lora CE = {lo}")
    if fr is not None and lo is not None and abs(fr - lo) > 1e-9:
        for m in args.methods:
            mv = mean(m)
            if mv is None:
                continue
            rec = (fr - mv) / (fr - lo)
            tag = ""
            if m == "loca_f":
                tag = "  <-- GATE: need >=0.85 and > mezo"
            print(f"  {m:9s} CE={mv:.4f}  recovery={rec:+.3f}{tag}")
    else:
        print("  need frozen + lora to compute recovery")


if __name__ == "__main__":
    main()
