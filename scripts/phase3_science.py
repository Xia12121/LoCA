"""Phase 3 — science curves (C4): why LoCA works and where it breaks (§5).

Sub-experiments (select with --exp):
  align   E3.1  alignment angle cos(F e, g) vs outer iteration, random vs sketch
  eta     E3.2  held-out CE vs eta (small eta good, large eta breaks)
  lingap  --    first-order predicted dL vs actual dL, gap ~ O(eta^2)

Writes to outputs/results.csv (metric names prefixed sci_*) for make_figures.
Run on the GPU box for the big models; gpt2 / pythia work on CPU for a preview.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from src.utils.config import load_config, apply_overrides
from src.utils.logging_csv import ResultRow, append_row, config_hash
from src.utils.seed import set_seed
from src.adapters.residual_lora import attach_adapters
from src.adapters.model_utils import get_handles
from src.data.loaders import load_task, make_collate_fn
from src.loca.feedback import build_feedback
from src.loca.solver import LoCASolver, LoCAConfig
from src.loca.diagnostics import alignment_angles, linearization_gap
from src.loca.runner import _resolve_device, _make_batches
from src.eval.perplexity import eval_perplexity


def _prep(cfg, device):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = getattr(torch, cfg["model"].get("dtype", "float32"))
    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"], dtype=dtype).to(device).eval()
    return tok, model


def exp_align(cfg, device):
    """E3.1: alignment cos(F e, g) vs outer iter, for random and sketch feedback."""
    tok, _ = _prep(cfg, device)
    collate = make_collate_fn(tok.pad_token_id)
    exs = load_task(cfg["train"]["dataset"], tok, n=cfg["train"]["n_train"],
                    max_len=cfg["train"]["max_len"], split="train")
    batches = _make_batches(exs, collate, cfg["train"]["batch_size"])
    diag = {k: v.to(device) for k, v in batches[0].items()}

    for kind in ["random", "sketch"]:
        _, model = _prep(cfg, device)
        blocks = attach_adapters(model, r=cfg["adapter"]["r"], seed=0)
        h = get_handles(model)
        probe = diag if kind == "sketch" else None
        F = build_feedback(h.hidden_size, len(blocks), kind=kind, seed=0,
                           model=model, blocks=blocks, probe_batch=probe,
                           rank_sketch=cfg["loca"].get("sketch_rank", 8))
        solver = LoCASolver(model, blocks, F, LoCAConfig(
            eta=cfg["loca"]["eta"], lam=cfg["loca"]["lam"], T=cfg["loca"]["T"],
            mode=cfg["loca"].get("mode", "jacobi")))

        chash = config_hash({**cfg, "feedback": kind})

        def on_iter(t, m, s, kind=kind, chash=chash):
            cos = float(np.mean(alignment_angles(model, blocks, F, diag)))
            print(f"[align kind={kind}] t={t} mean_cos={cos:+.3f} ce={m.global_ce:.4f}")
            append_row(cfg["log"]["csv"], ResultRow(
                method=f"loca_f_{kind}", model=cfg["model"]["name"],
                task=cfg["train"]["dataset"], seed=0, metric="sci_align_cos",
                value=cos, config_hash=chash, extra={"t": t, "feedback": kind, "ce": m.global_ce}))

        solver.fit(batches, on_iter=on_iter)


def exp_eta(cfg, device):
    """E3.2: held-out CE vs eta. Expect a U / cliff — small good, large breaks."""
    tok, _ = _prep(cfg, device)
    collate = make_collate_fn(tok.pad_token_id)
    exs = load_task(cfg["train"]["dataset"], tok, n=cfg["train"]["n_train"],
                    max_len=cfg["train"]["max_len"], split="train")
    eval_exs = load_task(cfg["train"]["dataset"], tok, n=cfg["train"].get("n_eval", 200),
                         max_len=cfg["train"]["max_len"], split="eval")
    batches = _make_batches(exs, collate, cfg["train"]["batch_size"])
    diag = {k: v.to(device) for k, v in batches[0].items()}

    for eta in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        _, model = _prep(cfg, device)
        blocks = attach_adapters(model, r=cfg["adapter"]["r"], seed=0)
        h = get_handles(model)
        F = build_feedback(h.hidden_size, len(blocks), kind=cfg["loca"].get("feedback", "sketch"),
                           seed=0, model=model, blocks=blocks, probe_batch=diag,
                           rank_sketch=cfg["loca"].get("sketch_rank", 8))
        solver = LoCASolver(model, blocks, F, LoCAConfig(
            eta=eta, lam=cfg["loca"]["lam"], T=cfg["loca"]["T"], mode=cfg["loca"].get("mode", "jacobi")))
        killed = None
        try:
            solver.fit(batches)
        except Exception as e:
            killed = str(e)
        ppl = eval_perplexity(model, tok, eval_exs, batch_size=cfg["train"]["batch_size"], device=device)
        print(f"[eta sweep] eta={eta} ce={ppl['ce']:.4f} killed={killed}")
        append_row(cfg["log"]["csv"], ResultRow(
            method="loca_f", model=cfg["model"]["name"], task=cfg["train"]["dataset"],
            seed=0, metric="sci_eta_ce", value=ppl["ce"],
            config_hash=config_hash({**cfg, "eta": eta}),
            extra={"eta": eta, "killed": killed}))


def exp_lingap(cfg, device):
    """First-order predicted dL vs actual dL after a single -eta F e move."""
    tok, model = _prep(cfg, device)
    collate = make_collate_fn(tok.pad_token_id)
    exs = load_task(cfg["train"]["dataset"], tok, n=cfg["train"]["n_train"],
                    max_len=cfg["train"]["max_len"], split="train")
    batches = _make_batches(exs, collate, cfg["train"]["batch_size"])
    diag = {k: v.to(device) for k, v in batches[0].items()}
    for eta in [0.005, 0.01, 0.02, 0.05, 0.1]:
        _, model = _prep(cfg, device)
        blocks = attach_adapters(model, r=cfg["adapter"]["r"], seed=0)
        h = get_handles(model)
        F = build_feedback(h.hidden_size, len(blocks), kind="sketch", seed=0,
                           model=model, blocks=blocks, probe_batch=diag,
                           rank_sketch=cfg["loca"].get("sketch_rank", 8))
        info = linearization_gap(model, blocks, F, diag, eta)
        print(f"[lingap] eta={eta} pred_dL={info['pred_dL']:.5f}")
        append_row(cfg["log"]["csv"], ResultRow(
            method="loca_f", model=cfg["model"]["name"], task=cfg["train"]["dataset"],
            seed=0, metric="sci_pred_dL", value=info["pred_dL"],
            config_hash=config_hash({**cfg, "eta": eta, "exp": "lingap"}), extra={"eta": eta}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/loca_f.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--exp", choices=["align", "eta", "lingap"], required=True)
    args = ap.parse_args()
    set_seed(0)
    cfg = apply_overrides(load_config(args.config), args.override)
    device = _resolve_device(cfg["model"].get("device", "auto"))
    {"align": exp_align, "eta": exp_eta, "lingap": exp_lingap}[args.exp](cfg, device)


if __name__ == "__main__":
    main()
