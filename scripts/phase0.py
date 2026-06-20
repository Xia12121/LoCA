"""Phase 0 — mechanism verification on a strong pretrained base (§5, E0.1-E0.3).

Run:  python -m scripts.phase0 --model gpt2 [--feedback random|sketch]

This is the GO/NO-GO gate for the whole idea. It prints a PASS/FAIL per check:
  E0.1  top_layer_error vs autograd            rel err < 1e-4
  E0.2  solve_block   vs numpy ridge           rtol   < 1e-5
  E0.3  LoCA-F on a few SFT examples near base: global CE decreases over outer
        iterations AND mean alignment cos(F e, g) > 0.

E0.3 is the scientific test: it only works in the small-change regime around a
strong frozen base. If it fails, STOP and report (do not proceed to Phase 1).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from src.adapters.residual_lora import attach_adapters
from src.adapters.model_utils import get_handles
from src.data.loaders import Example, tokenize_example, make_collate_fn
from src.loca.closed_form import GramAccumulator
from src.loca.feedback import build_feedback
from src.loca.solver import LoCASolver, LoCAConfig
from src.loca.diagnostics import alignment_angles
from src.loca.top_error import top_layer_error
from src.utils.seed import set_seed

# A few self-contained instruction/response pairs (offline, reproducible).
SAMPLES = [
    ("Translate to French: Hello, how are you?", " Bonjour, comment allez-vous ?"),
    ("Summarize: The cat sat on the mat and slept.", " A cat slept on a mat."),
    ("Question: What is the capital of France?", " The capital of France is Paris."),
    ("Complete: The opposite of hot is", " cold."),
    ("Question: What color is the sky on a clear day?", " The sky is blue."),
    ("Write a polite greeting.", " Good morning! I hope you have a wonderful day."),
    ("Question: How many days are in a week?", " There are seven days in a week."),
    ("Complete: Water freezes at zero degrees", " Celsius."),
]


def _green(s):
    return f"\033[92m{s}\033[0m"


def _red(s):
    return f"\033[91m{s}\033[0m"


def status(ok: bool) -> str:
    return _green("PASS") if ok else _red("FAIL")


def e01_top_error(model, tok, device) -> bool:
    h = get_handles(model)
    ex = tokenize_example(Example(*SAMPLES[0]), tok, max_len=64)
    collate = make_collate_fn(tok.pad_token_id)
    batch = collate([ex])
    ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    pmask = batch["predict_mask"].to(device)
    targets = batch["targets"].to(device)
    labels = batch["labels"].to(device)
    embeds = h.embed_tokens(ids).detach().requires_grad_(True)
    with torch.enable_grad():
        out = model(inputs_embeds=embeds, attention_mask=attn, output_hidden_states=True, use_cache=False)
        hL_full = out.hidden_states[-1]
        hL_full.retain_grad()
        loss = torch.nn.functional.cross_entropy(
            out.logits[:, :-1, :].reshape(-1, out.logits.shape[-1]), labels[:, 1:].reshape(-1),
            ignore_index=-100, reduction="sum")
        loss.backward()
    g = hL_full.grad[pmask].double()
    e = top_layer_error(hL_full.detach()[pmask], targets[pmask], h.lm_head, reduction="sum").double()
    rel = (e - g).norm() / g.norm()
    ok = rel < 1e-4
    print(f"  E0.1 top_layer_error vs autograd : rel_err={rel:.2e}  {status(ok)}")
    model.zero_grad(set_to_none=True)
    return ok


def e02_closed_form() -> bool:
    torch.manual_seed(0)
    N, r, d, lam = 300, 16, 32, 0.2
    P = torch.randn(N, r, dtype=torch.float64)
    Rho = torch.randn(N, d, dtype=torch.float64)
    acc = GramAccumulator(d, r)
    acc.add(P, Rho)
    B = acc.solve(lam).double().numpy()
    G = (P.t() @ P).numpy()
    C = (Rho.t() @ P).numpy()
    B_ref = C @ np.linalg.inv(G + lam * np.eye(r))
    rel = np.abs(B - B_ref).max()
    ok = rel < 1e-5
    print(f"  E0.2 solve_block vs numpy ridge   : max_abs={rel:.2e}  {status(ok)}")
    return ok


def e03_descent(model, tok, device, feedback_kind: str, eta: float, T: int) -> bool:
    h = get_handles(model)
    blocks = attach_adapters(model, r=32, seed=0, dtype=torch.float32)
    collate = make_collate_fn(tok.pad_token_id)
    exs = [tokenize_example(Example(p, c), tok, max_len=64) for p, c in SAMPLES]
    batch = collate(exs)
    probe = {k: v.to(device) for k, v in batch.items()}

    F = build_feedback(h.hidden_size, len(blocks), kind=feedback_kind, seed=0,
                       model=model, blocks=blocks, probe_batch=probe)

    # baseline alignment before training
    align0 = alignment_angles(model, blocks, F, probe)
    solver = LoCASolver(model, blocks, F, LoCAConfig(eta=eta, lam=0.1, T=T, mode="jacobi"))
    hist = solver.fit([probe])
    align1 = alignment_angles(model, blocks, F, probe)

    ce = [m.global_ce for m in hist]
    ce_drop = ce[0] - ce[-1]
    mean_align0 = float(np.mean(align0))
    mean_align1 = float(np.mean(align1))
    monotone = all(ce[i + 1] <= ce[i] + 1e-3 for i in range(len(ce) - 1))

    print(f"  E0.3 LoCA-F descent (feedback={feedback_kind}, eta={eta}, T={T}):")
    print(f"        global CE per outer iter : {[round(c, 4) for c in ce]}")
    print(f"        CE drop                  : {ce_drop:+.4f}  (want > 0)")
    print(f"        mean align cos(Fe,g)     : init={mean_align0:+.3f} -> final={mean_align1:+.3f}  (want > 0)")
    print(f"        CE monotone non-increase : {monotone}")
    ok = (ce_drop > 0) and (mean_align1 > 0)
    print(f"        E0.3 verdict             : {status(ok)}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--feedback", default="random", choices=["random", "sketch"])
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--T", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    set_seed(0)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"[phase0] loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device).eval()

    print("[phase0] running checks:")
    ok1 = e01_top_error(model, tok, args.device)
    ok2 = e02_closed_form()
    ok3 = e03_descent(model, tok, args.device, args.feedback, args.eta, args.T)

    print("\n[phase0] SUMMARY")
    allok = ok1 and ok2 and ok3
    print(f"  E0.1={status(ok1)}  E0.2={status(ok2)}  E0.3={status(ok3)}")
    if allok:
        print(_green("  PHASE 0 PASSED — cleared to proceed to Phase 1."))
    else:
        print(_red("  PHASE 0 FAILED — STOP. Fix before Phase 1 (see §8 kill criteria)."))
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
