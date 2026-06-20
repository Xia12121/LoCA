"""T5 mechanics — offline tiny model: pipeline runs, finite, least-squares sane.

This is the OFFLINE smoke test (random-init model). The SCIENTIFIC Phase-0 E0.3
checks (global CE decreases, alignment angle > 0) require a strong pretrained base
and live in scripts/phase0.py.
"""
import torch

from src.adapters.residual_lora import attach_adapters
from src.adapters.model_utils import get_handles
from src.loca.hooks import cached_forward
from src.loca.feedback import build_random_feedback
from src.loca.solver import LoCASolver, LoCAConfig


def _tiny_model():
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(vocab_size=128, n_positions=64, n_embd=64, n_layer=3, n_head=4)
    torch.manual_seed(0)
    return GPT2LMHeadModel(cfg).eval()


def _batch(B=2, T=12, V=128):
    from src.data.loaders import make_collate_fn
    ids = torch.randint(0, V, (B, T))
    exs = []
    for b in range(B):
        labels = ids[b].clone()
        labels[: T // 2] = -100  # first half is "prompt"
        exs.append({"input_ids": ids[b], "labels": labels,
                    "attention_mask": torch.ones(T, dtype=torch.long)})
    return make_collate_fn(0)(exs)


def test_cached_forward_shapes():
    m = _tiny_model()
    blocks = attach_adapters(m, r=8, seed=0)
    h = get_handles(m)
    b = _batch()
    caches, h_L = cached_forward(m, blocks, b["input_ids"], b["attention_mask"], b["label_mask"])
    n_tok = int(b["label_mask"].sum())
    assert len(caches) == len(blocks)
    assert h_L.shape == (n_tok, h.hidden_size)
    for c in caches:
        assert c.s.shape == (n_tok, h.hidden_size)
        assert c.h0.shape == (n_tok, h.hidden_size)


def test_solver_runs_and_is_finite():
    m = _tiny_model()
    blocks = attach_adapters(m, r=8, seed=0)
    h = get_handles(m)
    F = build_random_feedback(h.hidden_size, len(blocks), seed=0)
    batches = [_batch() for _ in range(3)]
    solver = LoCASolver(m, blocks, F, LoCAConfig(eta=0.02, lam=0.5, T=4, mode="jacobi"))
    hist = solver.fit(batches)
    assert len(hist) == 4
    for mtr in hist:
        assert torch.isfinite(torch.tensor(mtr.global_ce))
        assert mtr.b_norm_max < 1e6


def test_least_squares_reduces_fixed_target_residual():
    """For a fixed target, the closed-form B reduces the regression residual vs B=0."""
    from src.loca.closed_form import GramAccumulator
    torch.manual_seed(0)
    N, r, d, lam = 300, 8, 16, 0.1
    P = torch.randn(N, r)
    Rho = torch.randn(N, d)
    acc = GramAccumulator(d, r)
    acc.add(P, Rho)
    B = acc.solve(lam)
    res0 = (Rho ** 2).sum()                       # B=0 residual
    res1 = ((P @ B.t() - Rho) ** 2).sum()         # solved residual
    assert res1 < res0


def test_gauss_seidel_runs():
    m = _tiny_model()
    blocks = attach_adapters(m, r=8, seed=0)
    h = get_handles(m)
    F = build_random_feedback(h.hidden_size, len(blocks), seed=0)
    batches = [_batch() for _ in range(2)]
    solver = LoCASolver(m, blocks, F, LoCAConfig(eta=0.02, lam=0.5, T=2, mode="gauss_seidel"))
    hist = solver.fit(batches)
    assert len(hist) == 2
