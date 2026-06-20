"""E0.1 — top_layer_error vs autograd (acceptance: rtol < 1e-4)."""
import torch

from src.loca.top_error import top_layer_error


def _reference_grad(h_L, labels, lm_head):
    h = h_L.clone().requires_grad_(True)
    z = lm_head(h)
    loss = torch.nn.functional.cross_entropy(z, labels, reduction="sum")
    (g,) = torch.autograd.grad(loss, h)
    return g


def test_matches_autograd_no_bias():
    torch.manual_seed(0)
    n_tok, d, V = 50, 32, 100
    h_L = torch.randn(n_tok, d, dtype=torch.float64)
    labels = torch.randint(0, V, (n_tok,))
    lm_head = torch.nn.Linear(d, V, bias=False).double()
    e = top_layer_error(h_L, labels, lm_head, reduction="sum").double()
    g = _reference_grad(h_L, labels, lm_head)
    assert torch.allclose(e, g, rtol=1e-4, atol=1e-6), (e - g).abs().max()


def test_matches_autograd_with_bias_and_chunk():
    torch.manual_seed(1)
    n_tok, d, V = 40, 24, 257
    h_L = torch.randn(n_tok, d, dtype=torch.float64)
    labels = torch.randint(0, V, (n_tok,))
    lm_head = torch.nn.Linear(d, V, bias=True).double()
    e = top_layer_error(h_L, labels, lm_head, reduction="sum", vocab_chunk=64).double()
    g = _reference_grad(h_L, labels, lm_head)
    assert torch.allclose(e, g, rtol=1e-4, atol=1e-6), (e - g).abs().max()


def test_mean_reduction_scales():
    torch.manual_seed(2)
    n_tok, d, V = 20, 16, 50
    h_L = torch.randn(n_tok, d, dtype=torch.float64)
    labels = torch.randint(0, V, (n_tok,))
    lm_head = torch.nn.Linear(d, V, bias=False).double()
    e_sum = top_layer_error(h_L, labels, lm_head, reduction="sum")
    e_mean = top_layer_error(h_L, labels, lm_head, reduction="mean")
    assert torch.allclose(e_sum / n_tok, e_mean, rtol=1e-6)
