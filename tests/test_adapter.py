"""T2 — ResidualLoRA + AdaptedBlock: B=0 ≡ base, cache shapes, set_B writeback."""
import torch

from src.adapters.residual_lora import ResidualLoRA, AdaptedBlock


class DummyBlock(torch.nn.Module):
    """Stand-in transformer block: returns (linear(x),) like HF decoder layers."""

    def __init__(self, d):
        super().__init__()
        self.lin = torch.nn.Linear(d, d)

    def forward(self, hidden_states, **kwargs):
        return (self.lin(hidden_states),)


def test_B_zero_is_identity_on_base():
    d, r = 16, 4
    ad = ResidualLoRA(d, r, seed=0)
    x = torch.randn(3, 5, d)
    base = torch.randn(3, 5, d)
    out = ad(base, x)
    assert torch.allclose(out, base), "B=0 must give exactly the base output"


def test_adapted_block_b_zero_matches_block():
    d, r = 16, 4
    blk = DummyBlock(d)
    ad = ResidualLoRA(d, r, seed=1)
    ab = AdaptedBlock(blk, ad, layer_idx=0)
    x = torch.randn(2, 7, d)
    base_out = blk(x)[0]
    adapted = ab(x)[0]
    assert torch.allclose(adapted, base_out, atol=1e-6)


def test_set_B_changes_output_additively():
    d, r = 16, 4
    ad = ResidualLoRA(d, r, seed=2)
    x = torch.randn(4, d)
    base = torch.randn(4, d)
    B = torch.randn(d, r)
    ad.set_B(B)
    expected = base + (x @ ad.A.t()) @ B.t()
    assert torch.allclose(ad(base, x), expected, atol=1e-5)


def test_cache_shapes_label_positions():
    d, r = 16, 4
    blk = DummyBlock(d)
    ab = AdaptedBlock(blk, ResidualLoRA(d, r, seed=3), layer_idx=0)
    Bsz, T = 2, 5
    x = torch.randn(Bsz, T, d)
    mask = torch.zeros(Bsz, T, dtype=torch.bool)
    mask[0, 2:] = True   # 3 tokens
    mask[1, 4:] = True   # 1 token
    ab.enable_cache(mask)
    ab(x)
    assert ab.cache_s.shape == (4, d)
    assert ab.cache_h0.shape == (4, d)
    ab.disable_cache()
