"""T6 — baselines run a few steps without error; frozen == base."""
import torch

from src.baselines.mezo import train_mezo, MeZOConfig
from src.baselines.full_sft import train_full_sft
from src.baselines.frozen import train_frozen


def _tiny_model():
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(vocab_size=128, n_positions=64, n_embd=64, n_layer=2, n_head=4)
    torch.manual_seed(0)
    return GPT2LMHeadModel(cfg).eval()


def _batches(n=2, B=2, T=12, V=128):
    out = []
    for _ in range(n):
        ids = torch.randint(0, V, (B, T))
        labels = ids.clone()
        labels[:, : T // 2] = -100
        out.append({"input_ids": ids, "attention_mask": torch.ones(B, T, dtype=torch.long),
                    "labels": labels, "label_mask": labels.ne(-100)})
    return out


def test_frozen_is_base():
    m = _tiny_model()
    ref = {k: v.clone() for k, v in m.state_dict().items()}
    m2, hist = train_frozen(m)
    for k, v in m2.state_dict().items():
        assert torch.equal(v, ref[k])
    assert hist == []


def test_mezo_runs():
    m = _tiny_model()
    m, blocks, hist = train_mezo(m, None, _batches(), MeZOConfig(steps=10, log_every=5), seed=0)
    assert len(hist) >= 1
    assert all(torch.isfinite(b.adapter.B).all() for b in blocks)


def test_full_sft_runs():
    m = _tiny_model()
    m, hist = train_full_sft(m, _batches(), {"lr": 1e-4, "epochs": 1}, device="cpu")
    assert len(hist) >= 1
