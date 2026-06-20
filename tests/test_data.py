"""T1 — data loaders: tokenization, prompt masking, collation."""
import pytest
import torch

from src.data.loaders import load_task, make_collate_fn, tokenize_example, Example


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    except Exception as e:  # offline / no cache
        pytest.skip(f"tokenizer unavailable: {e}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def test_prompt_is_masked(tokenizer):
    ex = Example(prompt="Question: hi\n", completion="hello world")
    out = tokenize_example(ex, tokenizer, max_len=64)
    labels = out["labels"]
    # at least one masked (prompt) and one unmasked (completion) position
    assert (labels == -100).any()
    assert (labels != -100).any()
    # unmasked labels equal the corresponding input_ids
    keep = labels != -100
    assert torch.equal(labels[keep], out["input_ids"][keep])


def test_format_json_offline():
    # format_json needs no network
    class W:  # minimal tokenizer stub mapping char->id
        bos_token_id = 1
        eos_token_id = 2
        def __call__(self, text, add_special_tokens=True):
            return {"input_ids": [ (ord(c) % 100) + 3 for c in text ]}
    exs = load_task("format_json", W(), n=8, max_len=128, split="train")
    assert len(exs) == 8
    for e in exs:
        assert e["input_ids"].ndim == 1
        assert e["labels"].shape == e["input_ids"].shape
        assert (e["labels"] != -100).any()


def test_collate_pads_and_masks(tokenizer):
    exs = load_task("format_json", tokenizer, n=6, max_len=128, split="train")
    collate = make_collate_fn(tokenizer.pad_token_id)
    batch = collate(exs)
    B = len(exs)
    assert batch["input_ids"].shape[0] == B
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert batch["label_mask"].dtype == torch.bool
    # label_mask matches labels != -100
    assert torch.equal(batch["label_mask"], batch["labels"].ne(-100))
