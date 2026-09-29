from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, ModernBertConfig, ModernBertModel

from jeff.encoder import ENCODER_MODELS, EncoderDecisionModel
from jeff.models import is_encoder, load_decision_model

BASE = "answerdotai/ModernBERT-base"
REVISION = ENCODER_MODELS[BASE]
ROWS = [
    {"state": "The parcel never arrived.", "question": {"type": "choice", "instructions": "Route the message.",
                                                        "criteria": {"billing": "Charges.", "delivery": "Shipping.", "other": None}}},
    {"state": "The sky is green.", "question": {"type": "noul", "instructions": "Is the statement true?"}},
]


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny random ModernBERT with the real tokenizer, saved like a downloaded base model (no GPU, seconds on CPU)."""
    path = tmp_path_factory.mktemp("tiny-modernbert")
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
    config = ModernBertConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                              num_attention_heads=2, pad_token_id=tokenizer.pad_token_id)
    torch.manual_seed(0)
    ModernBertModel(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def model(base: Path, train: bool = False) -> EncoderDecisionModel:
    return EncoderDecisionModel(base_model=str(base), revision=REVISION, device="cpu", train=train)


def test_one_logit_per_option_in_the_decoder_layout(tiny_base: Path) -> None:
    m = model(tiny_base)
    batch = m.prepare(ROWS)
    assert batch.counts == (3, 2)
    logits = m(batch)
    assert logits.shape == (2, 255)
    assert torch.isfinite(logits[0, :3]).all() and torch.isfinite(logits[1, :2]).all()
    assert (logits[0, 3:] == -1e9).all() and (logits[1, 2:] == -1e9).all()
    probabilities = m.predict(ROWS)
    assert [len(p) for p in probabilities] == [3, 2] and all(abs(sum(p) - 1) < 1e-5 for p in probabilities)


def test_gradients_reach_backbone_and_scorer(tiny_base: Path) -> None:
    m = model(tiny_base, train=True)
    logits = m(m.prepare(ROWS))
    loss = -torch.log_softmax(logits[:, :2], dim=-1)[:, 0].mean()
    loss.backward()
    assert m.scorer.weight.grad is not None and m.scorer.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.backbone.parameters())


def test_save_load_round_trip_through_the_factory(tiny_base: Path, tmp_path: Path) -> None:
    m = model(tiny_base)
    before = m.predict(ROWS, temperature=1.0)
    m.save(tmp_path / "ckpt", temperature=1.7, step=3)
    assert is_encoder(tmp_path / "ckpt", None)
    loaded = load_decision_model(checkpoint=tmp_path / "ckpt", device="cpu")
    assert isinstance(loaded, EncoderDecisionModel) and loaded.temperature == 1.7
    after = loaded.predict(ROWS, temperature=1.0)
    assert all(abs(a - b) < 1e-5 for pa, pb in zip(before, after) for a, b in zip(pa, pb))


def test_factory_routes_by_base_model_name() -> None:
    assert is_encoder(None, "answerdotai/ModernBERT-large")
    assert not is_encoder(None, "Qwen/Qwen3.5-0.8B")


def test_rejects_decoder_checkpoints_and_missing_revision(tmp_path: Path) -> None:
    (tmp_path / "decision_config.json").write_text('{"format_version": 1, "base_model": "Qwen/Qwen3.5-0.8B"}')
    with pytest.raises(ValueError, match="not an encoder"):
        EncoderDecisionModel(checkpoint=tmp_path, device="cpu")
    with pytest.raises(ValueError, match="base_model and revision"):
        EncoderDecisionModel(device="cpu")
