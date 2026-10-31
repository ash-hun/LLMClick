"""Fixtures that need no download or GPU: no network, a tiny random LM, and the fakes and runs several files share."""

import sys
import socket
from typing import Any
from pathlib import Path

import pytest

from tests.speech_fakes import FakeEmbedder, FakeListener, FakePiper, FakeRater, FakeSpeaker, fake_feature_models
from modeling.llm.models.transformer import TransformerBackbone
from modeling.llm.models.config import LLMBackboneConfig
from tests.recipe_fakes import FakeHarness, FakeWandb
from tests.recipe_inputs import raw
from core import pipeline

CHATML = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test that reaches a real API is slow, costs money and follows the other side's state: external calls fail."""
    connect = socket.socket.connect

    def guarded(self: socket.socket, address: Any) -> None:
        if self.family in (socket.AF_INET, socket.AF_INET6) and address[0] not in LOCAL_HOSTS:
            raise RuntimeError(f"a test tried to reach {address[0]}: mock the client instead")
        connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 2-layer GPT-2 with random weights and a word-level tokenizer over the sample rows, saved like a Hub model."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    import torch

    special = ["<unk>", "<pad>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|fim_prefix|>", "<|fim_middle|>",
               "<|fim_suffix|>", "<|box_start|>", "<|box_end|>"]
    tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    corpus = [path.read_text() for path in sorted(Path("samples").glob("*.jsonl"))]
    corpus += ["system user assistant Context Question Options no yes A B C D E F"]
    tokenizer.train_from_iterator(corpus, trainers.WordLevelTrainer(special_tokens=special))
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", pad_token="<pad>",
                                   eos_token="<|im_end|>", chat_template=CHATML)
    torch.manual_seed(0)
    # no dropout, like the Qwen models: training-mode and evaluation-mode outputs agree, so tests can compare them
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(fast), n_positions=256, n_embd=32, n_layer=2, n_head=2,
                                       resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
                                       bos_token_id=fast.eos_token_id, eos_token_id=fast.eos_token_id))
    path = tmp_path_factory.mktemp("tiny-model")
    model.save_pretrained(path)
    fast.save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def trained(tiny_model: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A tiny `llm_sft` experiment the evaluation recipes score, trained once per test file."""
    out = tmp_path_factory.mktemp("trained")
    result = pipeline.build(raw("llm_sft", tiny_model, out)).run()
    return result


@pytest.fixture
def backbone(tiny_model: Path) -> TransformerBackbone:
    """The tiny model loaded as a transformer backbone on the CPU."""
    loaded = TransformerBackbone(LLMBackboneConfig(architecture="transformer", name=str(tiny_model)))
    loaded.load("cpu")
    return loaded


@pytest.fixture
def wandb(monkeypatch: pytest.MonkeyPatch) -> FakeWandb:
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeHarness:
    """lm_eval's `simple_evaluate` replaced by a FakeHarness."""
    import lm_eval

    harness = FakeHarness()
    monkeypatch.setattr(lm_eval, "simple_evaluate", harness.simple_evaluate)
    return harness


@pytest.fixture
def piper(monkeypatch: pytest.MonkeyPatch) -> FakePiper:
    """The Piper program, Whisper and the quality and speaker raters of the tts recipes replaced by fakes;
    `data_speech` patches other places and keeps its own `piper` fixture."""
    fake = FakePiper()
    for name in ("speak", "fit", "export"):
        monkeypatch.setattr(f"modeling.tts.piper.{name}", getattr(fake, name))
    monkeypatch.setattr("modeling.tts.piper.version", lambda: "test")
    monkeypatch.setattr("modeling.tts.stages.Listener", FakeListener(fake, heard_as="a recording"))
    monkeypatch.setattr("modeling.tts.stages.Rater", lambda directory: FakeRater(fake))
    monkeypatch.setattr("modeling.tts.stages.Speaker", lambda recordings: FakeSpeaker(fake))
    return fake


@pytest.fixture
def wakeword_front_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """openWakeWord's feature models replaced by FakeEmbedder; the wakeword files use it on every test."""
    monkeypatch.setattr("modeling.wakeword.features.ensure", fake_feature_models)
    monkeypatch.setattr("modeling.wakeword.features.Embedder", FakeEmbedder)
