"""Fixtures that need no download or GPU: Example JSONL for jev, and a tiny random LM for the tuning recipes."""

import json
from pathlib import Path

import pytest

CRITERIA = {"billing": "Charges or refunds.", "technical": "Something is broken.", "account": "Login or profile."}
LABELS = list(CRITERIA)


def example(prefix: str, index: int) -> dict:
    label = LABELS[index % len(LABELS)]
    return {"id": f"{prefix}-{index:04d}", "suite": prefix, "family": f"{prefix}-fam-{index}",
            "state": f"Customer message {prefix} {index}: my {label} problem needs attention today, please route it.",
            "question": {"type": "choice", "instructions": "Route the request to one department.", "criteria": CRITERIA},
            "label": label, "target": label, "source": {"dataset": "fixture"}}


def write(path: Path, prefix: str, count: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(example(prefix, i)) + "\n" for i in range(count)))
    return path


@pytest.fixture
def fixture_dir(tmp_path: Path) -> Path:
    write(tmp_path / "train.jsonl", "fix", 60)
    write(tmp_path / "test.jsonl", "bench", 12)
    return tmp_path


@pytest.fixture
def local_config(fixture_dir: Path) -> dict:
    return {
        "pipeline": {"recipe": "jev", "name": "local-test", "seed": 7, "output_dir": str(fixture_dir / "output")},
        "data": {"builders": [{"name": "local_jsonl", "path": str(fixture_dir / "train.jsonl")}],
                 "folds": {"dev": 6, "temperature": 4, "validation": 5},
                 "mix": {"size": 100, "sweep_size": 10, "panel_layout": False, "escape": True, "adversarial": True}},
        "model": {"backbone": "qwen3_5", "name": "Qwen/Qwen3.5-0.8B", "revision": "2fc06364715b967f1860aea9cf38778875588b17"},
        "training": {"epochs": 1, "lr": 5e-6, "eval_every": 10},
        "evaluation": {"benchmarks": [{"name": "local_jsonl", "path": str(fixture_dir / "test.jsonl")}]},
    }


CHATML = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


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
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(fast), n_positions=256, n_embd=32, n_layer=2, n_head=2,
                                       bos_token_id=fast.eos_token_id, eos_token_id=fast.eos_token_id))
    path = tmp_path_factory.mktemp("tiny-model")
    model.save_pretrained(path)
    fast.save_pretrained(path)
    return path
