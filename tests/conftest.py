"""Fixtures that need no download or GPU: a tiny random LM and tokenizer the recipes can train on a CPU."""

from pathlib import Path

import pytest

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
