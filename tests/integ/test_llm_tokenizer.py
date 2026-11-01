"""`llm_tokenizer` end to end, and `model.tokenizer` applied by a training recipe: rows, embeddings, adapter, fingerprint."""

import json
from typing import Any
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from tests.recipe_inputs import raw
from core import pipeline_runner

WORDS = ["토크나이저에", "체크포인트는", "파이프라인은"]


def tokenizer_config(model: Path, out: Path, **keys: Any) -> dict[str, Any]:
    return {
        "pipeline": {"recipe": "llm_tokenizer", "name": "vocab", "seed": 3, "output_dir": str(out)},
        "model": {"architecture": "transformer", "name": str(model)},
        "data": {"sources": [{"name": "local_jsonl", "path": "samples/llm_pretrain_ko.jsonl"}], "validation": 0.2},
        "vocabulary": {"count": 40, "min_frequency": 3},
        "validation": {"min": {"token_reduction": 0.05}},
        **keys,
    }


def test_llm_tokenizer_picks_words_that_shorten_held_out_text(bpe_model: Path, tmp_path: Path) -> None:
    first = pipeline_runner.build(tokenizer_config(bpe_model, tmp_path)).run()
    train, metrics = first["stages"]["train"], first["stages"]["validate"]["metrics"]
    words = Path(train["tokens"]).read_text(encoding="utf-8").split()
    assert 0 < len(words) <= 40 and metrics["chars_per_token"] > metrics["base_chars_per_token"]
    assert len(AutoTokenizer.from_pretrained(train["checkpoint"])) > len(AutoTokenizer.from_pretrained(bpe_model))
    assert json.dumps(pipeline_runner.build(tokenizer_config(bpe_model, tmp_path)).run()) == json.dumps(first)
    config = tokenizer_config(bpe_model, tmp_path)
    config["model"]["tokenizer"] = {"tokens": WORDS}
    with pytest.raises(ValueError, match="model.tokenizer applies them"):
        pipeline_runner.build(config)


def sft_with_tokens(model: Path, out: Path, tokenizer: dict[str, Any], **training: Any) -> dict[str, Any]:
    config = raw("llm_sft", model, out, **training)
    config["model"]["tokenizer"] = tokenizer
    return config


def test_new_tokens_start_as_the_mean_of_their_pieces_and_ship_with_the_checkpoint(bpe_model: Path, tmp_path: Path) -> None:
    from modeling.llm.models.backbone_config import LLMBackboneConfig
    from modeling.llm.models.transformer_backbone import TransformerBackbone

    template = tmp_path / "chat.jinja"
    template.write_text("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    edit = {"tokens": WORDS, "special_tokens": ["<calc>"], "chat_template": str(template)}
    base = AutoTokenizer.from_pretrained(bpe_model)
    backbone = TransformerBackbone(LLMBackboneConfig(architecture="transformer", name=str(bpe_model), tokenizer=edit))
    backbone.load("cpu")
    embeddings = backbone.model.get_input_embeddings().weight
    word = backbone.tokenizer(f" {WORDS[0]}", add_special_tokens=False)["input_ids"]
    pieces = base(f" {WORDS[0]}", add_special_tokens=False)["input_ids"]
    assert len(word) == 1 and torch.allclose(embeddings[word[0]], embeddings[pieces].mean(dim=0))
    assert embeddings.shape[0] >= len(backbone.tokenizer) and word[0] in backbone.new_tokens
    assert backbone.tokenizer.convert_tokens_to_ids("<calc>") in backbone.new_tokens
    assert backbone.tokenizer.chat_template == template.read_text()
    result = pipeline_runner.build(sft_with_tokens(bpe_model, tmp_path, edit)).run()
    saved = AutoTokenizer.from_pretrained(result["stages"]["train"]["checkpoint"])
    assert saved(f" {WORDS[0]}", add_special_tokens=False)["input_ids"] == word and saved.chat_template == template.read_text()
    assert "<calc>" in saved.get_vocab()


def test_a_lora_adapter_trains_the_new_rows_and_nothing_else_of_the_embeddings(bpe_model: Path, tmp_path: Path) -> None:
    adapter = {"r": 4, "targets": ["c_attn"]}
    result = pipeline_runner.build(sft_with_tokens(bpe_model, tmp_path, {"tokens": WORDS}, adapter=adapter, lr=1e-2)).run()
    checkpoint = Path(result["stages"]["train"]["checkpoint"])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tuned = load_file(checkpoint / "model.safetensors")["transformer.wte.weight"]
    base = AutoTokenizer.from_pretrained(bpe_model)
    start = load_file(bpe_model / "model.safetensors")["transformer.wte.weight"]
    assert tuned.shape[0] >= len(tokenizer) > len(base)
    assert torch.equal(tuned[: len(base)], start[: len(base)])
    word = tokenizer(f" {WORDS[0]}", add_special_tokens=False)["input_ids"][0]
    pieces = base(f" {WORDS[0]}", add_special_tokens=False)["input_ids"]
    assert not torch.allclose(tuned[word], start[pieces].mean(dim=0))


def test_tokenizer_files_count_by_content_and_the_key_keeps_old_fingerprints(bpe_model: Path, tmp_path: Path) -> None:
    words = tmp_path / "tokens.txt"
    words.write_text("\n".join(WORDS[:1]) + "\n", encoding="utf-8")

    def fingerprint() -> str:
        return pipeline_runner.build(sft_with_tokens(bpe_model, tmp_path, {"tokens_file": str(words)})).fingerprint("train")

    before = fingerprint()
    words.write_text("\n".join(WORDS) + "\n", encoding="utf-8")
    assert fingerprint() != before
    plain = pipeline_runner.build(raw("llm_sft", bpe_model, tmp_path))
    assert "tokenizer" not in plain.config.model.model_dump()
    assert str(words) in pipeline_runner.build(sft_with_tokens(bpe_model, tmp_path, {"tokens_file": str(words)})).config.paths()
