"""`llm_tokenizer` modeling/llm/models/vocabulary_extender.py: words become merges, other text tokenizes as before."""

import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

TEXTS = [json.loads(line)["text"] for line in Path("samples/llm_pretrain_ko.jsonl").read_text(encoding="utf-8").splitlines()]


def test_a_word_becomes_one_token_after_a_space_and_at_the_start(bpe_model: Path) -> None:
    from modeling.llm.models.vocabulary_extender import extend

    tokenizer = AutoTokenizer.from_pretrained(bpe_model)
    word = "토크나이저에"
    assert len(tokenizer(word, add_special_tokens=False)["input_ids"]) > 1
    other = "학습률이 너무 크면 손실이 발산합니다."
    other_ids = tokenizer(other, add_special_tokens=False)["input_ids"]
    created = extend(tokenizer, [word])
    assert created and all(parts for parts in created.values())
    assert len(tokenizer(word, add_special_tokens=False)["input_ids"]) == 1
    lead = len(tokenizer("모델", add_special_tokens=False)["input_ids"])
    assert len(tokenizer(f"모델 {word}", add_special_tokens=False)["input_ids"]) == lead + 1
    assert tokenizer(other, add_special_tokens=False)["input_ids"] == other_ids
    assert all(tokenizer.decode(tokenizer(text, add_special_tokens=False)["input_ids"]) == text for text in TEXTS)
    assert extend(tokenizer, [word]) == {}


def test_extension_survives_saving_and_keeps_special_token_ids(bpe_model: Path, tmp_path: Path) -> None:
    from modeling.llm.models.vocabulary_extender import extend

    tokenizer = AutoTokenizer.from_pretrained(bpe_model)
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    extend(tokenizer, ["체크포인트는", "파이프라인은"])
    tokenizer.save_pretrained(tmp_path)
    again = AutoTokenizer.from_pretrained(tmp_path)
    text = "체크포인트는 파이프라인은<|im_end|>"
    assert again(text)["input_ids"] == tokenizer(text)["input_ids"] and again.convert_tokens_to_ids("<|im_end|>") == end


def test_candidates_rank_by_tokens_saved_and_chars_per_token_rises(bpe_model: Path) -> None:
    from modeling.llm.models.vocabulary_extender import candidates, chars_per_token, extend

    tokenizer = AutoTokenizer.from_pretrained(bpe_model)
    ranked = candidates(tokenizer, TEXTS[:100], 30, 3)
    assert ranked and [saved for _, _, saved in ranked] == sorted((saved for _, _, saved in ranked), reverse=True)
    assert all(count >= 3 and any(c.isalpha() for c in word) for word, count, _ in ranked)
    before = chars_per_token(tokenizer, TEXTS[100:])
    extend(tokenizer, [word for word, _, _ in ranked])
    assert chars_per_token(tokenizer, TEXTS[100:]) > before


def test_only_byte_level_bpe_tokenizers_are_extended(tiny_model: Path) -> None:
    from modeling.llm.models.vocabulary_extender import extend

    with pytest.raises(ValueError, match="byte-level BPE"):
        extend(AutoTokenizer.from_pretrained(tiny_model), ["word"])
