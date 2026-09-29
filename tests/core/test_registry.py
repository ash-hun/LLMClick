import pytest

from core.registry import Registry, catalogue


def test_duplicate_registration_of_a_different_object_fails() -> None:
    registry = Registry("thing")
    registry.register("a")(object())
    with pytest.raises(ValueError, match="already registered"):
        registry.register("a")(object())


def test_unknown_key_lists_alternatives() -> None:
    registry = Registry("thing")
    registry.register("b")(1)
    registry.register("a")(2)
    with pytest.raises(KeyError, match=r"\['a', 'b'\]"):
        registry.get("zzz")


def test_builtin_catalogue() -> None:
    names = catalogue()
    assert {"local_jsonl", "huggingface", "jeff_extra", "jeff_probability"} <= set(names["builder"])
    assert {"example", "classification", "boolean"} <= set(names["converter"])
    assert {"qwen3_5", "gemma4", "modernbert"} <= set(names["backbone"])
    assert "openai_compatible" in names["teacher"]
    assert {"local_jsonl", "huggingface", "jeff_panel", "jeff_jevbench_hard"} <= set(names["benchmark"])
