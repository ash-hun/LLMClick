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


def test_catalogue_lists_every_recipe_with_its_stages() -> None:
    recipes = catalogue()
    assert recipes["jev"]["stages"] == ["data", "benchmarks", "synthetic", "mix", "train", "validate", "evaluate"]
    assert {"llm_sft", "llm_instruction", "llm_dpo", "llm_grpo", "embedding_contrastive"} <= set(recipes)
    assert recipes["llm_dpo"]["stages"] == ["data", "train", "validate"]
