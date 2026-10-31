"""LLM model catalog: `model.architecture` in the YAML resolves here."""

from core.recipe_registry import Registry

LLM_BACKBONES = Registry("llm architecture")

from modeling.llm.models import hybrid_backbone, transformer_backbone  # noqa: E402, F401
