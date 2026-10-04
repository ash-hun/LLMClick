"""LLM model catalog: `model.architecture` in the YAML resolves here."""

from core.registry import Registry

LLM_BACKBONES = Registry("llm architecture")

from modeling.llm.models import hybrid, transformer  # noqa: E402, F401
