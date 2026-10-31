"""Embedding model catalog: `model.architecture` in the YAML resolves here."""

from core.recipe_registry import Registry

EMBEDDING_BACKBONES = Registry("embedding architecture")

from modeling.embedding.models import bi_encoder_backbone  # noqa: E402, F401
