"""Embedding model catalog: `model.architecture` in the YAML resolves here."""

from core.registry import Registry

EMBEDDING_BACKBONES = Registry("embedding architecture")

from modeling.embedding.models import bi_encoder  # noqa: E402, F401
