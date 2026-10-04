"""Decision heads: `model.head` in the YAML resolves here."""

from core.registry import Registry

HEADS = Registry("head")

from modeling.llm.models.heads import pointer, readout  # noqa: E402, F401
