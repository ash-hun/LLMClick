"""Decision heads: `model.head` in the YAML resolves here."""

from core.recipe_registry import Registry

HEADS = Registry("head")

from modeling.llm.models.heads import pointer_head, readout_head  # noqa: E402, F401
