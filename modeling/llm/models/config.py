"""The `model:` section of LLM recipes: the shared backbone keys plus what only language models have."""

from typing import Any

from pydantic import Field

from modeling.tuning.config import BackboneConfig


class LLMBackboneConfig(BackboneConfig):
    template: dict[str, Any] = Field(default_factory=dict, description="Extra chat-template arguments, e.g. enable_thinking")
    head: str | None = Field(default=None, description="Decision head on top of the backbone; decision recipes only")
