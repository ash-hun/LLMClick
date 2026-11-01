"""The `model:` section of LLM recipes: the shared backbone keys plus what only language models have."""

from typing import Any, ClassVar

from pydantic import Field

from modeling.tuning.tuning_config import BackboneConfig
from core.config.config_schema import Section


class VocabularyConfig(Section):
    """Changes to the tokenizer, applied when the base model (or `model.init`) is loaded and saved with the checkpoint."""
    tokens: list[str] = Field(default_factory=list, description="Words and terms to make single tokens, e.g. frequent "
                                                                "Korean words or domain terms; each also after a space")
    tokens_file: str | None = Field(default=None, description="More of them, one per line: e.g. the tokens.txt llm_tokenizer writes")
    special_tokens: list[str] = Field(default_factory=list, description="Control tokens never split, e.g. markers of a new tool format")
    chat_template: str | None = Field(default=None, description="A Jinja chat template file that replaces the model's own")

    def files(self) -> list[str]:
        return [path for path in (self.tokens_file, self.chat_template) if path is not None]


class LLMBackboneConfig(BackboneConfig):
    added_keys: ClassVar[frozenset[str]] = frozenset({"tokenizer"})
    template: dict[str, Any] = Field(default_factory=dict, description="Extra chat-template arguments, e.g. enable_thinking")
    head: str | None = Field(default=None, description="Decision head on top of the backbone; decision recipes only")
    tokenizer: VocabularyConfig | None = Field(default=None, description="Tokens to add and a chat template to use")
