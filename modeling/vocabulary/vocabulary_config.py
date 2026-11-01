"""Config of `llm_tokenizer`: the base model whose tokenizer is measured, text rows, and how many words to pick."""

from typing import Any, ClassVar

import torch
from pydantic import Field, model_validator

from modeling.llm.models.llm_backbone import LLMBackbone
from modeling.llm.methods.llm_method import LLMMethod
from modeling.tuning.training_method import Row
from core.config.config_schema import Section
from modeling.llm.llm_config import LLMConfig


class TextRows(LLMMethod):
    """Rows: {"text": str}. Nothing is trained; the row check is all `llm_tokenizer` needs of a method."""

    def check(self, row: Row) -> None:
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError("text must be a non-empty string")

    def loss(self, backbone: LLMBackbone, rows: list[Row]) -> torch.Tensor:
        raise NotImplementedError("llm_tokenizer trains nothing")

    def evaluate(self, backbone: LLMBackbone, rows: list[Row], batch_size: int, progress: Any) -> dict[str, float]:
        raise NotImplementedError("llm_tokenizer measures tokenizers, not models")


class VocabularySettings(Section):
    count: int = Field(default=1000, ge=1, le=100_000, description="Words to pick, those that save the most tokens first")
    min_frequency: int = Field(default=5, ge=1, description="Occurrences a word needs in the training rows")


class TokenizerConfig(LLMConfig):
    """`model` names the tokenizer to extend; rows are text in the language or domain it should cover better."""
    method_class: ClassVar = TextRows
    method: TextRows.Config = Field(default_factory=TextRows.Config)
    vocabulary: VocabularySettings = Field(default_factory=VocabularySettings)

    @model_validator(mode="after")
    def _measures_only(self) -> "TokenizerConfig":
        if self.model.tokenizer is not None:
            raise ValueError("llm_tokenizer picks the tokens; model.tokenizer applies them, in the recipe that trains")
        if self.training.adapter is not None:
            raise ValueError("llm_tokenizer trains nothing; remove training.adapter")
        return self
