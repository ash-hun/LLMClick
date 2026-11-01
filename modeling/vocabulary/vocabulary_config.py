"""Config of `llm_tokenizer`: the base model whose tokenizer is measured, text rows, and how many words to pick."""

from typing import ClassVar

from pydantic import Field, model_validator

from modeling.llm.methods.pretrain_method import Pretrain
from core.config.config_schema import Section
from modeling.llm.llm_config import LLMConfig


class VocabularySettings(Section):
    count: int = Field(default=1000, ge=1, le=100_000, description="Words to pick, those that save the most tokens first")
    min_frequency: int = Field(default=5, ge=1, description="Occurrences a word needs in the training rows")


class TokenizerConfig(LLMConfig):
    """`model` names the tokenizer to extend; rows are text in the language or domain it should cover better."""
    method_class: ClassVar = Pretrain  # its row check: the same text rows `llm_pretrain` trains on
    method: Pretrain.Config = Field(default_factory=Pretrain.Config)
    vocabulary: VocabularySettings = Field(default_factory=VocabularySettings)

    @model_validator(mode="after")
    def _measures_only(self) -> "TokenizerConfig":
        if self.model.tokenizer is not None:
            raise ValueError("llm_tokenizer picks the tokens; model.tokenizer applies them, in the recipe that trains")
        if self.training.adapter is not None:
            raise ValueError("llm_tokenizer trains nothing; remove training.adapter")
        return self
