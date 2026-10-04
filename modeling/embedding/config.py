"""One config class per embedding recipe: the family's catalog plus the method and its `method:` section."""

from typing import ClassVar

from pydantic import Field

from modeling.embedding.methods.contrastive import Contrastive
from modeling.embedding.models import EMBEDDING_BACKBONES
from modeling.tuning.config import TuningConfig


class EmbeddingConfig(TuningConfig):
    backbones: ClassVar = EMBEDDING_BACKBONES


class ContrastiveConfig(EmbeddingConfig):
    method_class: ClassVar = Contrastive
    method: Contrastive.Config = Field(default_factory=Contrastive.Config)
