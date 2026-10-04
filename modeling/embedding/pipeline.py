"""The embedding recipes."""

from typing import ClassVar

from modeling.embedding.config import ContrastiveConfig
from modeling.tuning.pipeline import TuningPipeline
from core.pipeline import recipe


class EmbeddingPipeline(TuningPipeline):
    """Base of every embedding recipe; a new method is a new subclass with its `kind` and config class."""


@recipe
class ContrastivePipeline(EmbeddingPipeline):
    kind: ClassVar[str] = "embedding_contrastive"
    config_class = ContrastiveConfig
