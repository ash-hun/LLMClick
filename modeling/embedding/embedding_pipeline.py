"""The embedding recipes."""

from typing import ClassVar

from modeling.embedding.embedding_config import ContrastiveConfig
from modeling.tuning.tuning_pipeline import TuningPipeline
from core.pipeline_runner import recipe


class EmbeddingPipeline(TuningPipeline):
    """Base of every embedding recipe; a new method is a new subclass with its `kind` and config class."""


@recipe
class ContrastivePipeline(EmbeddingPipeline):
    kind: ClassVar[str] = "embedding_contrastive"
    config_class = ContrastiveConfig
