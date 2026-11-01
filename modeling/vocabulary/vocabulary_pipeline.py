"""Recipe `llm_tokenizer`: text rows, then the words worth adding to a tokenizer, then the tokens they save."""

from typing import ClassVar

from modeling.vocabulary.vocabulary_stages import VocabularyMeasureStage, VocabularyStage
from modeling.vocabulary.vocabulary_config import TokenizerConfig
from modeling.modeling_pipeline import ModelingPipeline
from modeling.tuning.tuning_stages import DataStage
from core.pipeline_runner import recipe


@recipe
class VocabularyPipeline(ModelingPipeline[TokenizerConfig]):
    kind: ClassVar[str] = "llm_tokenizer"
    config_class = TokenizerConfig
    stage_classes = (DataStage, VocabularyStage, VocabularyMeasureStage)
