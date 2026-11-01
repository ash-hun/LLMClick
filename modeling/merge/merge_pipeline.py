"""Recipe `llm_merge`: data, then merging in place of training, then validation against the base model's loss."""

from typing import Any, ClassVar

from modeling.merge.merge_stages import MergeMeasureStage, MergeStage
from modeling.tuning.tuning_pipeline import TuningPipeline
from modeling.merge.merge_config import MergeConfig
from modeling.tuning.tuning_stages import DataStage
from modeling.merge.merge_rule import RULES
from core.pipeline_runner import recipe


@recipe
class MergePipeline(TuningPipeline):
    kind: ClassVar[str] = "llm_merge"
    config_class = MergeConfig
    stage_classes = (DataStage, MergeStage, MergeMeasureStage)

    @classmethod
    def catalogue(cls) -> dict[str, list[str]]:
        catalogue: dict[str, Any] = super().catalogue()
        return {**catalogue, "rule": RULES.names()}
