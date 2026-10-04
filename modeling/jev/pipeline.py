"""The jev recipe: its config schema and stage order, registered as `pipeline.recipe: jev`."""

from typing import ClassVar

from modeling.jev.stages import (BenchmarksStage, DataStage, EvaluateStage, JevTrainStage, JevValidateStage, MixStage,
                                 SyntheticStage)
from modeling.pipeline import ModelingPipeline
from modeling.jev.config import JevConfig
from modeling.jev import registry
from core.pipeline import recipe


@recipe
class JevPipeline(ModelingPipeline[JevConfig]):
    kind: ClassVar[str] = "jev"
    config_class = JevConfig
    stage_classes = (DataStage, BenchmarksStage, SyntheticStage, MixStage, JevTrainStage, JevValidateStage,
                     EvaluateStage)

    @classmethod
    def catalogue(cls) -> dict[str, list[str]]:
        return {**super().catalogue(), **registry.names()}
