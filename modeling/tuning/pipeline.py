"""Pipeline shared by every fine-tuning recipe; a recipe only names itself and its config class."""

from typing import Any, ClassVar

from modeling.tuning.stages import DataStage, MeasureStage, TuneStage
from modeling.tuning.sources import SOURCES
from modeling.tuning.config import TuningConfig
from modeling.pipeline import ModelingPipeline


class TuningPipeline(ModelingPipeline[TuningConfig]):
    config_class: ClassVar[type[TuningConfig]]
    stage_classes = (DataStage, TuneStage, MeasureStage)

    @classmethod
    def catalogue(cls) -> dict[str, list[str]]:
        method: Any = cls.config_class.method_class
        heads = cls.config_class.heads
        return {**super().catalogue(), "architecture": cls.config_class.backbones.names(), "source": SOURCES.names(),
                "method_keys": sorted(method.Config.model_fields), **({"head": heads.names()} if heads else {})}
