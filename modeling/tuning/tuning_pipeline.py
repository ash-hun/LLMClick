"""Pipeline shared by every fine-tuning recipe; a recipe only names itself and its config class."""

from typing import Any, ClassVar

from modeling.tuning.tuning_stages import DataStage, MeasureStage, TuneStage
from modeling.tuning.row_sources import SOURCES
from modeling.tuning.tuning_config import TuningConfig
from modeling.modeling_pipeline import ModelingPipeline


class TuningPipeline(ModelingPipeline[TuningConfig]):
    config_class: ClassVar[type[TuningConfig]]
    stage_classes = (DataStage, TuneStage, MeasureStage)

    @classmethod
    def catalogue(cls) -> dict[str, list[str]]:
        method: Any = cls.config_class.method_class
        return {**super().catalogue(), "architecture": cls.config_class.backbones.names(), "source": SOURCES.names(),
                "method_keys": sorted(method.Config.model_fields)}
