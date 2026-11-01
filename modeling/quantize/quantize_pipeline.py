"""Recipe `llm_quantize`: data, then quantization in place of training, then validation against the base model's loss."""

from typing import Any, ClassVar

from modeling.quantize.quantize_stages import QuantizeMeasureStage, QuantizeStage
from modeling.quantize.quantize_config import QuantizeConfig
from modeling.tuning.tuning_pipeline import TuningPipeline
from modeling.quantize.quantize_scheme import SCHEMES
from modeling.tuning.tuning_stages import DataStage
from core.pipeline_runner import recipe


@recipe
class QuantizePipeline(TuningPipeline):
    kind: ClassVar[str] = "llm_quantize"
    config_class = QuantizeConfig
    stage_classes = (DataStage, QuantizeStage, QuantizeMeasureStage)

    @classmethod
    def catalogue(cls) -> dict[str, list[str]]:
        catalogue: dict[str, Any] = super().catalogue()
        return {**catalogue, "quantization": SCHEMES.names()}
