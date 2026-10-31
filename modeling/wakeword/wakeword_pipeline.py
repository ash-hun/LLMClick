"""`wakeword`: recordings of a phrase and of everything else -> an openWakeWord detector, tested on streams."""

from typing import ClassVar

from modeling.wakeword.wakeword_stages import ClipsStage, FitDetectorStage, StreamTestStage, WindowsStage
from modeling.wakeword.wakeword_config import WakewordConfig
from modeling.modeling_pipeline import ModelingPipeline
from modeling.tuning.row_sources import SOURCES
from core.pipeline_runner import recipe


@recipe
class WakewordPipeline(ModelingPipeline[WakewordConfig]):
    kind: ClassVar[str] = "wakeword"
    config_class = WakewordConfig
    stage_classes = (ClipsStage, WindowsStage, FitDetectorStage, StreamTestStage)

    @classmethod
    def catalogue(cls) -> dict[str, list[str]]:
        return {**super().catalogue(), "source": SOURCES.names()}
