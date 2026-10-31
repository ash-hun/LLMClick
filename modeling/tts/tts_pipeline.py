"""The text-to-speech recipes: `tts_piper` fine-tunes a Piper voice, `tts_qwen` clones one with Qwen3-TTS; both listen."""

from typing import ClassVar

from modeling.tts.tts_stages import QUALITY, CloneStage, FitStage, PiperListenStage, QwenListenStage, VoiceDataStage
from modeling.tts.tts_config import PiperConfig, QwenConfig
from modeling.modeling_pipeline import ModelingPipeline
from modeling.tuning.row_sources import SOURCES
from core.pipeline_runner import recipe


@recipe
class PiperPipeline(ModelingPipeline[PiperConfig]):
    kind: ClassVar[str] = "tts_piper"
    config_class = PiperConfig
    stage_classes = (VoiceDataStage, FitStage, PiperListenStage)

    @classmethod
    def catalogue(cls) -> dict[str, list[str]]:
        return {**super().catalogue(), "quality": sorted(QUALITY), "source": SOURCES.names()}


@recipe
class QwenPipeline(ModelingPipeline[QwenConfig]):
    kind: ClassVar[str] = "tts_qwen"
    config_class = QwenConfig
    stage_classes = (VoiceDataStage, CloneStage, QwenListenStage)

    @classmethod
    def catalogue(cls) -> dict[str, list[str]]:
        return {**super().catalogue(), "source": SOURCES.names()}
