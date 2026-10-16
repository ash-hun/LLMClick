"""The text-to-speech recipe: `tts_piper` fine-tunes a Piper voice and checks it by listening."""

from typing import ClassVar

from modeling.tts.stages import QUALITY, FitStage, ListenStage, VoiceDataStage
from modeling.pipeline import ModelingPipeline
from modeling.tuning.sources import SOURCES
from modeling.tts.config import PiperConfig
from core.pipeline import recipe


@recipe
class PiperPipeline(ModelingPipeline[PiperConfig]):
    kind: ClassVar[str] = "tts_piper"
    config_class = PiperConfig
    stage_classes = (VoiceDataStage, FitStage, ListenStage)

    @classmethod
    def catalogue(cls) -> dict[str, list[str]]:
        return {**super().catalogue(), "quality": sorted(QUALITY), "source": SOURCES.names()}
