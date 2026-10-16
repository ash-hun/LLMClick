"""Config of `data_speech`: the seeds and prompts of the text recipes, plus the voice and the speech checks."""

from pydantic import Field

from modeling.tts.config import AsrConfig, HubFile
from core.config.schema import Section
from data.config import SeedConfig


class VoiceSettings(Section):
    """The voice that reads the texts, and how."""
    model: HubFile = Field(description="A Piper voice (.onnx); its .onnx.json is taken from beside it")
    samples: int = Field(default=1, ge=1, description="Recordings per text; Piper's noise makes each one different")
    length_scale: float | None = Field(default=None, gt=0, description="Above 1 speaks slower; null: the voice's own")
    noise_scale: float | None = Field(default=None, ge=0, description="null: the voice's own")
    noise_w_scale: float | None = Field(default=None, ge=0, description="null: the voice's own")

    def scales(self) -> dict[str, float]:
        return {name: value for name in ("length_scale", "noise_scale", "noise_w_scale") if (value := getattr(self, name)) is not None}


class SpeechVerifySettings(Section):
    min_seconds: float = Field(default=0.5, ge=0)
    max_seconds: float = Field(default=15.0, gt=0, description="Piper trains on short clips; Whisper hears 30 s at most")
    max_clipped: float = Field(default=0.001, ge=0, le=1, description="Share of samples at full scale")
    max_silent: float = Field(default=0.5, ge=0, le=1, description="Share of silent 20 ms frames")
    asr: AsrConfig | None = Field(default_factory=AsrConfig, description="null skips the listening check")
    max_wer: float = Field(default=0.2, ge=0, description="Word error rate of the transcript against the text")


class SpeechSelectSettings(Section):
    max_rows: int | None = Field(default=None, ge=1)


class SpeechConfig(SeedConfig):
    voice: VoiceSettings
    verify: SpeechVerifySettings = Field(default_factory=SpeechVerifySettings)
    select: SpeechSelectSettings = Field(default_factory=SpeechSelectSettings)
    licence: str = Field(min_length=1, description="Terms of the voice, and so of every recording it makes; kept in the manifest")

    def paths(self) -> list[str]:
        return [*super().paths(), *([] if self.voice.model.hub else [self.voice.model.name])]
