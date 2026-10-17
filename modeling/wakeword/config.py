"""Config of `wakeword`: the phrase, its recordings and the others, the augmentation, the detector and the streaming test."""

import re

from pydantic import Field, model_validator

from modeling.config import Keyed, ModelingConfig, ValidationConfig
from core.config.schema import Section


class WakewordData(Section):
    positive: list[Keyed] = Field(min_length=1, description="Row sources of recordings of the phrase (`audio`, optional `text`)")
    negative: list[Keyed] = Field(min_length=1, description="Row sources of anything else: near misses, commands, speech")
    background: list[str] = Field(default_factory=list, description="Directories of noise WAVs mixed under every window; "
                                                                    "empty: Gaussian noise")
    validation: float = Field(default=0.2, gt=0, lt=1, description="Share of each class held out for the streaming test")


class AugmentConfig(Section):
    rounds: int = Field(default=4, ge=1, description="Windows cut from every training recording, each with its own noise")
    snr: tuple[float, float] = Field(default=(5.0, 25.0), description="Range of speech-to-noise ratio, dB")
    gain: tuple[float, float] = Field(default=(-12.0, 0.0), description="Range of gain on the whole window, dB")
    tail: float = Field(default=0.3, ge=0, lt=1, description="Most seconds of noise after a recording's end in its window")
    followed: int = Field(default=2, ge=0, description="Negative windows per phrase recording where other speech follows "
                                                       "the phrase, so the detector fires as the phrase ends and not again")
    noise: int = Field(default=200, ge=0, description="Windows of noise alone, as more negatives")


class WakewordTraining(Section):
    steps: int = Field(default=3000, ge=1)
    batch_size: int = Field(default=256, ge=2, description="Half positive windows, half negative")
    learning_rate: float = Field(default=1e-3, gt=0)
    layer_dim: int = Field(default=32, ge=1, description="Width of openWakeWord's DNN head")
    blocks: int = Field(default=1, ge=0, description="Hidden blocks after the first layer")


class WakewordValidation(ValidationConfig):
    threshold: float = Field(default=0.5, gt=0, lt=1, description="Score at or above which the detector fires")
    snr: float | None = Field(default=20.0, description="Speech-to-noise ratio of the test streams, dB; null: clean")


class WakewordConfig(ModelingConfig):
    phrase: str = Field(min_length=1, description="What wakes the agent, e.g. 'Jarvis, wake up'; names the model file")
    data: WakewordData
    augment: AugmentConfig = Field(default_factory=AugmentConfig)
    training: WakewordTraining = Field(default_factory=WakewordTraining)
    validation: WakewordValidation = Field(default_factory=WakewordValidation)

    @model_validator(mode="after")
    def _registered(self) -> "WakewordConfig":
        from modeling.tuning.sources import SOURCES
        for source in [*self.data.positive, *self.data.negative]:
            if source.name not in SOURCES:
                raise ValueError(f"Unknown source {source.name!r}; registered: {SOURCES.names()}")
        for name, (low, high) in (("augment.snr", self.augment.snr), ("augment.gain", self.augment.gain)):
            if low > high:
                raise ValueError(f"{name} is a range: [low, high]")
        return self

    @property
    def slug(self) -> str:
        """The phrase as a file name, the name openWakeWord reports its score under: 'Jarvis, wake up' -> jarvis_wake_up."""
        return "_".join(re.findall(r"[a-z0-9]+", self.phrase.lower())) or "wakeword"

    def paths(self) -> list[str]:
        local = [str(path) for source in [*self.data.positive, *self.data.negative] for key in ("path", "audio_dir")
                 if (path := source.params.get(key)) is not None]
        return [*super().paths(), *local, *self.data.background]
