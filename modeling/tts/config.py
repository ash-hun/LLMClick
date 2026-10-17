"""Config of `tts_piper`: the voice to start from, the recordings, Piper's training settings and the listening test."""

import re
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from modeling.config import ModelingConfig, ValidationConfig, local_paths
from modeling.tuning.config import DataConfig, is_commit
from core.utils.files import directory_signature, sha256_file
from core.config.schema import Section
from core.settings import get_settings

CHECKPOINT_FILE = "model.ckpt"  # inside a checkpoint directory this recipe writes, next to the voice
WHISPER = "openai/whisper-base"
WHISPER_REVISION = "e37978b90ca9030d5170a5c07aadb050351a65bb"
# Options this recipe sets itself; `training.args` may not override them.
MANAGED = frozenset({"data.voice_name", "data.csv_path", "data.audio_dir", "data.espeak_voice", "data.cache_dir",
                     "data.config_path", "data.batch_size", "data.validation_split", "data.num_test_examples",
                     "model.sample_rate", "model.warmstart_ckpt", "trainer.max_epochs", "trainer.default_root_dir",
                     "trainer.accelerator", "trainer.devices", "seed_everything", "ckpt_path"})


class HubFile(Section):
    """One file: in a Hub repository at a pinned commit, or a local path (a checkpoint directory of an earlier
    experiment counts as its `model.ckpt`)."""
    name: str = Field(description="Hub repository ID, or a local file or directory")
    revision: str | None = Field(default=None, description="40-character commit; with `file`, for a Hub repository")
    file: str | None = Field(default=None, description="Path of the file inside the Hub repository")

    @model_validator(mode="after")
    def _pinned(self) -> "HubFile":
        # A local path is checked when it is read: a config that continues another experiment must validate first.
        if (self.revision, self.file) != (None, None) and not (self.file and is_commit(self.revision)):
            raise ValueError("a Hub file needs `file` and a 40-character commit `revision` (a branch is not reproducible)")
        return self

    @property
    def hub(self) -> bool:
        return self.file is not None

    def local(self, *companions: str) -> Path:
        """The file on disk, downloaded first when it is on the Hub; `companions` are suffixes of files that come
        with it (a voice's `.json`), downloaded alongside."""
        if not self.hub:
            path = Path(self.name)
            if not path.exists():
                raise FileNotFoundError(f"{self.name!r} does not exist; run the experiment that writes it")
            return path / CHECKPOINT_FILE if path.is_dir() else path
        from huggingface_hub import hf_hub_download
        assert self.file is not None
        token = get_settings().HF_TOKEN or None
        for suffix in companions:
            hf_hub_download(self.name, self.file + suffix, revision=self.revision, token=token)
        return Path(hf_hub_download(self.name, self.file, revision=self.revision, token=token))

    def identity(self) -> Any:
        """A pinned Hub file is its coordinates; a local one counts by content, so a rebuilt checkpoint is a new input."""
        if self.hub:
            return self.model_dump(mode="json")
        path = Path(self.name)
        if path.is_dir():
            return directory_signature(path)
        return [str(path.resolve()), sha256_file(path)] if path.is_file() else None


class AsrConfig(Section):
    """The speech recognizer that checks what a recording says."""
    name: str = Field(default=WHISPER, description="Whisper model on the Hub")
    revision: str = Field(default=WHISPER_REVISION, description="40-character commit")
    language: str | None = Field(default=None, description="Language code (en, ko, ...); null lets Whisper detect it")


class VoiceModelConfig(Section):
    quality: Literal["medium", "high"] = Field(default="medium", description="Piper's model size; `init` must have the same")
    espeak_voice: str = Field(description="espeak-ng voice that turns text into phonemes, e.g. en-us, en-gb-x-rp")
    sample_rate: int = Field(default=22050, ge=8000)
    init: HubFile = Field(description="Piper checkpoint (.ckpt) whose weights training starts from")


class PiperTrainingConfig(Section):
    epochs: int = Field(default=100, ge=1, description="Epochs over the training rows from the starting weights")
    batch_size: int = Field(default=16, ge=1)
    args: dict[str, str | int | float | bool] = Field(default_factory=dict,
                                                      description="More `piper.train fit` options, e.g. model.learning_rate: 1.0e-4")

    @model_validator(mode="after")
    def _unmanaged(self) -> "PiperTrainingConfig":
        clash = sorted(MANAGED & set(self.args))
        if clash:
            raise ValueError(f"training.args cannot set {clash}: the recipe sets them from the config")
        return self


class PiperValidationConfig(ValidationConfig):
    asr: AsrConfig = Field(default_factory=AsrConfig)
    rows: int | None = Field(default=50, ge=1, description="Held-out rows to synthesize and transcribe; null: all of them")
    phrases: dict[str, str] = Field(default_factory=dict,
                                    description="Fixed replies the agent plays (name -> text), e.g. web_search: Searching "
                                                "Reference.; the best-heard take is kept as phrases/<name>.wav and must be understood")
    phrase_takes: int = Field(default=3, ge=1, description="Takes of each phrase; Piper's unseeded noise makes each one "
                                                           "different, and Whisper picks the best")

    @model_validator(mode="after")
    def _named(self) -> "PiperValidationConfig":
        bad = sorted(name for name in self.phrases if not re.fullmatch(r"[A-Za-z0-9_-]+", name))
        if bad:
            raise ValueError(f"validation.phrases names become file names: {bad} may only use letters, digits, _ and -")
        return self


class PiperConfig(ModelingConfig):
    model: VoiceModelConfig
    data: DataConfig
    training: PiperTrainingConfig = Field(default_factory=PiperTrainingConfig)
    validation: PiperValidationConfig = Field(default_factory=PiperValidationConfig)

    @model_validator(mode="after")
    def _registered(self) -> "PiperConfig":
        from modeling.tuning.sources import SOURCES
        SOURCES.check(*(source.name for source in self.data.sources))
        return self

    def paths(self) -> list[str]:
        from modeling.tts.audio import AUDIO_KEYS
        return [*super().paths(), *([] if self.model.init.hub else [self.model.init.name]),
                *local_paths(self.data.sources, AUDIO_KEYS)]
