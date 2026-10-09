"""The experiment under evaluation: its config rebuilt, its checkpoint located, its backbone and method ready."""

from pathlib import Path
from typing import Any

import yaml

from core.pipeline import build
from core.utils.files import directory_signature
from evaluation.config import BUILT, SourceModel
from modeling.tuning.backbone import Backbone
from modeling.tuning.config import TuningConfig
from modeling.tuning.method import TrainingMethod

CONFIG = "config.yaml"


class SourceExperiment:
    """`config.yaml` of a finished modeling experiment, read back into its config class, so the evaluation measures
    the model exactly as the recipe that trained it does: same backbone class, same method, same metrics."""

    def __init__(self, model: SourceModel) -> None:
        self.model = model
        self.root = Path(model.experiment)

    @property
    def config_path(self) -> Path:
        return self.root / CONFIG

    def exists(self) -> bool:
        return self.config_path.is_file()

    def config(self) -> TuningConfig:
        if not self.exists():
            raise FileNotFoundError(f"model.experiment {self.model.experiment!r} has no {CONFIG}; run that experiment first")
        config = build(yaml.safe_load(self.config_path.read_text()) or {}).config
        if not isinstance(config, TuningConfig):
            raise ValueError(f"recipe {config.recipe!r} of {self.model.experiment!r} trains nothing this channel can measure")
        return config

    def located(self) -> Path | None:
        """Where the checkpoint would be, or None for the weights the experiment started from (`base`)."""
        wanted = self.model.checkpoint
        if wanted == "base":
            return None
        return self.root / wanted / "checkpoint" if wanted in BUILT else Path(wanted)

    def checkpoint(self) -> Path | None:
        """The directory to load, checked to exist, or None for `base`."""
        path = self.located()
        if path is not None and not path.is_dir():
            hint = {"validate": "; it exists only once the experiment passed validation", "train": "; run the experiment first"}
            raise FileNotFoundError(f"checkpoint {str(path)!r} does not exist{hint.get(self.model.checkpoint, '')}")
        return path

    def identity(self) -> Any:
        """What decides the scores besides the rows: the experiment's config and the content of the checkpoint.
        Tolerant of a missing experiment, so an evaluation config validates before the experiment has run."""
        if not self.exists():
            return [self.model.model_dump(mode="json"), None]
        config, checkpoint = self.config(), self.located()
        weights = [config.model.model_dump(mode="json")] if checkpoint is None else directory_signature(checkpoint)
        return [config.identity(), config.versions(), self.model.checkpoint, weights]

    def build(self) -> tuple[Backbone, TrainingMethod[Any], TuningConfig]:
        config = self.config()
        return config.build_backbone(), config.build_method(), config
