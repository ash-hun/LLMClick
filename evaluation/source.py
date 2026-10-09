"""The model under evaluation: an experiment's config rebuilt and its checkpoint located, or a Hub model as it is."""

from pathlib import Path
from typing import Any

import yaml

from core.pipeline import build
from core.utils.files import directory_signature
from evaluation.config import BUILT, SourceModel
from modeling.tuning.backbone import Backbone
from modeling.tuning.config import BackboneConfig, TuningConfig
from modeling.tuning.method import TrainingMethod

CONFIG = "config.yaml"


class SourceExperiment:
    """`config.yaml` of a finished modeling experiment, read back into its config class, so the evaluation measures
    the model exactly as the recipe that trained it does: same backbone class, same method, same metrics. For a Hub
    model (`model.name`) there is no config: only `origin` and, with an architecture, `hub_backbone` apply."""

    def __init__(self, model: SourceModel) -> None:
        self.model = model
        self.root = Path(model.experiment) if model.experiment is not None else None

    @property
    def hub(self) -> bool:
        return self.model.hub

    @property
    def config_path(self) -> Path | None:
        return self.root / CONFIG if self.root is not None else None

    def exists(self) -> bool:
        return self.config_path is not None and self.config_path.is_file()

    def config(self) -> TuningConfig:
        if self.hub:
            raise ValueError(f"model.name {self.model.name!r} is a model without an experiment: only the benchmark and "
                             "embedding recipes can score it; the others need the recipe that trained the model")
        if not self.exists():
            raise FileNotFoundError(f"model.experiment {self.model.experiment!r} has no {CONFIG}; run that experiment first")
        assert self.config_path is not None
        config = build(yaml.safe_load(self.config_path.read_text()) or {}).config
        if not isinstance(config, TuningConfig):
            raise ValueError(f"recipe {config.recipe!r} of {self.model.experiment!r} trains nothing this channel can measure")
        return config

    def recipe(self) -> str | None:
        return self.config().recipe if self.exists() else None

    def described(self) -> dict[str, Any]:
        """The model as reports name it."""
        if self.hub:
            return {"experiment": None, "name": self.model.name, "revision": self.model.revision,
                    "architecture": self.model.architecture, "checkpoint": "hub", "recipe": None}
        checkpoint = self.located()
        return {"experiment": self.model.experiment, "name": None, "revision": None, "architecture": None,
                "checkpoint": str(checkpoint) if checkpoint is not None else "base", "recipe": self.recipe()}

    def located(self) -> Path | None:
        """Where the checkpoint would be, or None for the weights the experiment started from (`base`) and Hub models."""
        wanted = self.model.checkpoint
        if self.hub or wanted == "base" or self.root is None:
            return None
        return self.root / wanted / "checkpoint" if wanted in BUILT else Path(wanted)

    def checkpoint(self) -> Path | None:
        """The directory to load, checked to exist, or None for `base` and Hub models."""
        path = self.located()
        if path is not None and not path.is_dir():
            hint = {"validate": "; it exists only once the experiment passed validation", "train": "; run the experiment first"}
            raise FileNotFoundError(f"checkpoint {str(path)!r} does not exist{hint.get(self.model.checkpoint, '')}")
        return path

    def origin(self) -> tuple[str, str | None]:
        """What to load, as a loader wants it: a directory or a Hub ID with its revision."""
        if self.hub:
            name = str(self.model.name)
            return name, None if Path(name).is_dir() else self.model.revision
        return self.config().build_backbone().origin(self.checkpoint())

    def identity(self) -> Any:
        """What decides the scores besides the rows: the experiment's config and the content of the checkpoint, or
        the Hub model's name and revision. Tolerant of a missing experiment, so an evaluation config validates
        before the experiment has run."""
        if self.hub:
            name = str(self.model.name)
            return [self.model.model_dump(mode="json"), directory_signature(Path(name)) if Path(name).is_dir() else None]
        if not self.exists():
            return [self.model.model_dump(mode="json"), None]
        config, checkpoint = self.config(), self.located()
        weights = [config.model.model_dump(mode="json")] if checkpoint is None else directory_signature(checkpoint)
        return [config.identity(), config.versions(), self.model.checkpoint, weights]

    def build(self) -> tuple[Backbone, TrainingMethod[Any], TuningConfig]:
        config = self.config()
        return config.build_backbone(), config.build_method(), config

    def hub_backbone(self) -> Backbone:
        """A Hub model through the catalog entry `model.architecture` names, in whichever family has it."""
        from modeling.embedding.models import EMBEDDING_BACKBONES
        from modeling.llm.models import LLM_BACKBONES
        from modeling.llm.models.config import LLMBackboneConfig
        architecture, name = self.model.architecture, str(self.model.name)
        if architecture is None:
            raise ValueError("model.architecture is needed to load a Hub model outside an experiment")
        if architecture in EMBEDDING_BACKBONES:
            backbone: Backbone = EMBEDDING_BACKBONES.get(architecture)(BackboneConfig(architecture=architecture, name=name, revision=self.model.revision))
            return backbone
        if architecture in LLM_BACKBONES:
            backbone = LLM_BACKBONES.get(architecture)(LLMBackboneConfig(architecture=architecture, name=name, revision=self.model.revision))
            return backbone
        raise ValueError(f"unknown architecture {architecture!r}; registered: {EMBEDDING_BACKBONES.names() + LLM_BACKBONES.names()}")
