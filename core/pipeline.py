"""The one pipeline runner: a recipe declares its stages, this runs them in order and skips what is already built."""

import logging
from abc import ABC
from pathlib import Path
from typing import Any, ClassVar, Generic

import yaml

from core.config.experiment import Experiment
from core.registry import RECIPES, load_channels
from core.stage import ConfigT, Outputs, Stage
from core.utils.files import locked, sha256_json
from core.config.schema import BaseConfig
from core.progress import Progress

logger = logging.getLogger(__name__)


class Pipeline(ABC, Generic[ConfigT]):
    kind: ClassVar[str]                                  # the `pipeline.recipe` key
    config_class: ClassVar[type[BaseConfig]]
    stage_classes: ClassVar[tuple[type[Stage[Any]], ...]]  # in execution order

    def __init__(self, config: ConfigT, progress: Progress | None = None) -> None:
        self.config = config
        self.progress = progress or Progress()
        self.experiment = Experiment(config)
        self.stages = self.build_stages()
        seen: set[str] = set()
        for name, stage in self.stages.items():
            missing = [other for other in stage.dependencies() if other not in seen]
            if missing:
                raise TypeError(f"{type(self).__name__}: stage {name!r} requires later or unknown stages {missing}")
            seen.add(name)
        self.selected = self.select(config.stages)
        self._fingerprints: dict[str, str] = {}

    def build_stages(self) -> dict[str, Stage[Any]]:
        """One instance per stage class, in order; a recipe whose stages depend on the config overrides this."""
        return {cls.name: cls(self.config, self.progress) for cls in self.stage_classes}

    @classmethod
    def check(cls) -> None:
        """A recipe is well formed when stage names are unique and every stage only reads earlier ones."""
        seen: set[str] = set()
        for stage in cls.stage_classes:
            missing = [name for name in stage.requires if name not in seen]
            if stage.name in seen or missing:
                raise TypeError(f"{cls.__name__}: stage {stage.name!r} is duplicated or requires later stages {missing}")
            seen.add(stage.name)

    @classmethod
    def stage_names(cls) -> list[str]:
        return [stage.name for stage in cls.stage_classes]

    @classmethod
    def catalogue(cls) -> dict[str, list[str]]:
        return {"stages": cls.stage_names()}

    def select(self, wanted: list[str] | None) -> list[str]:
        """The stages this run was asked for, in the recipe's order."""
        names = list(self.stages)
        unknown = [name for name in wanted or [] if name not in names]
        if unknown:
            raise ValueError(f"Unknown stages {unknown} for recipe {self.kind!r}; choose from {names}")
        return names if wanted is None else [name for name in names if name in wanted]

    def plan(self) -> list[str]:
        """The selected stages plus everything they read, in the recipe's order."""
        needed: set[str] = set()

        def add(name: str) -> None:
            if name not in needed:
                needed.add(name)
                for dependency in self.stages[name].dependencies():
                    add(dependency)

        for name in self.selected:
            add(name)
        return [name for name in self.stages if name in needed]

    def fingerprint(self, name: str) -> str:
        """A pure function of the config: the stage, its own inputs and the fingerprints of the stages it reads."""
        if name not in self._fingerprints:
            stage = self.stages[name]
            upstream = [self.fingerprint(dependency) for dependency in stage.dependencies()]
            owner = [stage.scope] if stage.scope else [self.kind, name]
            self._fingerprints[name] = sha256_json([owner, stage.version, stage.identity(), upstream])
        return self._fingerprints[name]

    def run_stage(self, name: str, done: dict[str, Outputs]) -> Outputs:
        stage, key = self.stages[name], self.fingerprint(name)
        workdir = self.experiment.stage_dir(stage.scope or name, key)
        self.progress.stage_started(name)

        def waiting() -> None:  # another run (CLI or job) holds this stage; its result is reused when it is done
            logger.info("[%s] %s: waiting for another run that is building it", self.experiment.key, name)
            self.progress.update(0, None, "waiting for another run that is building this stage")

        try:
            # The lock makes "is it built?" and "build it" one step, across threads and processes.
            with locked(workdir.parent / f"{workdir.name}.lock", waiting):
                outputs = self.experiment.stage_outputs(workdir, key)
                status = "cached"
                if outputs is None:
                    logger.info("[%s] %s: running", self.experiment.key, name)
                    workdir.mkdir(parents=True, exist_ok=True)
                    outputs = stage.run(workdir, {dependency: done[dependency] for dependency in stage.dependencies()})
                    outputs = self.experiment.mark_done(workdir, key, outputs)
                    status = "done"
        except BaseException:
            self.progress.stage_finished(name, "failed")
            raise
        logger.info("[%s] %s: %s", self.experiment.key, name, status)
        self.experiment.link(name, key, workdir, outputs)
        self.progress.stage_finished(name, status)
        return outputs

    def run(self) -> dict[str, Any]:
        experiment = self.experiment
        experiment.ensure()
        plan = self.plan()
        done: dict[str, Outputs] = {}
        self.progress.begin(experiment.key, plan)
        try:
            with experiment.logging():
                for name in plan:
                    done[name] = self.run_stage(name, done)
        finally:
            self.progress.end()
        return {"experiment": experiment.key, "directory": str(experiment.root), "stages": done}


def recipe(cls: type[Pipeline[Any]]) -> type[Pipeline[Any]]:
    """Class decorator: check the stage graph and make the pipeline selectable as `pipeline.recipe: <kind>`."""
    cls.check()
    registered: type[Pipeline[Any]] = RECIPES.register(cls.kind)(cls)
    return registered


def build(raw: dict[str, Any], progress: Progress | None = None) -> Pipeline[Any]:
    """A config (YAML shape: optional `pipeline:` block plus sections) -> its recipe's pipeline, validated."""
    load_channels()
    merged = {**raw.get("pipeline", {}), **{key: value for key, value in raw.items() if key != "pipeline"}}
    kind = merged.get("recipe")
    if not isinstance(kind, str) or kind not in RECIPES:
        raise ValueError(f"pipeline.recipe is {kind!r}; registered: {RECIPES.names()}")
    cls: type[Pipeline[Any]] = RECIPES.get(kind)
    return cls(cls.config_class(**merged), progress)


def load(path: str | Path, progress: Progress | None = None) -> Pipeline[Any]:
    return build(yaml.safe_load(Path(path).read_text()) or {}, progress)
