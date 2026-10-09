"""The data recipes: `data_synthetic` turns seeds into training rows through a teacher."""

from typing import Any, ClassVar

from core.pipeline import Pipeline, recipe
from core.stage import Stage
from data.config import SyntheticConfig
from data.evolve import EVOLVERS, EvolveStage
from data.prompts import GENERATORS, PromptsStage
from data.respond import RespondStage
from data.seeds import SeedsStage
from data.select import SelectStage
from data.teachers import TEACHERS
from data.verify import VerifyStage


@recipe
class SyntheticData(Pipeline[SyntheticConfig]):
    """seeds -> prompts -> (evolve) -> respond -> verify -> select. Every stage is cached by its inputs and safe to
    run again: the expensive ones keep each teacher call under its natural key, so a crash costs nothing twice."""
    kind: ClassVar[str] = "data_synthetic"
    config_class = SyntheticConfig
    stage_classes = (SeedsStage, PromptsStage, EvolveStage, RespondStage, VerifyStage, SelectStage)

    def build_stages(self) -> dict[str, Stage[Any]]:
        config: SyntheticConfig = self.config
        classes = [cls for cls in self.stage_classes if cls is not EvolveStage or config.evolve is not None]
        return {cls.name: cls(config, self.progress) for cls in classes}

    @classmethod
    def catalogue(cls) -> dict[str, list[str]]:
        return {**super().catalogue(), "teacher": TEACHERS.names(), "generator": GENERATORS.names(), "evolver": EVOLVERS.names()}
