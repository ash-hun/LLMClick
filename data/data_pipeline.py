"""The data recipes: `data_synthetic` turns seeds into training rows through a teacher, `data_speech` into recordings."""

from typing import Any, ClassVar

from core.pipeline_runner import Pipeline, recipe
from core.stage_base import Stage
from data.data_config import SyntheticConfig
from data.evolve_stage import EVOLVERS, EvolveStage
from data.prompts_stage import GENERATORS, PromptsStage
from data.respond_stage import RespondStage
from data.seeds_stage import SeedsStage
from data.select_stage import SelectStage
from data.teacher_client import TEACHERS
from data.verify_stage import VerifyStage
from data.speech.speech_config import SpeechConfig
from data.speech.speech_select_stage import SpeechSelectStage
from data.speech.speech_verify_stage import SpeechVerifyStage
from data.speech.synthesize_stage import SynthesizeStage


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


@recipe
class SpeechData(Pipeline[SpeechConfig]):
    """seeds -> prompts -> synthesize -> verify -> select: texts from seeds, read aloud by a Piper voice, checked by
    listening, and written as a Piper training set. Each recording and transcript is kept under its natural key."""
    kind: ClassVar[str] = "data_speech"
    config_class = SpeechConfig
    stage_classes = (SeedsStage, PromptsStage, SynthesizeStage, SpeechVerifyStage, SpeechSelectStage)

    @classmethod
    def catalogue(cls) -> dict[str, list[str]]:
        return {**super().catalogue(), "teacher": TEACHERS.names(), "generator": GENERATORS.names()}
