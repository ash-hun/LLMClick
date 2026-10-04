"""What makes a pipeline a modeling pipeline: it trains, and a trained model is always validated."""

from core.pipeline import Pipeline
from core.stage import Outputs
from modeling.stages import ModelingConfigT, TrainStage, ValidateStage
from modeling.tracker import Tracker


class ModelingPipeline(Pipeline[ModelingConfigT]):
    @classmethod
    def check(cls) -> None:
        super().check()
        for base in (TrainStage, ValidateStage):
            if not any(issubclass(stage, base) for stage in cls.stage_classes):
                raise TypeError(f"{cls.__name__} needs a {base.__name__} subclass among its stages")

    def select(self, wanted: list[str] | None) -> list[str]:
        """Asking for training is asking for validation too: it cannot be left out of `stages`."""
        selected = super().select(wanted)
        if TrainStage.name in selected and ValidateStage.name not in selected:
            selected = [name for name in self.stage_names() if name in {*selected, ValidateStage.name}]
        return selected

    def run_stage(self, name: str, done: dict[str, Outputs]) -> Outputs:
        """After a train stage, built now or taken from cache, its history goes to the tracker if it has not yet."""
        outputs = super().run_stage(name, done)
        stage = self.stages[name]
        if isinstance(stage, TrainStage):
            run = self.experiment.stage_dir(name, self.fingerprint(name))
            Tracker(self.config.tracker).sync(run, self.config.name, stage.history(outputs), self.config.model_dump(mode="json"))
        return outputs
