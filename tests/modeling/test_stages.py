"""The modeling contract on a toy recipe: training always brings validation, and failed validation stops the run."""

from pathlib import Path
from typing import ClassVar

import pytest

from core.pipeline import Pipeline
from core.stage import Outputs, Stage
from modeling.config import ModelingConfig, ValidationConfig
from modeling.pipeline import ModelingPipeline
from modeling.stages import TrainStage, ValidateStage, ValidationFailed, violations


class ToyConfig(ModelingConfig):
    recipe: str = "toy-model"
    score: float = 0.9


class Train(TrainStage[ToyConfig]):
    sections: ClassVar[tuple[str, ...]] = ("score",)

    def train(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        (workdir / "model.txt").write_text(str(self.config.score))
        return {"checkpoint": str(workdir / "model.txt")}


class Validate(ValidateStage[ToyConfig]):
    def measure(self, workdir: Path, inputs: dict[str, Outputs]) -> dict[str, float]:
        return {"accuracy": float(Path(inputs["train"]["checkpoint"]).read_text())}


class Report(Stage[ToyConfig]):
    name: ClassVar[str] = "report"
    requires: ClassVar[tuple[str, ...]] = ("validate",)

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        return {"checkpoint": inputs["validate"]["checkpoint"]}


class Toy(ModelingPipeline[ToyConfig]):
    kind: ClassVar[str] = "toy-model"
    config_class = ToyConfig
    stage_classes = (Train, Validate, Report)


def test_violations() -> None:
    bounds = ValidationConfig(min={"accuracy": 0.8}, max={"ece": 0.1, "nll": 1.0})
    assert violations({"accuracy": 0.9, "ece": 0.05, "nll": 0.5}, bounds) == []
    assert violations({"accuracy": 0.7, "ece": 0.2}, bounds) == [
        "nll was not measured", "accuracy=0.7000 is below the minimum 0.8", "ece=0.2000 is above the maximum 0.1"]


def test_train_cannot_be_selected_without_validate(tmp_path: Path) -> None:
    assert Toy(ToyConfig(name="t", output_dir=str(tmp_path), stages=["train"])).plan() == ["train", "validate"]


def test_passing_validation_hands_the_checkpoint_on(tmp_path: Path) -> None:
    result = Toy(ToyConfig(name="t", output_dir=str(tmp_path), validation={"min": {"accuracy": 0.8}})).run()
    assert result["stages"]["validate"]["passed"] is True
    assert result["stages"]["report"]["checkpoint"] == result["stages"]["train"]["checkpoint"]


def test_failing_validation_stops_the_pipeline_and_writes_the_report(tmp_path: Path) -> None:
    pipeline = Toy(ToyConfig(name="t", output_dir=str(tmp_path), score=0.4, validation={"min": {"accuracy": 0.8}}))
    with pytest.raises(ValidationFailed, match="below the minimum"):
        pipeline.run()
    manifest = pipeline.experiment.manifest()
    assert set(manifest["stages"]) == {"train"}  # validate is not recorded, report never ran
    assert '"passed": false' in next(tmp_path.glob("_stages/validate-*/validation.json")).read_text()


def test_tracker_does_not_change_the_experiment(tmp_path: Path) -> None:
    plain = Toy(ToyConfig(name="t", output_dir=str(tmp_path)))
    tagged = Toy(ToyConfig(name="t", output_dir=str(tmp_path), tracker={"tags": ["x"]}))
    assert plain.experiment.key == tagged.experiment.key


def test_modeling_recipe_needs_train_and_validate() -> None:
    class NoValidate(ModelingPipeline[ToyConfig]):
        kind: ClassVar[str] = "no-validate"
        config_class = ToyConfig
        stage_classes = (Train,)

    with pytest.raises(TypeError, match="ValidateStage"):
        NoValidate.check()
    assert issubclass(ModelingPipeline, Pipeline)
