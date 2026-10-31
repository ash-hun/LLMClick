"""`framework` on toy recipes: the runner's caching, sharing, stage selection, locking and progress, and the validation gate."""

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from modeling.modeling_stages import TrainStage, ValidateStage, ValidationFailed
from modeling.modeling_pipeline import ModelingPipeline
from modeling.modeling_config import ModelingConfig
from core.config.config_schema import BaseConfig
from core.progress_reporter import StateProgress
from core.stage_base import Outputs, Stage
from core.pipeline_runner import Pipeline

RUNS: list[str] = []


class ToyConfig(BaseConfig):
    recipe: str = "toy"
    text: str = "a"
    repeat: int = 2


class Write(Stage[ToyConfig]):
    name: ClassVar[str] = "write"
    sections: ClassVar[tuple[str, ...]] = ("text",)

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        RUNS.append(self.name)
        (workdir / "text.txt").write_text(self.config.text)
        return {"file": str(workdir / "text.txt")}


class Repeat(Stage[ToyConfig]):
    name: ClassVar[str] = "repeat"
    requires: ClassVar[tuple[str, ...]] = ("write",)
    sections: ClassVar[tuple[str, ...]] = ("repeat",)

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        RUNS.append(self.name)
        self.progress.update(1, 1, "repeating")
        return {"value": Path(inputs["write"]["file"]).read_text() * self.config.repeat}


class Toy(Pipeline[ToyConfig]):
    kind: ClassVar[str] = "toy"
    config_class = ToyConfig
    stage_classes = (Write, Repeat)


@pytest.fixture(autouse=True)
def clear() -> None:
    RUNS.clear()


def toy(tmp_path: Path, **keys: Any) -> Toy:
    return Toy(ToyConfig(name="t", output_dir=str(tmp_path), **keys))


def test_second_run_builds_nothing_and_returns_the_same(tmp_path: Path) -> None:
    first = toy(tmp_path).run()
    assert first["stages"]["repeat"] == {"value": "aa"} and RUNS == ["write", "repeat"]
    second = toy(tmp_path).run()
    assert json.dumps(second) == json.dumps(first) and RUNS == ["write", "repeat"]  # same bytes, not only equal
    assert (Path(first["directory"]) / "write" / "text.txt").read_text() == "a"  # experiment links to the stage directory


def test_experiments_share_a_stage_whose_inputs_match(tmp_path: Path) -> None:
    a, b = toy(tmp_path), toy(tmp_path, repeat=3)
    a.run()
    assert b.run()["stages"]["repeat"] == {"value": "aaa"}
    assert RUNS == ["write", "repeat", "repeat"]
    assert a.experiment.key != b.experiment.key and a.fingerprint("write") == b.fingerprint("write")


def test_stages_select_what_runs_but_not_which_experiment(tmp_path: Path) -> None:
    only_write = toy(tmp_path, stages=["write"])
    assert only_write.plan() == ["write"] and only_write.experiment.key == toy(tmp_path).experiment.key
    only_write.run()
    assert toy(tmp_path, stages=["repeat"]).plan() == ["write", "repeat"]  # what a stage reads comes along
    toy(tmp_path, stages=["repeat"]).run()
    assert RUNS == ["write", "repeat"]
    with pytest.raises(ValueError, match="Unknown stages"):
        toy(tmp_path, stages=["nope"])


def test_missing_output_file_rebuilds_the_stage(tmp_path: Path) -> None:
    first = toy(tmp_path).run()
    Path(first["stages"]["write"]["file"]).unlink()
    toy(tmp_path).run()
    assert RUNS == ["write", "repeat", "write"]


def test_failed_stage_is_not_recorded_and_reruns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    progress = StateProgress()
    monkeypatch.setattr(Repeat, "run", lambda self, workdir, inputs: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        Toy(ToyConfig(name="t", output_dir=str(tmp_path)), progress).run()
    assert progress.snapshot()["stages"] == {"write": "done", "repeat": "failed"}
    monkeypatch.undo()
    assert toy(tmp_path).run()["stages"]["repeat"] == {"value": "aa"} and RUNS == ["write", "repeat"]


def test_progress_reports_stage_states(tmp_path: Path) -> None:
    progress = StateProgress()
    Toy(ToyConfig(name="t", output_dir=str(tmp_path)), progress).run()
    state = progress.snapshot()
    assert state["stages"] == {"write": "done", "repeat": "done"} and state["note"] == "repeating"
    again = StateProgress()
    Toy(ToyConfig(name="t", output_dir=str(tmp_path)), again).run()
    assert again.snapshot()["stages"] == {"write": "cached", "repeat": "cached"}


def test_recipe_with_a_backward_dependency_is_rejected() -> None:
    class Broken(Pipeline[ToyConfig]):
        kind: ClassVar[str] = "broken"
        config_class = ToyConfig
        stage_classes = (Repeat, Write)

    with pytest.raises(TypeError, match="requires later stages"):
        Broken.check()


def test_unknown_config_keys_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Extra inputs"):
        ToyConfig(name="t", output_dir=str(tmp_path), txet="typo")


def test_concurrent_runs_keep_their_own_pipeline_logs(tmp_path: Path) -> None:
    """Two jobs in one process (the API's workers) log through the same root logger; each experiment's
    pipeline.log must hold its own records only."""
    import logging
    import threading

    started = threading.Barrier(2, timeout=5)

    def run(text: str) -> None:
        pipeline = Toy(ToyConfig(name=text, output_dir=str(tmp_path), text=text))
        pipeline.experiment.ensure()
        with pipeline.experiment.logging():
            started.wait()  # both handlers are attached before either records
            logging.getLogger("test").warning("record of %s", text)  # above the root level pytest leaves

    threads = [threading.Thread(target=run, args=(text,)) for text in ("left", "right")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for text, other in (("left", "right"), ("right", "left")):
        log = next(tmp_path.glob(f"{text}-*")) / "pipeline.log"
        assert f"record of {text}" in log.read_text() and f"record of {other}" not in log.read_text()


def test_a_run_waiting_for_a_locked_stage_says_so_and_reuses_the_result(tmp_path: Path) -> None:
    import threading
    import time

    from core.file_store import locked

    first = toy(tmp_path)
    lock = first.experiment.stage_dir("write", first.fingerprint("write"))
    holding, release = threading.Event(), threading.Event()

    def builder() -> None:
        with locked(lock.parent / f"{lock.name}.lock"):
            holding.set()
            release.wait(5)

    progress = StateProgress()
    notes: list[str] = []
    original = progress.update

    def recording(done: int, total: int | None = None, note: str = "") -> None:
        notes.append(note)
        original(done, total, note)

    progress.update = recording  # type: ignore[method-assign]
    thread = threading.Thread(target=builder)
    thread.start()
    assert holding.wait(5)
    waiter = threading.Thread(target=lambda: Toy(ToyConfig(name="t", output_dir=str(tmp_path)), progress).run())
    waiter.start()
    for _ in range(500):
        if any("waiting" in note for note in notes):
            break
        time.sleep(0.01)
    assert any("waiting for another run" in note for note in notes) and waiter.is_alive()
    release.set()
    thread.join(5)
    waiter.join(5)
    assert progress.snapshot()["stages"] == {"write": "done", "repeat": "done"}


class ToyModelConfig(ModelingConfig):
    recipe: str = "toy-model"
    score: float = 0.9


class ToyTrain(TrainStage[ToyModelConfig]):
    sections: ClassVar[tuple[str, ...]] = ("score",)

    def train(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        (workdir / "model.txt").write_text(str(self.config.score))
        return {"checkpoint": str(workdir / "model.txt")}


class ToyValidate(ValidateStage[ToyModelConfig]):
    def measure(self, workdir: Path, inputs: dict[str, Outputs]) -> dict[str, float]:
        return {"accuracy": float(Path(inputs["train"]["checkpoint"]).read_text())}


class ToyReport(Stage[ToyModelConfig]):
    name: ClassVar[str] = "report"
    requires: ClassVar[tuple[str, ...]] = ("validate",)

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        return {"checkpoint": inputs["validate"]["checkpoint"]}


class ToyModel(ModelingPipeline[ToyModelConfig]):
    kind: ClassVar[str] = "toy-model"
    config_class = ToyModelConfig
    stage_classes = (ToyTrain, ToyValidate, ToyReport)


def test_train_cannot_be_selected_without_validate(tmp_path: Path) -> None:
    assert ToyModel(ToyModelConfig(name="t", output_dir=str(tmp_path), stages=["train"])).plan() == ["train", "validate"]


def test_passing_validation_hands_the_checkpoint_on(tmp_path: Path) -> None:
    result = ToyModel(ToyModelConfig(name="t", output_dir=str(tmp_path), validation={"min": {"accuracy": 0.8}})).run()
    validated = Path(result["stages"]["validate"]["checkpoint"])
    assert result["stages"]["validate"]["passed"] is True and result["stages"]["report"]["checkpoint"] == str(validated)
    assert validated.is_symlink() and validated.resolve() == Path(result["stages"]["train"]["checkpoint"]).resolve()
    followup = Path(result["directory"]) / "validate" / "checkpoint"  # what model.init of a follow-up experiment reads
    assert followup.resolve() == validated.resolve()


def test_failing_validation_stops_the_pipeline_and_writes_the_report(tmp_path: Path) -> None:
    pipeline = ToyModel(ToyModelConfig(name="t", output_dir=str(tmp_path), score=0.4, validation={"min": {"accuracy": 0.8}}))
    with pytest.raises(ValidationFailed, match="below the minimum"):
        pipeline.run()
    manifest = pipeline.experiment.manifest()
    assert set(manifest["stages"]) == {"train"}  # validate is not recorded, report never ran
    assert '"passed": false' in next(tmp_path.glob("_stages/validate-*/validation.json")).read_text()
    assert not list(tmp_path.glob("_stages/validate-*/checkpoint"))  # no validated checkpoint to continue from


def test_tracker_does_not_change_the_experiment(tmp_path: Path) -> None:
    plain = ToyModel(ToyModelConfig(name="t", output_dir=str(tmp_path)))
    tagged = ToyModel(ToyModelConfig(name="t", output_dir=str(tmp_path), tracker={"tags": ["x"]}))
    assert plain.experiment.key == tagged.experiment.key


def test_modeling_recipe_needs_train_and_validate() -> None:
    class NoValidate(ModelingPipeline[ToyModelConfig]):
        kind: ClassVar[str] = "no-validate"
        config_class = ToyModelConfig
        stage_classes = (ToyTrain,)

    with pytest.raises(TypeError, match="ValidateStage"):
        NoValidate.check()
    assert issubclass(ModelingPipeline, Pipeline)
