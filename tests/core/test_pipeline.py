"""The runner on a toy recipe: caching, sharing between experiments, stage selection, locking and progress."""

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from core.config.schema import BaseConfig
from core.pipeline import Pipeline
from core.progress import StateProgress
from core.stage import Outputs, Stage

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
