"""data -> benchmarks -> mix on local fixtures, twice: the second run must change nothing."""

import json
from pathlib import Path
from typing import Any

from modeling.jev.tuning import trainer
from core.utils.files import sha256_file
from core.progress import StateProgress
from core import pipeline

STAGES = ["data", "benchmarks", "mix"]


def build(raw: dict[str, Any], **keys: Any) -> pipeline.Pipeline[Any]:
    return pipeline.build({**raw, "pipeline": {**raw["pipeline"], "stages": STAGES, **keys}})


def snapshot(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "pipeline.log" and p.suffix != ".lock"}


def test_local_stages_run_and_are_idempotent(local_config: dict, fixture_dir: Path) -> None:
    first = build(local_config).run()
    assert first["stages"]["data"]["rows"] == {"train": 45, "dev": 6, "temperature": 4, "validation": 5}
    assert Path(first["stages"]["mix"]["train"]).exists()
    assert first["stages"]["mix"]["leaks"]["public"] == 0
    before = snapshot(fixture_dir / "output")
    assert build(local_config).run() == first
    assert snapshot(fixture_dir / "output") == before
    manifest = json.loads((Path(first["directory"]) / "manifest.json").read_text())
    assert set(manifest["stages"]) == set(STAGES) | {"synthetic"}  # mix reads synthetic, which is {} when disabled


def test_training_change_reuses_the_data_stages(local_config: dict) -> None:
    a = build(local_config)
    b = build({**local_config, "training": {**local_config["training"], "lr": 1e-5}})
    assert a.experiment.key != b.experiment.key
    assert [a.fingerprint(s) for s in STAGES] == [b.fingerprint(s) for s in STAGES]
    assert a.fingerprint("train") != b.fingerprint("train")


def test_edited_local_file_is_a_new_data_stage(local_config: dict, fixture_dir: Path) -> None:
    before = build(local_config).fingerprint("data")
    train = fixture_dir / "train.jsonl"
    train.write_text(train.read_text().replace("needs attention", "needs help"))
    assert build(local_config).fingerprint("data") != before


def test_full_plan_ends_with_validate_then_evaluate(local_config: dict) -> None:
    assert pipeline.build(local_config).plan()[-3:] == ["train", "validate", "evaluate"]
    assert build(local_config, stages=["train"]).plan()[-2:] == ["train", "validate"]


def test_train_argv_mirrors_training_config(local_config: dict) -> None:
    config = pipeline.build(local_config).config
    args = trainer.argv(config.training, config.model, Path("t"), Path("d"), Path("c"), Path("run"), Path("out"), 7)
    joined = " ".join(args)
    assert "--base-model Qwen/Qwen3.5-0.8B" in joined and "--eval-every 10" in joined and "--lr 5e-06" in joined
    assert "--patience" not in joined


def test_training_events_become_progress(tmp_path: Path) -> None:
    events, progress = tmp_path / "events.jsonl", StateProgress()
    tail = trainer.TrainingEvents(events, progress)
    tail.poll()  # no file yet
    events.write_text(json.dumps({"kind": "training_started", "step": 0, "total_steps": 8}) + "\n"
                      + json.dumps({"kind": "training_step", "step": 3, "loss": 0.5}) + "\n" + '{"kind": "trai')
    tail.poll()
    assert (progress.state["done"], progress.state["total"], progress.state["note"]) == (3, 8, "loss 0.5000")
    with events.open("a") as stream:
        stream.write('ning_step", "step": 4, "loss": 0.25}\n')
    tail.poll()
    assert progress.state["done"] == 4


def test_pilot_progress_ends_at_stop_after(tmp_path: Path) -> None:
    events, progress = tmp_path / "events.jsonl", StateProgress()
    events.write_text(json.dumps({"kind": "training_started", "step": 0, "total_steps": 18}) + "\n")
    trainer.TrainingEvents(events, progress, limit=4).poll()
    assert progress.state["total"] == 4
