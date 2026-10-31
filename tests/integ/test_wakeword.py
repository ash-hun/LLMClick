"""`wakeword` end to end with a fake front end: windows, a real detector exported to ONNX, and the phrase-then-command test."""

import json
from pathlib import Path

import pytest
import numpy as np

from tests.recipe_inputs import wakeword_config as config, wakeword_recordings as recordings
from modeling.wakeword.stages import firings, rng_for, window
from modeling.wakeword.config import AugmentConfig
from modeling.stages import ValidationFailed
from tests.speech_fakes import FakeEmbedder
from modeling.wakeword import features
from core import pipeline

pytestmark = pytest.mark.usefixtures("wakeword_front_end")


def test_recordings_become_a_detector_that_fires_once_on_the_phrase_and_a_rerun_builds_nothing(tmp_path: Path) -> None:
    result = pipeline.build(config(recordings(tmp_path))).run()
    stages = result["stages"]
    assert stages["data"]["rows"] == {"train": {"positive": 7, "negative": 7}, "validation": {"positive": 3, "negative": 3}}
    assert stages["features"]["windows"] == {"positive": 14, "negative": 38}
    bundle = Path(stages["validate"]["checkpoint"])
    assert {path.name for path in bundle.iterdir()} == {"jarvis_wake_up.onnx", "wakeword.json", "summary.json", *features.MODELS}
    assert json.loads((bundle / "wakeword.json").read_text())["phrase"] == "Jarvis, wake up"
    assert stages["validate"]["metrics"] == {"recall": 1.0, "false_accepts": 0.0, "scenario": 1.0}
    items = [json.loads(line) for line in (Path(stages["validate"]["report"]).parent / "items.jsonl").read_text().splitlines()]
    scenario = [item for item in items if item["test"] == "scenario"]
    assert len(scenario) == 3 and all(len(item["fired_at"]) == 1 for item in scenario)

    assert json.dumps(pipeline.build(config(tmp_path / "data")).run()) == json.dumps(result)


def test_a_detector_that_fires_on_everything_fails_the_gate(tmp_path: Path) -> None:
    with pytest.raises(ValidationFailed, match="false_accepts"):
        pipeline.build(config(recordings(tmp_path), validation={"threshold": 0.001, "max": {"false_accepts": 0.0}})).run()


def test_windows_firings_and_config_errors(tmp_path: Path) -> None:
    samples = np.ones(features.RATE, dtype=np.float32) * 0.1
    augment = AugmentConfig()
    first, again = window(samples, rng_for(1, "a"), augment, [], crop=True), window(samples, rng_for(1, "a"), augment, [], crop=True)
    assert len(first) == features.WINDOW and np.array_equal(first, again)
    assert len(FakeEmbedder(tmp_path)(first)) == features.FRAMES
    assert firings(np.array([0.1, 0.9, 0.9, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9]), 0.5) == [1, 14]
    root = recordings(tmp_path)
    (root / "positive.jsonl").write_text(json.dumps({"audio": "wake/0.wav", "text": "Jarvis, look up."}) + "\n" * 2)
    with pytest.raises(ValueError, match="not the phrase"):
        pipeline.build(config(root)).run()
    (root / "positive.jsonl").write_text((json.dumps({"audio": "wake/0.wav", "text": "Jarvis, wake up."}) + "\n") * 2)
    (root / "negative.jsonl").write_text(json.dumps({"audio": "other/1.wav", "text": "Hey Jarvis, wake up now."}) + "\n" * 2)
    with pytest.raises(ValueError, match="says the phrase"):
        pipeline.build(config(root)).run()
    with pytest.raises(ValueError, match="is a range"):
        pipeline.build(config(root, augment={"snr": [30, 5]}))


def test_moved_recordings_make_the_same_windows(tmp_path: Path) -> None:
    built = [pipeline.build(config(recordings(tmp_path / where))).run()["stages"] for where in ("here", "there")]
    arrays = [np.load(stages["features"]["features"]) for stages in built]
    assert np.array_equal(arrays[0]["x"], arrays[1]["x"]) and np.array_equal(arrays[0]["y"], arrays[1]["y"])
    assert built[0]["validate"]["metrics"] == built[1]["validate"]["metrics"]
