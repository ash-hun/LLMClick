"""`wakeword` with a fake front end: windows, a real detector exported to ONNX, and the phrase-then-command test."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from core import pipeline
from modeling.wakeword import detector, features
from modeling.wakeword.config import AugmentConfig, WakewordTraining
from modeling.wakeword.stages import firings, rng_for, window
from modeling.stages import ValidationFailed
from tests.speech import FakeEmbedder, fake_feature_models, melody

# "Jarvis" then "wake up" as two notes; the negatives are other notes, and the first note alone (the name alone)
NAME, WAKE, OTHERS = 880, 1320, [300, 440, 1500, 2200]


@pytest.fixture(autouse=True)
def front_end(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("modeling.wakeword.features.ensure", fake_feature_models)
    monkeypatch.setattr("modeling.wakeword.features.Embedder", FakeEmbedder)


def recordings(root: Path) -> Path:
    """Two sets of rows.jsonl beside their wavs, as data_speech writes them; outside output/, which the signature of
    their directory would otherwise include."""
    root = root / "data"
    positive, negative = [], []
    for index in range(10):
        melody(root / "wake" / f"{index}.wav", [(NAME, 0.4), (WAKE, 0.4 + 0.05 * index)])
        positive.append({"audio": f"wake/{index}.wav", "text": "Jarvis, wake up."})
        if index % 3:
            melody(root / "other" / f"{index}.wav", [(OTHERS[index % len(OTHERS)], 1.0 + 0.1 * index)])
            negative.append({"audio": f"other/{index}.wav", "text": "Please search the web."})
        else:
            melody(root / "other" / f"{index}.wav", [(NAME, 0.5 + 0.05 * index)])
            negative.append({"audio": f"other/{index}.wav", "text": "Jarvis."})
    for name, rows in (("positive", positive), ("negative", negative)):
        (root / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return root


def config(root: Path, **keys: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "pipeline": {"recipe": "wakeword", "name": "wake", "seed": 7, "output_dir": str(root.parent / "output")},
        "phrase": "Jarvis, wake up",
        "data": {"positive": [{"name": "local_jsonl", "path": str(root / "positive.jsonl")}],
                 "negative": [{"name": "local_jsonl", "path": str(root / "negative.jsonl")}], "validation": 0.3},
        "augment": {"rounds": 2, "noise": 10, "snr": [20, 30]},
        "training": {"steps": 150, "batch_size": 32},
        "validation": {"min": {"recall": 1.0, "scenario": 1.0}, "max": {"false_accepts": 0.0}},
    }
    return {**base, **keys}


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


def test_weighting_negatives_lowers_their_scores_and_the_cosine_schedule_trains(tmp_path: Path) -> None:
    import onnxruntime
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, features.FRAMES, features.DIM)).astype(np.float32)
    y = (rng.random(200) < 0.5).astype(np.float32)
    x[y == 1] += 0.1  # two classes that overlap, so the weight decides where the boundary falls

    def negative_scores(**training: Any) -> float:
        onnx = tmp_path / f"{sorted(training.items())}.onnx"
        detector.fit(x, y, WakewordTraining(steps=60, batch_size=32, **training), 7, onnx, lambda step: None)
        scores = onnxruntime.InferenceSession(str(onnx)).run(None, {"input": x[y == 0]})[0]
        return float(scores.mean())

    assert negative_scores(negative_weight=30.0) < negative_scores(negative_weight=1.0)
    assert 0 < negative_scores(schedule="cosine") < 1
