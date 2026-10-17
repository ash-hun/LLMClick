"""`evaluation_wakeword` with a fake front end: a toy detector scored on its recordings, ambient audio and noise."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evaluation.wakeword import CHUNK, concatenated, hit, poisson, wilson
from tests.modeling.test_wakeword import config as wakeword_config, recordings
from modeling.wakeword import features
from tests.speech import melody
from core import pipeline


@pytest.fixture(autouse=True)
def front_end(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.speech import FakeEmbedder, fake_feature_models
    monkeypatch.setattr("modeling.wakeword.features.ensure", fake_feature_models)
    monkeypatch.setattr("modeling.wakeword.features.Embedder", FakeEmbedder)


def evaluation(root: Path, experiment: str, **keys: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "pipeline": {"recipe": "evaluation_wakeword", "name": "wake-eval", "seed": 3, "output_dir": str(root.parent / "output")},
        "model": {"experiment": experiment, "checkpoint": "validate"},
        "data": {"positive": [{"name": "local_jsonl", "path": str(root / "positive.jsonl")}],
                 "negative": [{"name": "local_jsonl", "path": str(root / "negative.jsonl")}],
                 "ambient": [str(root / "ambient")]},
        "conditions": [None, 10],
    }
    return {**base, **keys}


def test_a_detector_is_scored_on_recordings_and_ambient_audio_and_a_new_threshold_only_rereads_the_scores(tmp_path: Path) -> None:
    root = recordings(tmp_path)
    for index in range(4):  # 40 s without the phrase: other notes only
        melody(root / "ambient" / f"{index}.wav", [(440 + 200 * (index % 3), 4.0), (1500, 3.0), (300, 3.0)])
    trained = pipeline.build(wakeword_config(root, validation={"min": {}, "max": {}})).run()
    result = pipeline.build(evaluation(root, trained["directory"])).run()["stages"]
    report = json.loads(Path(result["report"]["report"]).read_text())

    assert report["data"]["positive"] == 10 and report["data"]["negative"] == 10 and report["data"]["ambient"] == 4
    assert report["data"]["ambient_hours"] == pytest.approx(40 / 3600)
    assert report["settings"]["conditions"] == ["clean", "snr10"] and report["model"]["phrase"] == "Jarvis, wake up"
    assert len(report["sweep"]) == 99 and set(report["at_targets"]) == {"0.5", "0.1"}
    clean = report["operating_point"]["conditions"]["clean"]
    assert clean["positives"] == 10 and 0 <= clean["frr_interval"][0] <= clean["frr"] <= clean["frr_interval"][1] <= 1
    assert {"frr_clean", "frr_snr10", "fa_per_hour", "frr_clean_at_0.5_fa_per_hour"} <= set(result["report"]["scores"])
    frr = [point["frr"]["clean"] for point in report["sweep"]]
    assert frr == sorted(frr)  # a higher threshold never hears more
    texts = {entry["text"] for entry in report["operating_point"]["negatives_by_text"]}
    assert texts == {"Please search the web.", "Jarvis."}

    traces = Path(result["traces"]["traces"])
    written = traces.stat().st_mtime_ns
    assert json.dumps(pipeline.build(evaluation(root, trained["directory"])).run()["stages"]) == json.dumps(result)
    pipeline.build(evaluation(root, trained["directory"], threshold=0.9)).run()
    assert traces.stat().st_mtime_ns == written


def test_intervals_hits_and_ambient_chunks() -> None:
    assert wilson(0, 10)[0] == 0 and wilson(10, 10)[1] == 1 and wilson(5, 10)[0] < 0.5 < wilson(5, 10)[1]
    assert poisson(0, 10)[0] == 0 and poisson(0, 10)[1] == pytest.approx(0.369, abs=0.01)  # exact: 3.689 / 10 h
    assert poisson(10, 1)[0] == pytest.approx(4.80, abs=0.05) and poisson(10, 1)[1] == pytest.approx(18.39, abs=0.1)
    scores = np.zeros(80, dtype=np.float32)
    first = next(i for i in range(80) if features.window_end(i) / features.RATE >= 2.0 + 0.5)  # 0.5 s phrase starting at 2 s
    scores[first] = 0.9
    assert hit(scores, 0.5, 0.5, 1.0) == pytest.approx(features.window_end(first) / features.RATE - 2.5)
    assert hit(scores, 0.5, 0.95, 1.0) is None


def test_ambient_chunks_overlap_by_a_window(tmp_path: Path) -> None:
    melody(tmp_path / "a.wav", [(440, 50.0)])
    melody(tmp_path / "b.wav", [(880, 30.0)])
    whole = np.concatenate([features.clip(tmp_path / "a.wav"), features.clip(tmp_path / "b.wav")])
    chunks = list(concatenated([str(tmp_path / "a.wav"), str(tmp_path / "b.wav")]))
    assert [len(chunk) for chunk in chunks[:-1]] == [CHUNK + features.WINDOW - features.HOP]
    for index, chunk in enumerate(chunks):
        assert np.array_equal(chunk, whole[index * CHUNK: index * CHUNK + len(chunk)])
    assert len(chunks) * CHUNK >= len(whole) - features.WINDOW


def test_shipped_config_validates_and_a_stage_checkpoint_needs_an_experiment() -> None:
    loaded = pipeline.load("configs/evaluation/wakeword.yaml")
    assert loaded.config.recipe == "evaluation_wakeword" and list(loaded.stages) == ["clips", "traces", "report"]
    with pytest.raises(ValueError, match="needs model.experiment"):
        pipeline.build({"pipeline": {"recipe": "evaluation_wakeword", "name": "x"}, "model": {"checkpoint": "validate"},
                        "data": {"positive": [{"name": "local_jsonl", "path": "x.jsonl"}]}})
