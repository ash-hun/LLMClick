"""`evaluation_wakeword` end to end with a fake front end: a toy detector scored on its recordings, ambient audio and noise."""

import json
from typing import Any
from pathlib import Path

import pytest

from tests.recipe_inputs import wakeword_config, wakeword_recordings as recordings
from tests.speech_fakes import melody
from core import pipeline_runner

pytestmark = pytest.mark.usefixtures("wakeword_front_end")


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
    trained = pipeline_runner.build(wakeword_config(root, validation={"min": {}, "max": {}})).run()
    result = pipeline_runner.build(evaluation(root, trained["directory"])).run()["stages"]
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
    assert json.dumps(pipeline_runner.build(evaluation(root, trained["directory"])).run()["stages"]) == json.dumps(result)
    pipeline_runner.build(evaluation(root, trained["directory"], threshold=0.9)).run()
    assert traces.stat().st_mtime_ns == written


def test_shipped_config_validates_and_a_stage_checkpoint_needs_an_experiment() -> None:
    loaded = pipeline_runner.load("configs/evaluation/wakeword.yaml")
    assert loaded.config.recipe == "evaluation_wakeword" and list(loaded.stages) == ["clips", "traces", "report"]
    with pytest.raises(ValueError, match="needs model.experiment"):
        pipeline_runner.build({"pipeline": {"recipe": "evaluation_wakeword", "name": "x"}, "model": {"checkpoint": "validate"},
                        "data": {"positive": [{"name": "local_jsonl", "path": "x.jsonl"}]}})
