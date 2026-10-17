"""`data_speech` end to end with a fake Piper and a fake Whisper: cached, resumable, and the same twice."""

import json
from pathlib import Path
from typing import Any

import pytest

from core import pipeline
from core.progress import StateProgress
from data.speech.select import chosen
from data.speech.synthesize import spoken_text
from tests.speech import FakeListener, FakePiper


@pytest.fixture
def piper(monkeypatch: pytest.MonkeyPatch) -> FakePiper:
    fake = FakePiper()
    monkeypatch.setattr("modeling.tts.piper.speak", fake.speak)
    monkeypatch.setattr("modeling.tts.piper.version", lambda: "test")
    monkeypatch.setattr("data.speech.verify.Listener", FakeListener(fake))
    return fake


def config(out: Path, **keys: Any) -> dict[str, Any]:
    voice = out / "voice.onnx"
    voice.parent.mkdir(parents=True, exist_ok=True)
    voice.write_text("a voice")
    voice.with_name("voice.onnx.json").write_text("{}")
    base: dict[str, Any] = {
        "pipeline": {"recipe": "data_speech", "name": "speech", "seed": 3, "output_dir": str(out / "output")},
        "seeds": [{"kind": "line", "values": ["Good evening, sir.", "All systems are online, sir.", "good evening,  sir.",
                                             "Please mumble the forecast for tomorrow.", "Hi.",
                                             "A loud warning about the reactor core temperature."]}],
        "prompts": {"generator": "passthrough"},
        "voice": {"model": {"name": str(voice)}, "samples": 2},
        "verify": {"min_seconds": 0.5},
        "licence": "test voice, test use only",
    }
    return {**base, **keys}


def test_texts_become_a_checked_piper_dataset_and_a_rerun_builds_nothing(tmp_path: Path, piper: FakePiper) -> None:
    result = pipeline.build(config(tmp_path)).run()
    stages = result["stages"]
    assert stages["prompts"]["count"] == 6 and stages["prompts"]["generated_now"] == 6  # passthrough needs no teacher
    assert stages["synthesize"]["count"] == 12 and len(piper.calls) == 12
    assert stages["verify"]["reasons"] == {"too short": 2, "misheard": 2, "clipped": 2}
    # the two evening lines differ only in case and spacing: one text, the better recording of the two kept
    assert stages["select"]["counts"] == {"verified": 6, "duplicate_text": 4, "over_max_rows": 0, "rows": 2, "seconds": 1.6}
    dataset = Path(stages["select"]["rows"]).parent
    lines = (dataset / "metadata.csv").read_text().splitlines()
    assert len(lines) == 2 and all((dataset / line.split("|")[0]).is_file() for line in lines)
    rows = [json.loads(line) for line in (dataset / "rows.jsonl").read_text().splitlines()]
    assert {row["text"].lower() for row in rows} == {"good evening, sir.", "all systems are online, sir."}
    manifest = json.loads((dataset / "manifest.json").read_text())
    assert manifest["licence"] == "test voice, test use only" and manifest["teacher"] is None
    assert set(manifest["stages"]) == {"seeds", "prompts", "synthesize", "verify", "select"}

    piper.calls = []
    again = StateProgress()
    second = pipeline.build(config(tmp_path), again).run()
    assert json.dumps(second) == json.dumps(result) and piper.calls == []
    assert set(again.snapshot()["stages"].values()) == {"cached"}


def test_a_crash_mid_synthesis_resumes_without_reading_a_text_twice(tmp_path: Path, piper: FakePiper) -> None:
    piper.fail_after = 5
    with pytest.raises(RuntimeError, match="went away"):
        pipeline.build(config(tmp_path)).run()
    piper.calls, piper.fail_after = [], None
    result = pipeline.build(config(tmp_path)).run()
    assert len(piper.calls) == 7 and result["stages"]["synthesize"]["spoken_now"] == 7


def test_a_changed_threshold_reverifies_without_speaking_or_transcribing_again(tmp_path: Path, piper: FakePiper,
                                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline.build(config(tmp_path)).run()
    listener = FakeListener(piper)
    monkeypatch.setattr("data.speech.verify.Listener", listener)
    piper.calls = []
    result = pipeline.build(config(tmp_path, verify={"min_seconds": 0.1, "max_clipped": 1.0, "max_wer": 2.0})).run()
    assert piper.calls == [] and result["stages"]["verify"]["count"] == 12
    assert listener.calls == 12  # a new verify stage: its own transcript cache, filled once


def test_a_line_seed_is_read_as_it_is_by_every_speaker_and_all_recordings_kept(tmp_path: Path, piper: FakePiper) -> None:
    keys = {"seeds": [{"kind": "line", "values": ["Jarvis, wake up."]}], "prompts": {"generator": "utterances"},
            "voice": {"model": {"name": str(tmp_path / "voice.onnx")}, "speakers": [3, 7, 11]}, "select": {"per_text": None}}
    result = pipeline.build(config(tmp_path, **keys)).run()  # utterances asks no teacher for a line
    assert sorted(piper.speakers) == [3, 7, 11] and piper.calls == ["Jarvis, wake up."] * 3
    assert result["stages"]["select"]["counts"]["rows"] == 3


def test_choosing_and_cleaning_texts(tmp_path: Path, piper: FakePiper) -> None:
    assert spoken_text(" a |b\n c ") == "a ,b c"
    rows = [{"id": "2", "text": "Hi there", "wer": 0.5, "sample": 0}, {"id": "1", "text": "hi  there", "wer": 0.0, "sample": 1},
            {"id": "3", "text": "Other", "wer": 0.0, "sample": 0}]
    assert [row["id"] for row in chosen(rows)] == ["1", "3"]
    assert [row["id"] for row in chosen(rows, None)] == ["1", "2", "3"]
    with pytest.raises(ValueError, match="set `teacher`"):
        pipeline.build(config(tmp_path, seeds=[{"kind": "situation", "values": ["a security alert"]}],
                              prompts={"generator": "utterances"})).run()
