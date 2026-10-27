"""`tts_piper` with a fake Piper and a fake Whisper: recordings found, training resumed, the voice gated by listening."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from core import pipeline
from modeling.stages import ValidationFailed
from modeling.tts import audio, quality
from modeling.tts.asr import wer
from modeling.tts.stages import latest
from modeling.tuning.sources import SOURCES, delimited
from modeling.config import Keyed, local_paths
from tests.speech import FakeListener, FakePiper, FakeRater, tone

TEXTS = ["Good evening, sir.", "All systems are online.", "Shall I run the diagnostics?", "The suit is ready.",
         "Accessing the alarm settings.", "You have three new messages."]


@pytest.fixture
def piper(monkeypatch: pytest.MonkeyPatch) -> FakePiper:
    fake = FakePiper()
    for name in ("speak", "fit", "export"):
        monkeypatch.setattr(f"modeling.tts.piper.{name}", getattr(fake, name))
    monkeypatch.setattr("modeling.tts.piper.version", lambda: "test")
    monkeypatch.setattr("modeling.tts.stages.Listener", FakeListener(fake, heard_as="a recording"))
    monkeypatch.setattr("modeling.tts.stages.Rater", lambda directory: FakeRater(fake))
    return fake


def recordings(root: Path) -> Path:
    """An LJSpeech-style set: wavs/<id>.wav and a metadata.csv of `<id>|text` lines."""
    for index, text in enumerate(TEXTS):
        tone(root / "wavs" / f"utt{index}.wav", 0.2 * len(text.split()))
    (root / "metadata.csv").write_text("".join(f"utt{index}|{text}\n" for index, text in enumerate(TEXTS)))
    (root / "start.ckpt").write_text("pretrained")
    return root


def config(root: Path, **keys: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "pipeline": {"recipe": "tts_piper", "name": "voice", "seed": 5, "output_dir": str(root / "output"), "device": "cpu"},
        "model": {"espeak_voice": "en-gb-x-rp", "init": {"name": str(root / "start.ckpt")}},
        "data": {"sources": [{"name": "delimited", "path": str(root / "metadata.csv"), "audio_dir": str(root / "wavs")}],
                 "validation": 0.34},
        "training": {"epochs": 3, "batch_size": 2, "args": {"model.learning_rate": 1.0e-4}},
        "validation": {"max": {"wer": 0.1}},
    }
    return {**base, **keys}


def test_recordings_train_export_and_pass_the_listening_test_and_a_rerun_builds_nothing(tmp_path: Path, piper: FakePiper) -> None:
    result = pipeline.build(config(recordings(tmp_path))).run()
    stages = result["stages"]
    assert stages["data"]["rows"] == {"train": 4, "validation": 2}
    fit = piper.fits[0]
    assert fit["model.warmstart_ckpt"] == tmp_path / "start.ckpt" and fit["resume"] is None
    assert fit["trainer.max_epochs"] == 3 and fit["trainer.accelerator"] == "cpu" and fit["model.learning_rate"] == 1.0e-4
    csv = Path(fit["data.csv_path"]).read_text().splitlines()
    assert len(csv) == 4 and all(Path(line.split("|")[0]).is_absolute() for line in csv)
    checkpoint = Path(stages["validate"]["checkpoint"])
    assert (checkpoint / "voice.onnx").read_text() == "onnx of epochs=3" and (checkpoint / "voice.onnx.json").is_file()
    metrics = stages["validate"]["metrics"]
    assert metrics["wer"] == 0.0 and metrics["reference_wer"] == 1.0 and metrics["duration_ratio"] == pytest.approx(1.0)
    assert (metrics["utmos"], metrics["dnsmos_bak"], metrics["reference_utmos"], metrics["reference_dnsmos_bak"]) == (4.0, 4.0, 3.0, 3.0)

    piper.fits, piper.calls = [], []
    assert json.dumps(pipeline.build(config(tmp_path)).run()) == json.dumps(result)
    assert piper.fits == [] and piper.calls == []


def test_a_crash_after_fitting_resumes_from_the_last_snapshot(tmp_path: Path, piper: FakePiper, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(checkpoint: Path, onnx: Path, log: Path) -> None:
        raise RuntimeError("export died")

    monkeypatch.setattr("modeling.tts.piper.export", broken)
    with pytest.raises(RuntimeError, match="export died"):
        pipeline.build(config(recordings(tmp_path))).run()
    monkeypatch.setattr("modeling.tts.piper.export", piper.export)
    pipeline.build(config(tmp_path)).run()
    run = Path(piper.fits[0]["trainer.default_root_dir"])
    assert len(piper.fits) == 2 and piper.fits[1]["resume"] == run / "lightning_logs" / "version_0" / "checkpoints" / "last.ckpt"
    assert latest(run) == run / "lightning_logs" / "version_1" / "checkpoints" / "last.ckpt"


def test_a_voice_that_mumbles_fails_validation(tmp_path: Path, piper: FakePiper) -> None:
    root = recordings(tmp_path)
    (root / "metadata.csv").write_text("".join(f"utt{index}|{text} mumble\n" for index, text in enumerate(TEXTS)))
    with pytest.raises(ValidationFailed, match="wer"):
        pipeline.build(config(root)).run()


def test_the_agents_fixed_replies_keep_their_best_take_and_must_be_understood(tmp_path: Path, piper: FakePiper,
                                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    replies = {"wake_up": "At your service, sir.", "web_search": "Searching Reference."}
    result = pipeline.build(config(recordings(tmp_path), validation={"phrases": replies, "max": {"wer": 0.1, "phrase_wer": 0.0}})).run()
    report = Path(result["stages"]["validate"]["report"]).parent
    kept = json.loads((report / "phrases.json").read_text())
    assert {name: entry["text"] for name, entry in kept.items()} == replies and (report / "phrases" / "web_search.wav").is_file()
    assert result["stages"]["validate"]["metrics"]["phrase_wer"] == 0.0 and result["stages"]["validate"]["metrics"]["phrase_dnsmos_sig"] == 4.0
    assert kept["wake_up"]["utmos"] == 4.0 and all("dnsmos_ovrl" in take for take in kept["wake_up"]["takes"])
    with pytest.raises(ValidationFailed, match="phrase_wer"):
        pipeline.build(config(tmp_path, validation={"phrases": {"web_search": "Searching mumble."}, "max": {"phrase_wer": 0.0}})).run()
    first_take_garbled = FakeListener(piper, heard_as="a recording")
    hears = first_take_garbled.transcribe
    first_take_garbled.transcribe = lambda path: "garbled" if path.name.endswith("-0.wav") else hears(path)  # type: ignore[method-assign]
    monkeypatch.setattr("modeling.tts.stages.Listener", first_take_garbled)
    again = pipeline.build(config(tmp_path, validation={"phrases": replies, "phrase_takes": 2, "max": {"phrase_wer": 0.0}})).run()
    picked = json.loads((Path(again["stages"]["validate"]["report"]).parent / "phrases.json").read_text())
    assert {entry["take"] for entry in picked.values()} == {1} and [take["heard"] for take in picked["web_search"]["takes"]] == ["garbled", "Searching Reference."]
    with pytest.raises(ValueError, match="may only use letters"):
        pipeline.build(config(tmp_path, validation={"phrases": {"web search": "Searching Reference."}}))


def test_recordings_of_a_source_are_found_on_disk(tmp_path: Path) -> None:
    root = recordings(tmp_path)
    source = Keyed(name="delimited", path=str(root / "metadata.csv"), audio_dir=str(root / "wavs"))
    found = list(audio.recordings(source))
    assert [number for number, _, _ in found] == list(range(len(TEXTS))) and all(path and path.is_file() for _, _, path in found)
    (root / "metadata.csv").write_text("utt0|ok\nmissing|no file\n")
    assert [path is None for _, _, path in audio.recordings(source)] == [False, True]
    assert audio.signatures([source, Keyed(name="huggingface")])[1] is None
    assert local_paths([source], audio.AUDIO_KEYS) == [str(root / "metadata.csv"), str(root / "wavs")]
    with pytest.raises(ValueError, match="Unknown source 'nowhere'; registered: \\['delimited'"):
        SOURCES.check("delimited", "nowhere")


def test_config_errors_and_units(tmp_path: Path) -> None:
    root = recordings(tmp_path)
    with pytest.raises(ValueError, match="training.args cannot set"):
        pipeline.build(config(root, training={"args": {"trainer.max_epochs": 9}}))
    with pytest.raises(ValueError, match="40-character commit"):
        pipeline.build(config(root, model={"espeak_voice": "en-us", "init": {"name": "jgkawell/jarvis", "revision": "main", "file": "a.ckpt"}}))
    (root / "metadata.csv").write_text("missing|No such file.\n" * 3)
    with pytest.raises(ValueError, match="needs `text` and an `audio` file"):
        pipeline.build(config(root)).run()
    assert delimited({"path": str(root / "metadata.csv"), "limit": 1}) == [{"audio": "missing", "text": "No such file."}]
    assert wer("the suit is ready", "the suit ready now") == 0.5 and wer("", "") == 0.0
    samples, rate = audio.read(root / "wavs" / "utt0.wav")
    assert len(audio.resample(samples, rate, 16000)) == round(len(samples) * 16000 / rate)
    measured = audio.measure(np.zeros(rate, dtype=np.float32), rate)
    assert measured == {"seconds": 1.0, "clipped": 0.0, "silent": 1.0}


def test_dnsmos_cuts_windows_as_microsofts_script_and_calibrates_its_outputs() -> None:
    second = np.arange(quality.RATE, dtype=np.float32)
    windows = quality.dnsmos_windows(second)  # doubled to 16 s, then 9.01 s windows one second apart
    assert windows.shape == (7, quality.DNSMOS_WINDOW) and windows[1][0] == 0 and windows[1][-1] == second[quality.DNSMOS_WINDOW % quality.RATE - 1]
    assert len(quality.dnsmos_windows(np.zeros(30 * quality.RATE, dtype=np.float32))) == 7  # later windows end a sample short
    raw = np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
    assert quality.calibrated(raw)["dnsmos_bak"] == pytest.approx(np.mean([-0.13166888 * x * x + 1.60915514 * x - 0.39604546 for x in (3, 4)]))
