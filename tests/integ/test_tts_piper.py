"""`tts_piper` end to end with a fake Piper and a fake Whisper: recordings found, training resumed, the voice gated by listening."""

import json
from pathlib import Path

import pytest
import numpy as np

from tests.recipe_inputs import TEXTS, tts_config as config, tts_recordings as recordings
from tests.speech_fakes import FakeListener, FakePiper
from modeling.tuning.sources import delimited
from modeling.stages import ValidationFailed
from modeling.tts.stages import latest
from modeling.tts.asr import wer
from modeling.tts import audio
from core import pipeline


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
    assert (metrics["speaker_similarity"], metrics["reference_speaker_similarity"]) == (0.9, 1.0)

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


def test_a_reply_filter_is_kept_where_whisper_still_hears_the_words(tmp_path: Path, piper: FakePiper,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_filter(source: Path, target: Path, chain: str) -> None:
        target.write_bytes(source.read_bytes())
        text = piper.spoken[str(source.resolve())]  # a filter that blurs names: Whisper hears the rest as said
        piper.spoken[str(source.with_suffix(".filtered.wav").resolve())] = "Change to job is." if "JARVIS" in text else text

    monkeypatch.setattr("modeling.tts.audio.filtered", fake_filter)
    replies = {"wake_up": "At your service, sir.", "persona_jarvis": "Change to JARVIS."}
    result = pipeline.build(config(recordings(tmp_path), validation={"phrases": replies, "phrase_takes": 2,
                                                                      "phrase_filter": "afftdn=nf=-20", "max": {"phrase_wer": 0.0}})).run()
    kept = json.loads((Path(result["stages"]["validate"]["report"]).parent / "phrases.json").read_text())
    assert (kept["wake_up"]["filtered"], kept["wake_up"]["file"]) == (True, "wake_up-0.filtered.wav")
    assert (kept["persona_jarvis"]["filtered"], kept["persona_jarvis"]["heard"]) == (False, "Change to JARVIS.")
    assert [(take["take"], take["filtered"]) for take in kept["wake_up"]["takes"]] == [(0, False), (0, True), (1, False), (1, True)]
