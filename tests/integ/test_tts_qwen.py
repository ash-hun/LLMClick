"""`tts_qwen` end to end with a fake Qwen3-TTS and a fake Whisper: the most natural clip cloned, the voice gated by listening."""

import json
from typing import Any
from pathlib import Path

import pytest

from tests.recipe_inputs import tts_config as config, tts_recordings as recordings
from tests.speech_fakes import FakePiper, FakeRater
from modeling.tts import audio_file, qwen_process
from core import pipeline_runner


class FakeQwen:
    """`speak` reads every job through FakePiper, so FakeListener hears it, and remembers each job's seed."""

    def __init__(self, piper: FakePiper) -> None:
        self.piper = piper
        self.seeds: dict[str, int] = {}
        self.voices: list[dict[str, str]] = []

    def speak(self, voice: dict[str, str], reference: Path, jobs: list[tuple[str, Path, int]], log: Path) -> None:
        assert reference.is_file()
        self.voices.append(voice)
        for text, wav, seed in jobs:
            if not wav.is_file():
                self.piper.speak(Path("qwen"), text, wav, log, {})
                self.seeds[wav.name] = seed


def qwen_config(root: Path, **keys: Any) -> dict[str, Any]:
    base = config(root, **keys)
    del base["model"], base["training"]
    return {**base, "pipeline": {**base["pipeline"], "recipe": "tts_qwen"}, **keys}


def test_qwen_clones_the_most_natural_clip_in_bounds_and_a_rerun_speaks_nothing(tmp_path: Path, piper: FakePiper,
                                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeQwen(piper)
    monkeypatch.setattr("modeling.tts.qwen_process.speak", fake.speak)
    monkeypatch.setattr("modeling.tts.qwen_process.version", lambda: "test")
    rated = FakeRater(piper)
    # the longer a recording the more natural, so the pick is the longest one inside the bounds
    monkeypatch.setattr("modeling.tts.tts_stages.Rater", lambda directory: (
        lambda path: {**rated(path), "utmos": audio_file.seconds(path)} if str(path.resolve()) not in piper.spoken else rated(path)))
    replies = {"wake_up": "At your service, sir.", "persona_jarvis": "Change to JARVIS."}
    settings = qwen_config(recordings(tmp_path), reference={"min_seconds": 0.5, "max_seconds": 0.85},
                           validation={"phrases": replies, "phrase_takes": 2, "max": {"wer": 0.1, "phrase_wer": 0.0},
                                       "min": {"speaker_similarity": 0.8}})
    result = pipeline_runner.build(settings).run()
    checkpoint = Path(result["stages"]["validate"]["checkpoint"])
    voice = json.loads((checkpoint / "voice.json").read_text())
    # the first of the three 0.8 s clips: a tie keeps the rows' order
    assert voice["reference_text"] == "All systems are online." and voice["model"].startswith("mlx-community/Qwen3-TTS")
    assert result["stages"]["train"]["seconds"] == pytest.approx(0.8) and (checkpoint / "reference.wav").is_file()
    metrics = result["stages"]["validate"]["metrics"]
    assert metrics["wer"] == 0.0 and metrics["phrase_wer"] == 0.0 and metrics["speaker_similarity"] == 0.9
    assert fake.seeds["wake_up-0.wav"] != fake.seeds["wake_up-1.wav"] and fake.seeds["wake_up-0.wav"] == qwen_process.seed(5, replies["wake_up"], 0)

    piper.calls = []
    assert json.dumps(pipeline_runner.build(settings).run()) == json.dumps(result) and piper.calls == []
    with pytest.raises(ValueError, match="No training recording lasts"):
        pipeline_runner.build(qwen_config(tmp_path, reference={"min_seconds": 5, "max_seconds": 9})).run()
    with pytest.raises(ValueError, match="below reference.max_seconds"):
        pipeline_runner.build(qwen_config(tmp_path, reference={"min_seconds": 9, "max_seconds": 5}))
