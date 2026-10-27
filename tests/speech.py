"""Stand-ins for the Piper program, Whisper and openWakeWord's front end, so the speech recipes run in tests without them."""

import wave
from pathlib import Path
from typing import Any

import numpy as np

RATE = 22050


def tone(path: Path, seconds: float, amplitude: float = 0.3, frequency: float = 440) -> None:
    melody(path, [(frequency, seconds)], amplitude)


def melody(path: Path, notes: list[tuple[float, float]], amplitude: float = 0.3) -> None:
    """(frequency, seconds) notes one after another: a stand-in for words, whose order matters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [np.sin(np.arange(int(RATE * seconds)) * frequency * 2 * np.pi / RATE) for frequency, seconds in notes]
    samples = (amplitude * np.concatenate(parts) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        stream.writeframes(samples.tobytes())


class FakePiper:
    """`speak` writes a tone a fifth of a second per word and remembers the text, so FakeListener can hear it;
    `fit` leaves Lightning's files behind, `export` a voice."""

    def __init__(self) -> None:
        self.spoken: dict[str, str] = {}
        self.calls: list[str] = []
        self.speakers: list[int | None] = []
        self.fits: list[dict[str, Any]] = []
        self.fail_after: int | None = None

    def speak(self, voice: Path, text: str, wav: Path, log: Path, scales: dict[str, float], speaker: int | None = None) -> None:
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("piper went away")
        self.calls.append(text)
        self.speakers.append(speaker)
        tone(wav, 0.2 * len(text.split()), amplitude=1.0 if "loud" in text else 0.3)
        self.spoken[str(wav.resolve())] = text

    def fit(self, values: dict[str, Any], resume: Path | None, log: Path) -> None:
        self.fits.append({**values, "resume": resume})
        run = Path(values["trainer.default_root_dir"])
        version = run / "lightning_logs" / f"version_{len(list(run.glob('lightning_logs/version_*')))}" / "checkpoints"
        version.mkdir(parents=True)
        (version / "last.ckpt").write_text(f"epochs={values['trainer.max_epochs']}")
        Path(values["data.config_path"]).write_text('{"audio": {"sample_rate": 22050}}')

    def export(self, checkpoint: Path, onnx: Path, log: Path) -> None:
        onnx.write_text("onnx of " + checkpoint.read_text())


class FakeListener:
    """Hears what FakePiper said, garbled when the text asks to be mumbled; a recording it did not make is heard as
    `heard_as`."""

    def __init__(self, piper: FakePiper, heard_as: str = "") -> None:
        self.piper, self.heard_as, self.calls = piper, heard_as, 0

    def __call__(self, config: Any, device: str | None) -> "FakeListener":
        return self

    def transcribe(self, path: Path) -> str:
        self.calls += 1
        text = self.piper.spoken.get(str(path.resolve()), self.heard_as)
        return "something else entirely" if "mumble" in text else text

    def release(self) -> None:
        pass

    def __enter__(self) -> "FakeListener":
        return self

    def __exit__(self, *failure: object) -> None:
        self.release()


class FakeRater:
    """Scores what FakePiper said 4 on every quality metric and any other recording 3."""

    def __init__(self, piper: FakePiper) -> None:
        self.piper = piper

    def __call__(self, path: Path) -> dict[str, float]:
        from modeling.tts.quality import METRICS
        return dict.fromkeys(METRICS, 4.0 if str(path.resolve()) in self.piper.spoken else 3.0)


class FakeEmbedder:
    """openWakeWord's front end in shape only: per 80 ms, the log spectrum of the last 0.79 s in 96 bands, so a
    tone's pitch is what tells two recordings apart."""

    def __init__(self, directory: Path) -> None:
        pass

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        from modeling.wakeword import features
        count = max(0, (len(samples) - features.SPAN) // features.HOP + 1)
        rows = [np.abs(np.fft.rfft(samples[i * features.HOP: i * features.HOP + features.SPAN]))[:features.DIM * 40]
                for i in range(count)]
        return np.array([np.log1p(row.reshape(features.DIM, 40).mean(axis=1)) for row in rows], dtype=np.float32).reshape(-1, features.DIM)


def fake_feature_models(directory: Path) -> Path:
    from modeling.wakeword import features
    directory.mkdir(parents=True, exist_ok=True)
    for name in features.MODELS:
        (directory / name).write_text("a feature model")
    return directory
