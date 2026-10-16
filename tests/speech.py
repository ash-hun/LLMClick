"""Stand-ins for the Piper program and Whisper, so the speech recipes run in tests without either."""

import wave
from pathlib import Path
from typing import Any

import numpy as np

RATE = 22050


def tone(path: Path, seconds: float, amplitude: float = 0.3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = (amplitude * np.sin(np.arange(int(RATE * seconds)) * 440 * 2 * np.pi / RATE) * 32767).astype(np.int16)
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
        self.fits: list[dict[str, Any]] = []
        self.fail_after: int | None = None

    def speak(self, voice: Path, text: str, wav: Path, log: Path, scales: dict[str, float]) -> None:
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("piper went away")
        self.calls.append(text)
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
