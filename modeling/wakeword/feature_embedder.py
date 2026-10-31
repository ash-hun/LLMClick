"""openWakeWord's frozen front end: 16 kHz audio to a 96-number speech embedding every 80 ms, as its runtime computes it."""

import urllib.request
from pathlib import Path

import numpy as np

from core.file_store import atomic_path, sha256_file
from modeling.tts import audio_file

RATE = 16000
HOP = 1280                     # 80 ms of audio between two embeddings
SPAN = 12640                   # audio one embedding reads: 76 melspectrogram frames of 10 ms, plus the frame edges
FRAMES = 16                    # embeddings the detector reads at once
WINDOW = SPAN + (FRAMES - 1) * HOP + 160  # 2 s: the audio that gives exactly FRAMES embeddings
DIM = 96
PCM = 32767                    # the melspectrogram model takes 16-bit sample values
MEL_WINDOW, MEL_STEP = 76, 8
RELEASE = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/"
MODELS = {  # file -> sha256; the openWakeWord runtime ships the same two files
    "melspectrogram.onnx": "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
    "embedding_model.onnx": "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
}


def ensure(directory: Path) -> Path:
    """`directory` holding both feature models, each downloaded once and checked against its pinned hash."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, digest in MODELS.items():
        path = directory / name
        if path.is_file() and sha256_file(path) == digest:
            continue
        with atomic_path(path) as temporary:
            urllib.request.urlretrieve(RELEASE + name, temporary)
            if sha256_file(temporary) != digest:
                raise ValueError(f"{RELEASE + name} does not match its pinned sha256 {digest}")
    return directory


def clip(path: Path) -> np.ndarray:
    """A recording as 16 kHz floats in [-1, 1]."""
    samples, rate = audio_file.read(path)
    return audio_file.resample(samples, rate, RATE)


def window_end(index: int) -> int:
    """The sample at which the detector's `index`-th window of FRAMES embeddings ends: when it can fire."""
    return SPAN + (index + FRAMES - 1) * HOP


class Embedder:
    """The two feature models on onnxruntime, loaded once for a stage's run."""

    def __init__(self, directory: Path) -> None:
        import onnxruntime as ort
        options = {"providers": ["CPUExecutionProvider"]}
        self.melspectrogram = ort.InferenceSession(str(directory / "melspectrogram.onnx"), **options)
        self.embedding = ort.InferenceSession(str(directory / "embedding_model.onnx"), **options)

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        """(n, 96) embeddings of `samples`, one per HOP once SPAN samples are in; openWakeWord's transform included."""
        pcm = np.round(np.clip(samples, -1, 1) * PCM).astype(np.float32)[None]
        spec = np.squeeze(self.melspectrogram.run(None, {"input": pcm})[0]) / 10 + 2
        starts = range(0, len(spec) - MEL_WINDOW + 1, MEL_STEP)
        if not len(starts):
            return np.zeros((0, DIM), dtype=np.float32)
        windows = np.stack([spec[start:start + MEL_WINDOW] for start in starts])[..., None].astype(np.float32)
        return np.asarray(self.embedding.run(None, {self.embedding.get_inputs()[0].name: windows})[0]).reshape(-1, DIM)
