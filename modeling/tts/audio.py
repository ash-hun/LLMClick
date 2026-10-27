"""WAV files: where a source's recordings are, their samples, resampling, and the measurements the speech checks use."""

import wave
import subprocess
from pathlib import Path
from typing import Any
from collections.abc import Iterator

import numpy as np

from core.utils.files import directory_signature
from modeling.tuning.sources import SOURCES
from modeling.config import Keyed

AUDIO_KEYS = ("path", "audio_dir")  # source keys that name local files: the rows, and where their audio starts
PCM16 = 2
FULL_SCALE = 32768.0
CLIPPED = 0.999        # |sample| at or above this share of full scale counts as clipped
SILENT = 0.01          # a frame whose RMS is below this is silence
FRAME_SECONDS = 0.02


def audio_root(source: Keyed) -> Path | None:
    """Where a source's relative audio paths start: `audio_dir`, else the directory of the source file."""
    params = source.params
    if params.get("audio_dir"):
        return Path(params["audio_dir"])
    return Path(params["path"]).parent if params.get("path") else None


def located(value: Any, root: Path | None) -> Path | None:
    """A row's audio file, absolute; an LJSpeech id without `.wav` finds its file too."""
    if not value:
        return None
    path = Path(str(value))
    path = path if path.is_absolute() or root is None else root / path
    for candidate in (path, path.with_name(path.name + ".wav")):
        if candidate.is_file():
            return candidate.resolve()
    return None


def recordings(source: Keyed) -> Iterator[tuple[int, dict[str, Any], Path | None]]:
    """(row number, row, its audio file or None) for every row of a source; the caller decides what is missing."""
    root = audio_root(source)
    for number, row in enumerate(SOURCES.get(source.name)(source.params)):
        yield number, row, located(row.get("audio"), root)


def signatures(sources: list[Keyed]) -> list[Any]:
    """Each source's audio directory as content, so re-recorded files are a new input to a stage."""
    return [directory_signature(root) if (root := audio_root(source)) else None for source in sources]


def seconds(path: Path) -> float:
    """Length from the header alone; any PCM width."""
    with wave.open(str(path)) as stream:
        return float(stream.getnframes() / stream.getframerate())


def read(path: Path) -> tuple[np.ndarray, int]:
    """16-bit PCM samples as floats in [-1, 1), channels averaged, and the sample rate."""
    with wave.open(str(path)) as stream:
        if stream.getsampwidth() != PCM16:
            raise ValueError(f"{path}: {8 * stream.getsampwidth()}-bit audio; convert it to 16-bit PCM WAV")
        raw = np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16)
        samples = raw.reshape(-1, stream.getnchannels()).mean(axis=1) / FULL_SCALE
        return samples.astype(np.float32), stream.getframerate()


def resample(samples: np.ndarray, rate: int, target: int) -> np.ndarray:
    """Band-limited resampling through the spectrum (what scipy.signal.resample does), without scipy."""
    if rate == target or not len(samples):
        return samples
    count = round(len(samples) * target / rate)
    spectrum = np.fft.rfft(samples)[: count // 2 + 1]
    return (np.fft.irfft(spectrum, count) * (count / len(samples))).astype(np.float32)


def filtered(source: Path, target: Path, chain: str) -> None:
    """`source` through an ffmpeg audio filter chain into `target`, a 16-bit PCM WAV at the source's rate."""
    try:
        done = subprocess.run(["ffmpeg", "-loglevel", "error", "-nostdin", "-y", "-i", str(source), "-af", chain,
                               "-c:a", "pcm_s16le", "-f", "wav", str(target)], capture_output=True, text=True, check=False)
    except FileNotFoundError as missing:
        raise FileNotFoundError("ffmpeg is not installed; an audio filter needs it on PATH") from missing
    if done.returncode:
        raise ValueError(f"ffmpeg filter {chain!r} on {source} failed: {done.stderr.strip()}")


def measure(samples: np.ndarray, rate: int) -> dict[str, float]:
    """Length, the share of clipped samples and the share of silent 20 ms frames."""
    frame = max(1, int(rate * FRAME_SECONDS))
    frames = samples[: len(samples) // frame * frame].reshape(-1, frame)
    rms = np.sqrt((frames ** 2).mean(axis=1)) if len(frames) else np.zeros(0)
    return {"seconds": len(samples) / rate,
            "clipped": float((np.abs(samples) >= CLIPPED).mean()) if len(samples) else 0.0,
            "silent": float((rms < SILENT).mean()) if len(rms) else 1.0}
