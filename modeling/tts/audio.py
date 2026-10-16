"""WAV files as numbers: read, resample, and the measurements the speech checks use."""

import wave
from pathlib import Path

import numpy as np

PCM16 = 2
FULL_SCALE = 32768.0
CLIPPED = 0.999        # |sample| at or above this share of full scale counts as clipped
SILENT = 0.01          # a frame whose RMS is below this is silence
FRAME_SECONDS = 0.02


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


def measure(samples: np.ndarray, rate: int) -> dict[str, float]:
    """Length, the share of clipped samples and the share of silent 20 ms frames."""
    frame = max(1, int(rate * FRAME_SECONDS))
    frames = samples[: len(samples) // frame * frame].reshape(-1, frame)
    rms = np.sqrt((frames ** 2).mean(axis=1)) if len(frames) else np.zeros(0)
    return {"seconds": len(samples) / rate,
            "clipped": float((np.abs(samples) >= CLIPPED).mean()) if len(samples) else 0.0,
            "silent": float((rms < SILENT).mean()) if len(rms) else 1.0}
