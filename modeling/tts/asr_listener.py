"""Whisper round trip: what a recording says, and its word error rate against what it should say."""

import re
from pathlib import Path
from typing import Any

from modeling.tts import audio_file
from modeling.tts.tts_config import AsrConfig
from core.device_selector import resolve_device
from core.settings import get_settings

WHISPER_RATE = 16000
WORD = re.compile(r"[\w']+")


def words(text: str) -> list[str]:
    # ponytail: no number or abbreviation normalisation, so "3" against "three" is an error; add a normaliser
    # (Whisper's EnglishTextNormalizer) when a threshold has to hold for text with many numbers
    return WORD.findall(text.lower())


def wer(reference: str, hypothesis: str) -> float:
    """Word edits (substitutions, insertions, deletions) per reference word."""
    expected, heard = words(reference), words(hypothesis)
    if not expected:
        return 0.0 if not heard else 1.0
    row = list(range(len(heard) + 1))
    for i, word in enumerate(expected, 1):
        previous, row[0] = row[0], i
        for j, other in enumerate(heard, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (word != other))
    return row[-1] / len(expected)


class Listener:
    """Whisper, loaded on first use and kept for the stage's run."""

    def __init__(self, config: AsrConfig, device: str | None) -> None:
        self.config, self.wanted = config, device
        self.model: Any = None
        self.processor: Any = None

    def load(self) -> None:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        token = get_settings().HF_TOKEN or None
        self.device = resolve_device(self.wanted)
        self.processor = WhisperProcessor.from_pretrained(self.config.name, revision=self.config.revision, token=token)
        model: Any = WhisperForConditionalGeneration.from_pretrained(self.config.name, revision=self.config.revision, token=token)
        self.model = model.to(self.device).eval()

    def transcribe(self, path: Path) -> str:
        # ponytail: Whisper hears the first 30 s only; Piper's training clips are far shorter, long-form needs chunking
        import torch
        if self.model is None:
            self.load()
        samples, rate = audio_file.read(path)
        features = self.processor(audio_file.resample(samples, rate, WHISPER_RATE), sampling_rate=WHISPER_RATE,
                                  return_tensors="pt").input_features.to(self.device)
        language = {"language": self.config.language} if self.config.language else {}
        with torch.no_grad():
            tokens = self.model.generate(features, task="transcribe", **language)
        return str(self.processor.batch_decode(tokens, skip_special_tokens=True)[0]).strip()

    def release(self) -> None:
        self.model = self.processor = None

    def __enter__(self) -> "Listener":
        return self

    def __exit__(self, *failure: object) -> None:
        self.release()
