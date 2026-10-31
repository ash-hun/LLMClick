"""`evaluation_wakeword` evaluation/wakeword_evaluation.py: Wilson and Poisson intervals, when a detector hits, and overlapping ambient chunks."""

from pathlib import Path

import pytest
import numpy as np

from evaluation.wakeword_evaluation import CHUNK, concatenated, hit, poisson, wilson
from modeling.wakeword import feature_embedder
from tests.speech_fakes import melody


def test_intervals_hits_and_ambient_chunks() -> None:
    assert wilson(0, 10)[0] == 0 and wilson(10, 10)[1] == 1 and wilson(5, 10)[0] < 0.5 < wilson(5, 10)[1]
    assert poisson(0, 10)[0] == 0 and poisson(0, 10)[1] == pytest.approx(0.369, abs=0.01)  # exact: 3.689 / 10 h
    assert poisson(10, 1)[0] == pytest.approx(4.80, abs=0.05) and poisson(10, 1)[1] == pytest.approx(18.39, abs=0.1)
    scores = np.zeros(80, dtype=np.float32)
    first = next(i for i in range(80) if feature_embedder.window_end(i) / feature_embedder.RATE >= 2.0 + 0.5)  # 0.5 s phrase starting at 2 s
    scores[first] = 0.9
    assert hit(scores, 0.5, 0.5, 1.0) == pytest.approx(feature_embedder.window_end(first) / feature_embedder.RATE - 2.5)
    assert hit(scores, 0.5, 0.95, 1.0) is None


def test_ambient_chunks_overlap_by_a_window(tmp_path: Path) -> None:
    melody(tmp_path / "a.wav", [(440, 50.0)])
    melody(tmp_path / "b.wav", [(880, 30.0)])
    whole = np.concatenate([feature_embedder.clip(tmp_path / "a.wav"), feature_embedder.clip(tmp_path / "b.wav")])
    chunks = list(concatenated([str(tmp_path / "a.wav"), str(tmp_path / "b.wav")]))
    assert [len(chunk) for chunk in chunks[:-1]] == [CHUNK + feature_embedder.WINDOW - feature_embedder.HOP]
    for index, chunk in enumerate(chunks):
        assert np.array_equal(chunk, whole[index * CHUNK: index * CHUNK + len(chunk)])
    assert len(chunks) * CHUNK >= len(whole) - feature_embedder.WINDOW
