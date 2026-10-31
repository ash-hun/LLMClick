"""`wakeword` modeling/wakeword/wakeword_detector.py: weighting negatives lowers their scores, and the cosine schedule trains."""

from typing import Any
from pathlib import Path

import numpy as np

from modeling.wakeword.wakeword_config import WakewordTraining
from modeling.wakeword import wakeword_detector, feature_embedder


def test_weighting_negatives_lowers_their_scores_and_the_cosine_schedule_trains(tmp_path: Path) -> None:
    import onnxruntime
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, feature_embedder.FRAMES, feature_embedder.DIM)).astype(np.float32)
    y = (rng.random(200) < 0.5).astype(np.float32)
    x[y == 1] += 0.1  # two classes that overlap, so the weight decides where the boundary falls

    def negative_scores(**training: Any) -> float:
        onnx = tmp_path / f"{sorted(training.items())}.onnx"
        wakeword_detector.fit(x, y, WakewordTraining(steps=60, batch_size=32, **training), 7, onnx, lambda step: None)
        scores = onnxruntime.InferenceSession(str(onnx)).run(None, {"input": x[y == 0]})[0]
        return float(scores.mean())

    assert negative_scores(negative_weight=30.0) < negative_scores(negative_weight=1.0)
    assert 0 < negative_scores(schedule="cosine") < 1
