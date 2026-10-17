"""The detector: openWakeWord's DNN head over FRAMES embeddings, trained here, run in ONNX as its runtime runs it."""

from typing import Any
from pathlib import Path
from collections.abc import Callable

import numpy as np

from modeling.wakeword.config import WakewordTraining
from core.utils.files import atomic_path
from modeling.wakeword import features

LOG_EVERY = 50
OPSET = 13


def network(training: WakewordTraining) -> Any:
    """openWakeWord's `dnn` model: flatten, a LayerNorm-ReLU layer, `blocks` more of them, one sigmoid score."""
    import torch.nn as nn
    layers: list[nn.Module] = [nn.Flatten(), nn.Linear(features.FRAMES * features.DIM, training.layer_dim),
                               nn.LayerNorm(training.layer_dim), nn.ReLU()]
    for _ in range(training.blocks):
        layers += [nn.Linear(training.layer_dim, training.layer_dim), nn.LayerNorm(training.layer_dim), nn.ReLU()]
    return nn.Sequential(*layers, nn.Linear(training.layer_dim, 1), nn.Sigmoid())


def fit(x: np.ndarray, y: np.ndarray, training: WakewordTraining, seed: int, onnx: Path,
        report: Callable[[int], None]) -> list[list[float]]:
    """Train on windows `x` (n, FRAMES, DIM) with labels `y`, every batch half positive, and export to `onnx`.
    Returns [step, loss] every LOG_EVERY steps."""
    import torch
    positives, negatives = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if not len(positives) or not len(negatives):
        raise ValueError(f"training needs both classes: {len(positives)} positive and {len(negatives)} negative windows")
    torch.manual_seed(seed)
    picker = np.random.default_rng(seed)
    model = network(training)
    optimizer = torch.optim.Adam(model.parameters(), lr=training.learning_rate)
    loss_of = torch.nn.BCELoss()
    inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
    half = training.batch_size // 2
    history: list[list[float]] = []
    model.train()
    for step in range(1, training.steps + 1):
        batch = np.concatenate([picker.choice(positives, half), picker.choice(negatives, training.batch_size - half)])
        loss = loss_of(model(inputs[batch]).squeeze(1), targets[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % LOG_EVERY == 0 or step == training.steps:
            history.append([step, loss.item()])
            report(step)
    model.eval()
    # ponytail: the TorchScript exporter, deprecated since torch 2.9 but enough for a few linear layers; move to
    # dynamo=True (and add onnxscript) when torch drops it
    with atomic_path(onnx) as temporary:
        torch.onnx.export(model, (torch.zeros(1, features.FRAMES, features.DIM),), str(temporary), dynamo=False,
                          opset_version=OPSET, input_names=["input"], output_names=["score"],
                          dynamic_axes={"input": {0: "batch"}, "score": {0: "batch"}})
    return history


class Scorer:
    """An exported detector with its feature models: audio in, one score per 80 ms window out."""

    def __init__(self, bundle: Path, model: str) -> None:
        import onnxruntime as ort
        self.embed = features.Embedder(bundle)
        self.session = ort.InferenceSession(str(bundle / model), providers=["CPUExecutionProvider"])

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        """Scores of every window of FRAMES embeddings in `samples` (16 kHz); window i ends at features.window_end(i)."""
        embedded = self.embed(samples)
        if len(embedded) < features.FRAMES:
            return np.zeros(0, dtype=np.float32)
        windows = np.lib.stride_tricks.sliding_window_view(embedded, features.FRAMES, axis=0).transpose(0, 2, 1)
        return np.asarray(self.session.run(None, {"input": np.ascontiguousarray(windows, dtype=np.float32)})[0]).reshape(-1)
