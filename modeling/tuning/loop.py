"""The one training loop: deterministic batches, accumulation, warmup then linear decay, resume snapshots, progress."""

import json
import math
import random
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from modeling.tuning.method import Row, TrainingMethod
from modeling.tuning.config import TrainingConfig
from modeling.tuning.backbone import Backbone
from core.utils import device
from core.utils.files import write_json
from core.progress import Progress

logger = logging.getLogger(__name__)
LOG = "training.jsonl"
RESUME = "resume.pt"
SUMMARY = "summary.json"
CHECKPOINT = "checkpoint"
Evaluation = Callable[[], dict[str, float]]  # the method's metrics on held-out rows, with the weights as they are


def schedule(count: int, training: TrainingConfig, seed: int, repeats: int = 1) -> list[list[list[int]]]:
    """Row indices per optimizer step: reshuffled every epoch from the seed, so a resumed run sees the same batches.
    With `repeats`, each step's batches come up that many times in a row."""
    batches: list[list[int]] = []
    for epoch in range(training.epochs):
        order = list(range(count))
        random.Random(seed + epoch).shuffle(order)
        batches += [order[start:start + training.batch_size] for start in range(0, count, training.batch_size)]
    steps = [batches[start:start + training.accumulation] for start in range(0, len(batches), training.accumulation)]
    steps = [step for step in steps for _ in range(repeats)]
    return steps[: training.max_steps] if training.max_steps else steps


def learning_rate(step: int, total: int, training: TrainingConfig) -> float:
    warmup = math.ceil(total * training.warmup_ratio)
    if step < warmup:
        return training.lr * (step + 1) / warmup
    return training.lr * max(total - step, 1) / max(total - warmup, 1)


def evaluating(backbone: Backbone, evaluation: Evaluation) -> dict[str, float]:
    """Run an evaluation in the middle of training: inference mode for every module that trains (dropout off),
    training mode again afterwards. Only numbers are kept; a method may also report counts, which are numbers too."""
    modules = backbone.modules().values()
    for module in modules:
        module.eval()
    try:
        return {key: float(value) for key, value in evaluation().items()}
    finally:
        for module in modules:
            module.train()


def history(run: Path) -> list[dict[str, Any]]:
    path = run / LOG
    return [json.loads(line) for line in path.read_text().split("\n") if line] if path.exists() else []


def tracked(event: dict[str, Any]) -> dict[str, float]:
    """An event's numbers as the tracker sees them: training numbers under `train/`, evaluation ones keep `eval/`."""
    return {key if "/" in key else f"train/{key}": float(value) for key, value in event.items() if key != "step"}


def fit(backbone: Backbone, method: TrainingMethod[Any], rows: list[Row], training: TrainingConfig, workdir: Path,
        seed: int, progress: Progress, on_step: Callable[[int, dict[str, float]], None] | None = None,
        evaluation: Evaluation | None = None) -> dict[str, Any]:
    """Train `backbone` in place and write `checkpoint/` and `summary.json`; a rerun continues from `resume.pt`.
    Every `training.eval_every` steps `evaluation` (the method's metrics on held-out rows) runs with the weights as
    they are, and its numbers join that step's event as `eval/<metric>`: the curve a run is judged by, next to the
    loss, in `training.jsonl` and in the tracker.
    Resume snapshots are taken only between groups of repeated steps (`method.repeats`): a method that reuses what it
    sampled keeps those samples in memory, so a snapshot inside a group would resume with fresh samples and a
    different run. Between groups nothing is kept, and the resumed run repeats the uninterrupted one exactly."""
    steps = schedule(len(rows), training, seed, method.repeats)
    optimizer = torch.optim.AdamW(method.parameter_groups(backbone), lr=training.lr, weight_decay=training.weight_decay)
    peaks = [group["lr"] for group in optimizer.param_groups]
    start, events = 0, []
    resume = workdir / RESUME
    if resume.exists():
        state = torch.load(resume, map_location=backbone.device, weights_only=True)
        backbone.restore(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start = int(state["step"])
        events = [event for event in history(workdir) if event["step"] <= start]
        logger.info("resuming from step %d of %d", start, len(steps))
    (workdir / LOG).write_text("".join(json.dumps(event) + "\n" for event in events))
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=backbone.device.startswith("cuda"))
    backbone.model.train()
    every = training.resume_every or 0  # 0: no snapshots
    next_snapshot = start + every
    progress.update(start, len(steps))
    for index in range(start, len(steps)):
        device.seed(backbone.device, seed + index)  # methods that sample (GRPO) repeat exactly after a resume
        rate = learning_rate(index, len(steps), training)
        for group, peak in zip(optimizer.param_groups, peaks, strict=True):
            group["lr"] = peak * rate / training.lr
        total = 0.0
        for batch in steps[index]:
            with autocast:
                loss = method.loss(backbone, [rows[i] for i in batch]) / len(steps[index])
            loss.backward()  # type: ignore[no-untyped-call]
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(backbone.trainable(), training.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if not math.isfinite(total):
            raise RuntimeError(f"Loss became {total} at step {index + 1}; lower training.lr")
        done = index + 1
        event = {"step": done, "loss": total, "lr": rate, **method.metrics}
        if evaluation is not None and training.eval_every and done % training.eval_every == 0:
            event.update({f"eval/{key}": value for key, value in evaluating(backbone, evaluation).items()})
        with (workdir / LOG).open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        if on_step is not None:
            on_step(done, tracked(event))
        progress.update(done, len(steps), ", ".join(f"{k} {v:.4f}" for k, v in event.items() if k not in {"step", "lr"}))
        if every and done >= next_snapshot and done % method.repeats == 0 and done < len(steps):
            temporary = resume.with_suffix(".tmp")
            torch.save({"step": done, "model": backbone.snapshot(), "optimizer": optimizer.state_dict()}, temporary)
            temporary.replace(resume)
            next_snapshot = done + every
    backbone.model.eval()
    finished = method.finish(backbone, progress)
    progress.update(len(steps), len(steps), "saving checkpoint")
    backbone.save(workdir / CHECKPOINT)
    summary = {"steps": len(steps), "rows": len(rows), "loss": history(workdir)[-1]["loss"] if steps else None, **finished}
    write_json(workdir / SUMMARY, summary)
    resume.unlink(missing_ok=True)
    return summary
