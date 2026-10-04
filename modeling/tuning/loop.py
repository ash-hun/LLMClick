"""The one training loop: deterministic batches, accumulation, warmup then linear decay, resume snapshots, progress."""

import json
import math
import random
import logging
from pathlib import Path
from typing import Any

import torch

from modeling.tuning.method import Row, TrainingMethod
from modeling.tuning.config import TrainingConfig
from modeling.tuning.backbone import Backbone
from core.utils.files import write_json
from core.progress import Progress

logger = logging.getLogger(__name__)
LOG = "training.jsonl"
RESUME = "resume.pt"
SUMMARY = "summary.json"
CHECKPOINT = "checkpoint"


def schedule(count: int, training: TrainingConfig, seed: int) -> list[list[list[int]]]:
    """Row indices per optimizer step: reshuffled every epoch from the seed, so a resumed run sees the same batches."""
    batches: list[list[int]] = []
    for epoch in range(training.epochs):
        order = list(range(count))
        random.Random(seed + epoch).shuffle(order)
        batches += [order[start:start + training.batch_size] for start in range(0, count, training.batch_size)]
    steps = [batches[start:start + training.accumulation] for start in range(0, len(batches), training.accumulation)]
    return steps[: training.max_steps] if training.max_steps else steps


def learning_rate(step: int, total: int, training: TrainingConfig) -> float:
    warmup = math.ceil(total * training.warmup_ratio)
    if step < warmup:
        return training.lr * (step + 1) / warmup
    return training.lr * max(total - step, 1) / max(total - warmup, 1)


def history(run: Path) -> list[dict[str, Any]]:
    path = run / LOG
    return [json.loads(line) for line in path.read_text().split("\n") if line] if path.exists() else []


def fit(backbone: Backbone, method: TrainingMethod[Any], rows: list[Row], training: TrainingConfig, workdir: Path,
        seed: int, progress: Progress) -> dict[str, Any]:
    """Train `backbone` in place and write `checkpoint/` and `summary.json`; a rerun continues from `resume.pt`."""
    steps = schedule(len(rows), training, seed)
    optimizer = torch.optim.AdamW(backbone.trainable(), lr=training.lr, weight_decay=training.weight_decay)
    start, events = 0, []
    resume = workdir / RESUME
    if resume.exists():
        state = torch.load(resume, map_location=backbone.device, weights_only=True)
        backbone.model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start = int(state["step"])
        events = [event for event in history(workdir) if event["step"] <= start]
        logger.info("resuming from step %d of %d", start, len(steps))
    (workdir / LOG).write_text("".join(json.dumps(event) + "\n" for event in events))
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=backbone.device == "cuda")
    backbone.model.train()
    progress.update(start, len(steps))
    for index in range(start, len(steps)):
        torch.manual_seed(seed + index)  # methods that sample (GRPO) repeat exactly after a resume
        rate = learning_rate(index, len(steps), training)
        for group in optimizer.param_groups:
            group["lr"] = rate
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
        event = {"step": index + 1, "loss": total, "lr": rate, **method.metrics}
        with (workdir / LOG).open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        progress.update(index + 1, len(steps), ", ".join(f"{k} {v:.4f}" for k, v in event.items() if k not in {"step", "lr"}))
        if training.resume_every and (index + 1) % training.resume_every == 0 and index + 1 < len(steps):
            temporary = resume.with_suffix(".tmp")
            torch.save({"step": index + 1, "model": backbone.model.state_dict(), "optimizer": optimizer.state_dict()},
                       temporary)
            temporary.replace(resume)
    backbone.model.eval()
    backbone.save(workdir / CHECKPOINT)
    summary = {"steps": len(steps), "rows": len(rows), "loss": history(workdir)[-1]["loss"] if steps else None}
    write_json(workdir / SUMMARY, summary)
    resume.unlink(missing_ok=True)
    return summary
