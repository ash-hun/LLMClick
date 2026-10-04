"""Full-weight SFT through jeff.train in a child process; a rerun resumes from resume.pt or returns the finished checkpoint."""

import json
from pathlib import Path

from modeling.jev.model.backbones import train_args
from modeling.jev.config import ModelConfig, TrainingConfig
from core.utils.device import Device, resolve_device
from core.utils.proc import run_module
from core.progress import Progress

RUN_NAME = "train"


class TrainingEvents:
    """Turns the events jeff.train appends to its event file into progress updates; safe to poll at any time."""

    def __init__(self, path: Path, progress: Progress) -> None:
        self.path = path
        self.progress = progress
        self.offset = 0
        self.total: int | None = None

    def poll(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            chunk = stream.read()
        complete = chunk[: chunk.rfind(b"\n") + 1]  # a line still being written waits for the next poll
        self.offset += len(complete)
        for line in complete.splitlines():
            event = json.loads(line)
            if event["kind"] == "training_started":
                self.total = int(event["total_steps"])
                self.progress.update(int(event["step"]), self.total)
            elif event["kind"] == "training_step":
                self.progress.update(int(event["step"]), self.total, f"loss {float(event['loss']):.4f}")


def argv(training: TrainingConfig, model: ModelConfig, train: Path, dev: Path, temperature: Path, run: Path,
         output: Path, seed: int) -> list[str]:
    args = ["--train", str(train), "--development", str(dev), "--temperature", str(temperature),
            "--run", str(run), "--output", str(output), "--seed", str(seed), *train_args(model)]
    for name, value in training.model_dump().items():
        if value is not None:
            args += [f"--{name.replace('_', '-')}", str(value)]
    return args


def finished(run: Path) -> bool:
    summary = run / "summary.json"
    return summary.exists() and bool(json.loads(summary.read_text()).get("complete"))


def train(training: TrainingConfig, model: ModelConfig, train_file: Path, dev: Path, temperature: Path, runs: Path,
          checkpoints: Path, seed: int, device: Device | None, progress: Progress) -> Path:
    run = runs / RUN_NAME
    selected = checkpoints / "selected"
    if finished(run) and selected.exists():
        return selected
    args = argv(training, model, train_file, dev, temperature, run, checkpoints, seed)
    resume = checkpoints / "resume.pt"
    if resume.exists():
        args += ["--resume", str(resume)]
    events = run / "events.jsonl"
    env: dict[str, str] = {"JEFF_EVENTS": str(events), "JEFF_DEVICE": resolve_device(device)}
    run_module("jeff.train", args, env=env, poll=TrainingEvents(events, progress).poll, log=runs / "train.log")
    if not selected.exists():
        raise RuntimeError("Training finished without a selected checkpoint (no evaluation passed the calibration gate)")
    return selected
