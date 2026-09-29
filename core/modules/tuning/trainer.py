"""Full-weight SFT through jeff.train in a child process; a rerun resumes from resume.pt or returns the finished checkpoint."""

import json
from pathlib import Path

from core.config.schema import ModelConfig, TrainingConfig
from core.modules.model.backbones import train_args
from core.utils.device import resolve_device
from core.utils.proc import run_module

RUN_NAME = "train"


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
          checkpoints: Path, seed: int, device: str | None) -> Path:
    run = runs / RUN_NAME
    selected = checkpoints / "selected"
    if finished(run) and selected.exists():
        return selected
    args = argv(training, model, train_file, dev, temperature, run, checkpoints, seed)
    resume = checkpoints / "resume.pt"
    if resume.exists():
        args += ["--resume", str(resume)]
    env = {"JEFF_EVENTS": str(run / "events.jsonl"), "JEFF_DEVICE": resolve_device(device)}
    run_module("jeff.train", args, env=env)
    if not selected.exists():
        raise RuntimeError("Training finished without a selected checkpoint (no evaluation passed the calibration gate)")
    return selected
