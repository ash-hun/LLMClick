"""The Piper program behind a process boundary: speak, fit and export, each run with PIPER_PYTHON."""

import subprocess
from pathlib import Path
from typing import Any

from core.utils.files import sha256_file
from core.settings import get_settings

LOCK = Path(__file__).resolve().parents[2] / "environment" / "piper" / "uv.lock"
LOG_TAIL = 5


class PiperFailed(RuntimeError):
    pass


def python() -> str:
    path = get_settings().PIPER_PYTHON
    if not Path(path).is_file():
        raise FileNotFoundError(f"PIPER_PYTHON {path!r} does not exist; build it with environment/piper/setup.sh")
    return path


def version() -> str | None:
    """Which Piper build makes a stage's files: the lock of its environment, so an upgrade rebuilds those stages."""
    return sha256_file(LOCK) if LOCK.is_file() else None


def run(args: list[str], log: Path, stdin: str | None = None) -> None:
    """One Piper module run; its output is appended to `log`, and a failure quotes the log's last lines."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        done = subprocess.run([python(), "-m", *args], input=stdin, stdout=stream, stderr=subprocess.STDOUT, text=True,
                              check=False)
    if done.returncode:
        tail = " | ".join(log.read_text().splitlines()[-LOG_TAIL:])
        raise PiperFailed(f"{args[0]} exited with {done.returncode} (log: {log}): {tail}")


def options(values: dict[str, Any]) -> list[str]:
    """`{"trainer.max_epochs": 10}` as Lightning CLI arguments."""
    flags: list[str] = []
    for name, value in values.items():
        flags += [f"--{name}", str(value).lower() if isinstance(value, bool) else str(value)]
    return flags


def speak(voice: Path, text: str, wav: Path, log: Path, scales: dict[str, float], speaker: int | None = None) -> None:
    """`text` read by `voice` (an .onnx with its .onnx.json beside it) into `wav`, written whole or not at all.
    `scales`: length_scale, noise_scale, noise_w_scale; a missing one keeps the voice's own. `speaker`: the id of
    one speaker of a multi-speaker voice."""
    wav.parent.mkdir(parents=True, exist_ok=True)
    temporary = wav.with_suffix(".tmp")
    # idempotent: Piper draws its noise unseeded, so speaking the same text twice gives two different files; callers
    # keep the first under its natural key and never speak it again
    run(["piper", "-m", str(voice), "-f", str(temporary), *options(scales),
         *(["--speaker", str(speaker)] if speaker is not None else [])], log, stdin=text)
    temporary.replace(wav)


def fit(values: dict[str, Any], resume: Path | None, log: Path) -> None:
    """`piper.train fit` with these options, continuing from `resume` (a Lightning checkpoint) when given."""
    run(["piper.train", "fit", *options(values), *(["--ckpt_path", str(resume)] if resume else [])], log)


def export(checkpoint: Path, onnx: Path, log: Path) -> None:
    temporary = onnx.with_suffix(".tmp")
    run(["piper.train.export_onnx", "--checkpoint", str(checkpoint), "--output-file", str(temporary)], log)
    temporary.replace(onnx)
