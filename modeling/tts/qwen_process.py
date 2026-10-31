"""Qwen3-TTS behind a process boundary: environment/qwen/speak.py run with QWEN_PYTHON, one model load per batch."""

import json
import hashlib
import subprocess
from pathlib import Path

from core.file_store import sha256_file
from core.settings import get_settings

ENVIRONMENT = Path(__file__).resolve().parents[2] / "environment" / "qwen"
LOCK, SCRIPT = ENVIRONMENT / "uv.lock", ENVIRONMENT / "speak.py"
LOG_TAIL = 5
RATE = 24000  # what the 12 Hz speech tokenizer decodes to


class QwenFailed(RuntimeError):
    pass


def python() -> str:
    path = get_settings().QWEN_PYTHON
    if not Path(path).is_file():
        raise FileNotFoundError(f"QWEN_PYTHON {path!r} does not exist; build it with environment/qwen/setup.sh (Apple silicon)")
    return path


def version() -> str | None:
    """Which build speaks a stage's files: the environment's lock and the script, so changing either respeaks them."""
    return hashlib.sha256((sha256_file(LOCK) + sha256_file(SCRIPT)).encode()).hexdigest() if LOCK.is_file() else None


def seed(base: int, text: str, take: int) -> int:
    """The sampling seed of one take: the same text and take always draw the same speech."""
    return int.from_bytes(hashlib.sha256(f"{base}\0{text}\0{take}".encode()).digest()[:4], "big")


def speak(voice: dict[str, str], reference: Path, jobs: list[tuple[str, Path, int]], log: Path) -> None:
    """Each (text, wav, seed) read in the voice cloned from `reference`; `voice` holds model, revision and
    reference_text. Wavs that exist are kept, so a rerun after a crash speaks only the rest."""
    missing = [(text, wav, number) for text, wav, number in jobs if not wav.is_file()]
    if not missing:
        return
    request = {"model": voice["model"], "revision": voice["revision"], "reference": str(reference.resolve()),
               "reference_text": voice["reference_text"],
               "jobs": [{"text": text, "wav": str(wav.resolve()), "seed": number} for text, wav, number in missing]}
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        done = subprocess.run([python(), "-I", str(SCRIPT)], input=json.dumps(request), stdout=stream,
                              stderr=subprocess.STDOUT, text=True, check=False)
    if done.returncode:
        tail = " | ".join(log.read_text().splitlines()[-LOG_TAIL:])
        raise QwenFailed(f"speak.py exited with {done.returncode} (log: {log}): {tail}")
