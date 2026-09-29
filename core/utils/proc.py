"""Run a jeff CLI module in a child process: GPU memory, JEFF_* environment and argparse stay isolated per stage."""

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def run_module(module: str, args: list[str], env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    argv = [sys.executable, "-m", module, *args]
    logger.info("run: %s", " ".join(argv))
    merged = {**os.environ, **(env or {})}
    # child stdout goes to our stderr so the CLI's own stdout stays one JSON document
    completed = subprocess.run(argv, env=merged, cwd=cwd, stdout=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"{module} exited with {completed.returncode}")
