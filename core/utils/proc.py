"""Run a jeff CLI module in a child process: GPU memory, JEFF_* environment and argparse stay isolated per stage."""

import logging
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)
POLL_SECONDS = 0.5
TAIL_LINES = 20


def run_module(module: str, args: list[str], env: dict[str, str] | None = None, cwd: Path | None = None,
               poll: Callable[[], None] | None = None, log: Path | None = None) -> None:
    """`poll` is called while the child runs and once after it exits, e.g. to turn its event file into progress.
    `log` receives the child's output instead of the terminal, which keeps progress bars readable; its tail is
    quoted when the child fails."""
    argv = [sys.executable, "-m", module, *args]
    logger.info("run: %s", " ".join(argv))
    merged = {**os.environ, **(env or {})}
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
    # without a log, child stdout goes to our stderr so the CLI's own stdout stays one JSON document
    sink = log.open("a") if log else None
    try:
        with subprocess.Popen(argv, env=merged, cwd=cwd, stdout=sink or sys.stderr,
                              stderr=subprocess.STDOUT if sink else None) as child:
            while True:
                try:
                    child.wait(timeout=POLL_SECONDS)
                    break
                except subprocess.TimeoutExpired:
                    if poll:
                        poll()
    finally:
        if sink:
            sink.close()
    if poll:
        poll()
    if child.returncode != 0:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-TAIL_LINES:]) if log else ""
        raise RuntimeError(f"{module} exited with {child.returncode}" + (f"; see {log}:\n{tail}" if log else ""))
