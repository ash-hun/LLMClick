"""Score one checkpoint on every frozen benchmark with jeff.evaluate; results already on disk are reused."""

import json
from pathlib import Path
from typing import Any

from core.utils.device import resolve_device
from core.utils.files import write_json
from core.utils.proc import run_module


def evaluate(checkpoint: Path, benchmarks: dict[str, Path], out: Path, batch_size: int, device: str | None) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name, data in benchmarks.items():
        output = out / f"{name}.json"
        if not output.exists():
            partial = output.with_suffix(".predictions.jsonl")
            if partial.exists():  # a crash between writing predictions and the summary; jeff refuses to overwrite it
                partial.unlink()
            env = {"JEFF_DEVICE": resolve_device(device)}
            run_module("jeff.evaluate", ["--data", str(data), "--output", str(output), "--local",
                                         "--checkpoint", str(checkpoint), "--batch-size", str(batch_size)], env=env)
        results[name] = json.loads(output.read_text())["overall"]
    write_json(out / "results.json", {"checkpoint": str(checkpoint), "benchmarks": results})
    return results
