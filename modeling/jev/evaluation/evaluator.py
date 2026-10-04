"""Score one checkpoint on Example JSONL sets with jeff.evaluate; results already on disk are reused."""

import json
from pathlib import Path
from typing import Any

from core.utils.device import Device, resolve_device
from core.utils.files import write_json
from core.utils.proc import run_module
from core.progress import Progress


def evaluate(checkpoint: Path, benchmarks: dict[str, Path], out: Path, batch_size: int, device: Device | None,
             progress: Progress) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for index, (name, data) in enumerate(benchmarks.items()):
        progress.update(index, len(benchmarks), name)
        output = out / f"{name}.json"
        if not output.exists():
            partial = output.with_suffix(".predictions.jsonl")
            if partial.exists():  # a crash between writing predictions and the summary; jeff refuses to overwrite it
                partial.unlink()
            env: dict[str, str] = {"JEFF_DEVICE": resolve_device(device)}
            run_module("jeff.evaluate", ["--data", str(data), "--output", str(output), "--local",
                                         "--checkpoint", str(checkpoint), "--batch-size", str(batch_size)], env=env,
                       log=out / "evaluate.log")
        results[name] = json.loads(output.read_text())["overall"]
    progress.update(len(benchmarks), len(benchmarks))
    write_json(out / "results.json", {"checkpoint": str(checkpoint), "benchmarks": results})
    return results
