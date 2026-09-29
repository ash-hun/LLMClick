"""Measure how long a trained decision model takes to decide: one decision at a time (what an API caller waits for)
and in batches (throughput), on GPU or CPU, over a fixed sample of panel questions."""

import argparse
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from jeff.types import Example

if TYPE_CHECKING:
    from jeff.mlx_backend import MlxDecisionModel

WARMUP = 5
# Compiled kernels that transformers binds to Qwen3.5 at import time when installed; they run on CUDA only.
CUDA_ONLY_KERNELS = ("causal_conv1d", "fla")


def use_torch_kernels() -> None:
    """Make transformers bind its plain PyTorch implementations (the CPU path) by hiding the CUDA-only kernel packages.
    Must run before any Qwen3.5 modelling code is imported."""
    if "transformers.models.qwen3_5.modeling_qwen3_5" in sys.modules:
        raise RuntimeError("Qwen3.5 modelling code is already imported with its CUDA kernels bound")
    for name in CUDA_ONLY_KERNELS:
        sys.modules[name] = None  # type: ignore[assignment]  # an import of it now raises ImportError


def sample(rows: list[Example], count: int, seed: int) -> list[Example]:
    """A fixed sample spread evenly across the benchmarks (their text lengths differ a lot)."""
    suites = sorted({row["suite"] for row in rows})
    rng = random.Random(seed)
    per_suite = count // len(suites)
    picked: list[Example] = []
    for suite in suites:
        picked.extend(rng.sample([row for row in rows if row["suite"] == suite], per_suite))
    return picked


def summarise(milliseconds: list[float]) -> dict[str, float]:
    ordered = sorted(milliseconds)
    return {"median_ms": statistics.median(ordered), "p95_ms": ordered[max(0, round(0.95 * len(ordered)) - 1)],
            "mean_ms": statistics.fmean(ordered), "min_ms": ordered[0], "max_ms": ordered[-1]}


def synchronise(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def time_batches(model: torch.nn.Module, rows: list[Example], batch_size: int, device: str) -> tuple[list[float], list[int]]:
    """Wall-clock time per batch, including tokenisation, the forward pass and the probability readout."""
    timings, tokens = [], []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        synchronise(device)
        began = time.perf_counter()
        prepared = model.prepare(batch)
        (model(prepared) / model.temperature).softmax(-1).cpu()
        synchronise(device)
        timings.append((time.perf_counter() - began) * 1000)
        tokens.append(prepared.input_tokens)
    return timings, tokens


def time_mlx(model: "MlxDecisionModel", rows: list[Example]) -> tuple[list[float], list[int]]:
    """Wall-clock time per decision with the MLX backend (one at a time, as it serves), including tokenisation."""
    timings, tokens = [], []
    for row in rows:
        began = time.perf_counter()
        ((_, count),) = model.decide([row])
        timings.append((time.perf_counter() - began) * 1000)
        tokens.append(count)
    return timings, tokens


def processor_name() -> str:
    """The chip's marketing name on a Mac (e.g. "Apple M4 Max"); elsewhere what the platform reports."""
    if platform.system() == "Darwin":
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True).stdout.strip()
    return platform.processor() or platform.machine()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/panel.jsonl"))
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu", "mlx"), required=True,
                        help="mlx: Apple's MLX framework on a Mac GPU (Qwen3.5 checkpoints; one decision at a time)")
    parser.add_argument("--threads", type=int, default=16, help="CPU threads (CPU runs only)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    rows = [json.loads(line) for line in args.data.read_text().split("\n") if line]
    chosen = sample(rows, args.rows, seed=20260927)
    if args.device == "mlx":
        if args.batch_sizes != [1]:
            raise ValueError("The MLX backend decides one question at a time; use --batch-sizes 1")
        from jeff.mlx_backend import MlxDecisionModel

        model = MlxDecisionModel(args.checkpoint)
        dtype = "bfloat16"  # MlxDecisionModel keeps the checkpoint's bfloat16 weights
        measure = lambda rows, batch_size: time_mlx(model, rows)  # noqa: E731
    else:
        if args.device == "cpu":
            use_torch_kernels()
        from jeff.models import load_decision_model  # imported late so the CPU path above can take effect

        model = load_decision_model(checkpoint=args.checkpoint, device=args.device, cpu_threads=args.threads)
        dtype = str(next(model.parameters()).dtype)
        measure = lambda rows, batch_size: time_batches(model, rows, batch_size, args.device)  # noqa: E731
    measure(chosen[:WARMUP], 1)  # warm-up: kernels, caches, lazy initialisation
    result: dict[str, object] = {
        "checkpoint": str(args.checkpoint), "base_model": model.base_model, "device": args.device,
        "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else processor_name(),
        "cpu_threads": args.threads if args.device == "cpu" else None, "dtype": dtype,
        "rows": len(chosen), "batches": {}}
    for batch_size in args.batch_sizes:
        timings, tokens = measure(chosen, batch_size)
        decisions = len(chosen)
        result["batches"][str(batch_size)] = {  # type: ignore[index]
            **summarise(timings), "decisions_per_second": decisions / (sum(timings) / 1000),
            "median_input_tokens_per_batch": statistics.median(tokens)}
        if batch_size == 1:
            by_suite: dict[str, list[float]] = {}
            for row, ms in zip(chosen, timings, strict=True):
                by_suite.setdefault(row["suite"], []).append(ms)
            result["single_decision_median_ms_by_suite"] = {suite: statistics.median(v) for suite, v in sorted(by_suite.items())}
        print(json.dumps({"batch_size": batch_size, **result["batches"][str(batch_size)]}), flush=True)  # type: ignore[index]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
