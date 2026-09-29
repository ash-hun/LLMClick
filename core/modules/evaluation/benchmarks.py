"""Frozen evaluation sets; each entry is `params -> Example JSONL` and never changes a file it already wrote."""

import subprocess
from pathlib import Path
from typing import Any

from core.modules.data.materialize import from_huggingface, from_local
from core.registry import BENCHMARKS
from core.utils.proc import run_module

JEVBENCH_REPOSITORY = "https://github.com/fstandhartinger/jevbench"


@BENCHMARKS.register("local_jsonl")
def local_jsonl(params: dict[str, Any], out: Path, seed: int) -> Path:
    return from_local(params, out / "rows.jsonl")


@BENCHMARKS.register("huggingface")
def huggingface(params: dict[str, Any], out: Path, seed: int) -> Path:
    return from_huggingface(params, out / "rows.jsonl")


@BENCHMARKS.register("jeff_probability")
def jeff_probability(params: dict[str, Any], out: Path, seed: int) -> Path:
    """Code-built probability questions as a benchmark (default seed differs from the builder so rows do not overlap)."""
    target = out / "train.jsonl"
    if not target.exists():
        run_module("jeff.probability", ["--out", str(out), "--count", str(params.get("count", 500)),
                                        "--seed", str(params.get("seed", seed + 1))])
    return target


@BENCHMARKS.register("jeff_panel")
def jeff_panel(params: dict[str, Any], out: Path, seed: int) -> Path:
    """BBH, Financial PhraseBank, JudgeBench, RAGTruth and WinoGrande at jeff's pinned revisions (4,599 rows)."""
    target = out / "panel.jsonl"
    if not target.exists():
        run_module("jeff.panel", ["--output", str(target)])
    return target


@BENCHMARKS.register("jeff_jevbench_hard")
def jeff_jevbench_hard(params: dict[str, Any], out: Path, seed: int) -> Path:
    target = out / "jevbench-hard.jsonl"
    if not target.exists():
        checkout = out / "repo"
        if not checkout.exists():
            subprocess.run(["git", "clone", "--quiet", JEVBENCH_REPOSITORY, str(checkout)], check=True)
        run_module("jeff.jevbench", ["--checkout", str(checkout), "--out", str(target)])
    return target
