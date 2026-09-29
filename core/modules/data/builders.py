"""Builders produce training rows (Example JSONL); each is `params -> path` and must be safe to run twice."""

from pathlib import Path
from typing import Any

from core.modules.data.materialize import from_huggingface, from_local
from core.registry import BUILDERS
from core.utils.proc import run_module


@BUILDERS.register("local_jsonl")
def local_jsonl(params: dict[str, Any], out: Path, seed: int) -> Path:
    return from_local(params, out / "train.jsonl")


@BUILDERS.register("huggingface")
def huggingface(params: dict[str, Any], out: Path, seed: int) -> Path:
    return from_huggingface(params, out / "train.jsonl")


@BUILDERS.register("jeff_extra")
def jeff_extra(params: dict[str, Any], out: Path, seed: int) -> Path:
    """The 27 public datasets jeff converts into decisions; `only` restricts to a subset of jeff.extra.BUILDERS."""
    target = out / "train.jsonl"
    if not target.exists():
        args = ["--out", str(out)]
        if params.get("only"):
            args += ["--only", *params["only"]]
        run_module("jeff.extra", args)
    return target


@BUILDERS.register("jeff_probability")
def jeff_probability(params: dict[str, Any], out: Path, seed: int) -> Path:
    """Code-built probability questions with exact targets; no downloads, no teacher."""
    target = out / "train.jsonl"
    if not target.exists():
        run_module("jeff.probability", ["--out", str(out), "--count", str(params.get("count", 3000)),
                                        "--seed", str(params.get("seed", seed))])
    return target
