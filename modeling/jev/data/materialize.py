"""Turn a Hugging Face dataset or a local file into a frozen Example JSONL through a converter."""

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from config import get_settings
from modeling.jev.registry import CONVERTERS
from jeff.data import validate, write_rows
from jeff.types import Example


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().split("\n") if line]


def convert_rows(rows: list[dict[str, Any]], converter: str, params: dict[str, Any]) -> list[Example]:
    convert = CONVERTERS.get(converter)
    converted = [row for index, raw in enumerate(rows) if (row := convert(raw, index, params)) is not None]
    validate(converted)
    return converted


def from_huggingface(params: dict[str, Any], out: Path) -> Path:
    """params: dataset, revision (commit), split, subset?, converter, limit?, plus converter params."""
    dataset = load_dataset(params["dataset"], name=params.get("subset"), split=params["split"], revision=params["revision"],
                           token=get_settings().HF_TOKEN or None)
    rows: list[dict[str, Any]] = [dict(row) for row in dataset]
    if params.get("limit"):
        rows = rows[: int(params["limit"])]
    converted = convert_rows(rows, params.get("converter", "example"), {"dataset": params["dataset"], **params})
    write_rows(out, converted)
    return out


def from_local(params: dict[str, Any], out: Path) -> Path:
    """params: path, converter? (default example)."""
    rows = read_jsonl(Path(params["path"]))
    converted = convert_rows(rows, params.get("converter", "example"), {"dataset": params["path"], **params})
    write_rows(out, converted)
    return out
