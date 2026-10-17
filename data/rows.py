"""Natural keys and atomic row files: what makes two runs of a data stage produce the same files."""

import json
import hashlib
from pathlib import Path
from typing import Any

from core.utils.files import atomic_path, read_jsonl, write_jsonl

Row = dict[str, Any]
KEY_LENGTH = 16


def key(*parts: Any) -> str:
    """A stable id from content: the same parts give the same key on every run and every machine."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:KEY_LENGTH]


def normalized(text: str) -> str:
    return " ".join(text.lower().split())


def read_rows(path: Path) -> list[Row]:
    """A stage's rows, or none when the file is not there yet."""
    return read_jsonl(path) if path.is_file() else []


def write_rows(path: Path, rows: list[Row]) -> None:
    write_jsonl(path, rows)


def trail(inputs: dict[str, Any], name: str, summary: Row) -> dict[str, Row]:
    """The run so far: the upstream stages' summaries plus this stage's, carried in the outputs to the manifest."""
    merged: dict[str, Row] = {}
    for outputs in inputs.values():
        merged.update(outputs.get("trail", {}))
    merged[name] = summary
    return merged


def cached(directory: Path, identifier: str) -> Row | None:
    """A per-item result kept under its natural key, so a rerun skips what was already done."""
    path = directory / f"{identifier}.json"
    if not path.is_file():
        return None
    row: Row = json.loads(path.read_text())
    return row


def keep(directory: Path, identifier: str, row: Row) -> Row:
    with atomic_path(directory / f"{identifier}.json") as temporary:
        temporary.write_text(json.dumps(row, ensure_ascii=False, sort_keys=True))
    return row
