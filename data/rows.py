"""Natural keys and atomic row files: what makes two runs of a data stage produce the same files."""

import json
import hashlib
from pathlib import Path
from typing import Any

Row = dict[str, Any]
KEY_LENGTH = 16


def key(*parts: Any) -> str:
    """A stable id from content: the same parts give the same key on every run and every machine."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:KEY_LENGTH]


def normalized(text: str) -> str:
    return " ".join(text.lower().split())


def read_rows(path: Path) -> list[Row]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def write_rows(path: Path, rows: list[Row]) -> None:
    """The whole file at once, through a temporary name: a reader never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    temporary.replace(path)


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
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{identifier}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(row, ensure_ascii=False, sort_keys=True))
    temporary.replace(path)
    return row
