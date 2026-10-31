"""Row sources: a `name:` key in `data.sources` resolves here and yields plain dict rows."""

import csv
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from core.recipe_registry import Registry
from core.settings import get_settings

SOURCES = Registry("source")
Row = dict[str, Any]
DELIMITED_COLUMNS = ["audio", "text"]  # an LJSpeech or Piper metadata.csv


def renamed(rows: list[Row], fields: dict[str, str] | None) -> list[Row]:
    """`fields` maps the key a method expects to the column the source calls it; other columns are kept."""
    if not fields:
        return rows
    return [{**row, **{target: row[column] for target, column in fields.items()}} for row in rows]


@SOURCES.register("local_jsonl")
def local_jsonl(params: dict[str, Any]) -> list[Row]:
    """params: path, limit?, fields?"""
    lines = [line for line in Path(params["path"]).read_text().split("\n") if line.strip()]
    return renamed([json.loads(line) for line in lines[: params.get("limit")]], params.get("fields"))


@SOURCES.register("huggingface")
def huggingface(params: dict[str, Any]) -> list[Row]:
    """params: dataset, revision (commit), split, subset?, limit?, fields?"""
    dataset = load_dataset(params["dataset"], name=params.get("subset"), split=params["split"],
                           revision=params["revision"], token=get_settings().HF_TOKEN or None)
    if params.get("limit"):
        dataset = dataset.select(range(min(int(params["limit"]), len(dataset))))
    return renamed([dict(row) for row in dataset], params.get("fields"))


@SOURCES.register("delimited")
def delimited(params: dict[str, Any]) -> list[Row]:
    """params: path, delimiter? ("|"), columns? (["audio", "text"]), limit?: a headerless file of delimited lines."""
    columns = params.get("columns") or DELIMITED_COLUMNS
    with Path(params["path"]).open(newline="", encoding="utf-8") as stream:
        lines = [line for line in csv.reader(stream, delimiter=params.get("delimiter", "|"), quoting=csv.QUOTE_NONE) if line]
    return [dict(zip(columns, line, strict=False)) for line in lines[: params.get("limit")]]
