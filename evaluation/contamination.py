"""How much of an evaluation set the model has seen in training: exact matches and near-duplicates between the
experiment's training rows and the evaluation items, by word n-grams. A measurement, not a filter: the report shows
it next to the scores so an inflated number can be read as such."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from modeling.tuning.stages import read_rows

NGRAM = 8
NEAR = 0.5       # share of an item's n-grams found in the training rows that makes it a near-duplicate
FLAGGED = 50     # how many flagged item indices the report keeps


def texts_of(value: Any) -> list[str]:
    """Every string inside a row or a benchmark document, in order; keys that start with `_` are annotations."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for key, item in value.items() if not str(key).startswith("_") for text in texts_of(item)]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in texts_of(item)]
    return []


def normalized(text: str) -> str:
    return " ".join(text.lower().split())


def shingles(text: str, ngram: int = NGRAM) -> list[int]:
    """Hashes of the word n-grams of a text; a text shorter than n is one shingle."""
    words = normalized(text).split()
    if len(words) <= ngram:
        return [hash(" ".join(words))] if words else []
    return [hash(" ".join(words[start:start + ngram])) for start in range(len(words) - ngram + 1)]


class Overlap:
    """The training side, indexed once: full-text hashes for exact matches, n-gram hashes for near-duplicates."""

    def __init__(self, texts: Iterable[str], ngram: int = NGRAM) -> None:
        self.ngram = ngram
        exact, grams = set(), []
        for text in texts:
            exact.add(hash(normalized(text)))
            grams.extend(shingles(text, ngram))
        self.exact = exact
        self.grams = np.unique(np.fromiter(grams, dtype=np.int64, count=len(grams)))

    def share(self, text: str) -> float:
        """The share of the text's n-grams that occur in the training rows."""
        own = np.fromiter(shingles(text, self.ngram), dtype=np.int64)
        if own.size == 0 or self.grams.size == 0:
            return 0.0
        return float(np.isin(own, self.grams, assume_unique=False).mean())

    def is_exact(self, text: str) -> bool:
        return hash(normalized(text)) in self.exact


def measure(training_rows: Path | None, items: list[Any], ngram: int = NGRAM, near: float = NEAR) -> dict[str, Any] | None:
    """Overlap of `items` (rows or benchmark documents) with the training rows at `training_rows`; None when the
    experiment's training rows cannot be found (a base model, or a recipe without a data stage)."""
    if training_rows is None or not training_rows.is_file():
        return None
    rows = read_rows(training_rows)
    overlap = Overlap(" ".join(texts_of(row)) for row in rows)
    texts = [" ".join(texts_of(item)) for item in items]
    exact = [index for index, text in enumerate(texts) if overlap.is_exact(text)]
    shares = [overlap.share(text) for text in texts]
    flagged = sorted({*exact, *(index for index, share in enumerate(shares) if share >= near)})
    return {"training_rows": len(rows), "items": len(items), "exact": len(exact), "near": len(flagged),
            "share_mean": sum(shares) / len(shares) if shares else 0.0, "ngram": ngram, "threshold": near,
            "flagged": flagged[:FLAGGED]}


def training_rows_of(experiment: Path) -> Path | None:
    """Where a tuning experiment keeps the rows it trained on (the data stage's link)."""
    path = experiment / "data" / "train.jsonl"
    return path if path.is_file() else None
