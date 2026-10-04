"""Carve dev, temperature and validation folds out of the builders' rows, whole families at a time (jeff.data.choose)."""

from jeff.data import choose
from jeff.types import Example

SEED_OFFSETS = {"dev": 2, "temperature": 3, "validation": 4}  # insertion order is carving order


def carve(rows: list[Example], sizes: dict[str, int], seed: int) -> dict[str, list[Example]]:
    """`sizes` maps dev, temperature and validation to row counts; everything left is train."""
    if len(rows) <= sum(sizes.values()):
        raise ValueError(f"{len(rows)} rows cannot yield folds {sizes} plus training rows")
    folds: dict[str, list[Example]] = {}
    taken: set[str] = set()
    for name, offset in SEED_OFFSETS.items():
        folds[name] = choose(rows, sizes[name], seed + offset, excluded=taken)
        taken |= {row["family"] for row in folds[name]}
    train_rows = sorted((row for row in rows if row["family"] not in taken), key=lambda row: row["id"])
    if not train_rows:
        raise ValueError("Every family went to the held-out folds; give more rows or smaller folds")
    return {"train": train_rows, **folds}
