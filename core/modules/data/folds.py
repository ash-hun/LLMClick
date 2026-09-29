"""Carve dev and temperature folds out of the builders' rows, whole families at a time (jeff.data.choose)."""

from jeff.data import choose
from jeff.types import Example

DEV_SEED_OFFSET = 2
TEMPERATURE_SEED_OFFSET = 3


def carve(rows: list[Example], dev: int, temperature: int, seed: int) -> dict[str, list[Example]]:
    if len(rows) <= dev + temperature:
        raise ValueError(f"{len(rows)} rows cannot yield dev={dev} and temperature={temperature} plus training rows")
    dev_rows = choose(rows, dev, seed + DEV_SEED_OFFSET)
    taken = {row["family"] for row in dev_rows}
    temperature_rows = choose(rows, temperature, seed + TEMPERATURE_SEED_OFFSET, excluded=taken)
    taken |= {row["family"] for row in temperature_rows}
    train_rows = sorted((row for row in rows if row["family"] not in taken), key=lambda row: row["id"])
    if not train_rows:
        raise ValueError("Every family went to dev/temperature; give more rows or smaller folds")
    return {"train": train_rows, "dev": dev_rows, "temperature": temperature_rows}
