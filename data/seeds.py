"""Stage `seeds`: every seed as one row with a stable id, whatever it came from."""

from pathlib import Path
from typing import Any, ClassVar

from core.stage import Outputs, Stage
from data.config import SyntheticConfig
from data.rows import Row, key, trail, write_rows
from modeling.config import keyed_identity
from modeling.tuning.sources import SOURCES

SEEDS = "seeds.jsonl"


def seed_rows(config: SyntheticConfig) -> list[Row]:
    """Pure: the config's seeds as rows. Inline values and rows of a source alike become {id, kind, text, row}."""
    rows: list[Row] = []
    for spec in config.seeds:
        if spec.values is not None:
            texts: list[tuple[str, Row]] = [(text, {}) for text in spec.values]
        else:
            assert spec.source is not None and spec.field is not None
            loaded = SOURCES.get(spec.source.name)(spec.source.params)
            texts = [(str(row[spec.field]), row) for row in loaded]
        for text, row in texts[: spec.limit]:
            rows.append({"id": key("seed", spec.kind, text), "kind": spec.kind, "text": text, "row": row})
    unique: dict[str, Row] = {}
    for row in rows:  # the same text twice in the seeds is one seed
        unique.setdefault(row["id"], row)
    return sorted(unique.values(), key=lambda row: row["id"])


class SeedsStage(Stage[SyntheticConfig]):
    name: ClassVar[str] = "seeds"
    sections: ClassVar[tuple[str, ...]] = ("seeds",)

    def identity(self) -> Any:
        sources = [spec.source for spec in self.config.seeds if spec.source is not None]
        return [super().identity(), keyed_identity(sources)]

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        self.progress.update(0, None, "reading seeds")
        rows = seed_rows(self.config)
        if not rows:
            raise ValueError("no seeds")
        write_rows(workdir / SEEDS, rows)
        kinds: dict[str, int] = {}
        for row in rows:
            kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
        summary = {"count": len(rows), "kinds": kinds}
        return {"rows": str(workdir / SEEDS), **summary, "trail": trail(inputs, self.name, summary)}
