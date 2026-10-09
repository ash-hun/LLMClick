"""Stage `select`: duplicates and leaks out, the rows assembled in the target recipe's format, the manifest written."""

from pathlib import Path
from typing import Any, ClassVar

from core.stage import Outputs, Stage
from core.utils.files import write_json
from data.config import SyntheticConfig
from data.rows import Row, key, normalized, read_rows, trail, write_rows
from data.teachers import teacher_from
from evaluation.contamination import shingles, texts_of
from modeling.tuning.sources import SOURCES

ROWS = "rows.jsonl"
MANIFEST = "manifest.json"


def text_of(row: Row) -> str:
    return f"{row.get('context', '')} {row['instruction']} {row['text']}"


class LeakIndex:
    """Evaluation items by their word n-grams; a row leaks an item when it holds at least `threshold` of the
    item's n-grams, whatever else the row contains."""

    def __init__(self, items: list[Any], ngram: int, threshold: float) -> None:
        self.ngram, self.threshold = ngram, threshold
        self.index: dict[int, set[int]] = {}
        self.sizes: list[int] = []
        for number, item in enumerate(items):
            grams = set(shingles(" ".join(texts_of(item)), ngram))
            self.sizes.append(len(grams))
            for gram in grams:
                self.index.setdefault(gram, set()).add(number)

    def leaks(self, text: str) -> bool:
        hits: dict[int, int] = {}
        for gram in set(shingles(text, self.ngram)):
            for number in self.index.get(gram, ()):
                hits[number] = hits.get(number, 0) + 1
        return any(self.sizes[number] and count / self.sizes[number] >= self.threshold for number, count in hits.items())


def best_first(group: list[Row]) -> list[Row]:
    """A prompt's verified answers, best first: by judge score, then reward, then the sample order."""
    return sorted(group, key=lambda row: (-float(row.get("score", 0)), -float(row.get("reward", 0)), int(row["sample"])))


def assemble(kind: str, group: list[Row], margin: float) -> Row | None:
    """Pure: one training row from a prompt's verified answers, or None when the kind cannot be made from them."""
    ranked = best_first(group)
    top = ranked[0]
    content = f"{top['context']}\n\n{top['instruction']}" if top.get("context") else top["instruction"]
    if kind == "sft":
        return {"messages": [{"role": "user", "content": content}, {"role": "assistant", "content": top["text"]}]}
    if kind == "grpo":
        return {"prompt": content, "answer": top["answer"]} if top.get("answer") is not None else None
    bottom = ranked[-1]
    if len(ranked) < 2 or float(top.get("score", 0)) - float(bottom.get("score", 0)) < margin:
        return None
    return {"prompt": content, "chosen": top["text"], "rejected": bottom["text"]}


class SelectStage(Stage[SyntheticConfig]):
    name: ClassVar[str] = "select"
    requires: ClassVar[tuple[str, ...]] = ("verify",)
    sections: ClassVar[tuple[str, ...]] = ("select",)

    def identity(self) -> Any:
        from modeling.config import keyed_identity
        leak = self.config.select.leak
        return [super().identity(), keyed_identity(leak.against) if leak else None]

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        config, settings = self.config, self.config.select
        verified = read_rows(Path(inputs["verify"]["rows"]))
        counts: dict[str, int] = {"verified": len(verified), "exact_duplicate": 0, "near_duplicate": 0, "leaked": 0, "unassembled": 0}
        # duplicates: exact by normalized text, near by shared n-grams with rows kept so far
        self.progress.update(0, None, "removing duplicates")
        seen: set[str] = set()
        kept_grams: list[set[int]] = []
        unique: list[Row] = []
        for row in sorted(verified, key=lambda row: (row["prompt_id"], row["sample"])):
            text = normalized(text_of(row))
            if text in seen:
                counts["exact_duplicate"] += 1
                continue
            seen.add(text)
            if settings.dedup.near is not None:
                grams = set(shingles(text, settings.dedup.ngram))
                # ponytail: every kept row is compared, O(n^2); MinHash with LSH when a run passes ~50k rows
                if grams and any(len(grams & other) / len(grams) >= settings.dedup.near for other in kept_grams):
                    counts["near_duplicate"] += 1
                    continue
                kept_grams.append(grams)
            unique.append(row)
        # leaks: rows that reproduce evaluation items
        if settings.leak is not None:
            self.progress.update(0, None, "removing leaks")
            items = [item for source in settings.leak.against for item in SOURCES.get(source.name)(source.params)]
            index = LeakIndex(items, settings.leak.ngram, settings.leak.threshold)
            clean = []
            for row in unique:
                if index.leaks(text_of(row)):
                    counts["leaked"] += 1
                else:
                    clean.append(row)
            unique = clean
        # assemble one training row per prompt
        self.progress.update(0, None, f"assembling {settings.assemble} rows")
        groups: dict[str, list[Row]] = {}
        for row in unique:
            groups.setdefault(row["prompt_id"], []).append(row)
        rows: list[Row] = []
        for prompt_id in sorted(groups):
            made = assemble(settings.assemble, groups[prompt_id], settings.margin)
            if made is None:
                counts["unassembled"] += 1
                continue
            rows.append({**made, "_prompt_id": prompt_id, "_id": key("row", settings.assemble, prompt_id)})
        if settings.max_rows is not None:
            rows = rows[: settings.max_rows]
        counts["rows"] = len(rows)
        write_rows(workdir / ROWS, [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows])
        stages = trail(inputs, self.name, {"counts": counts})
        manifest = {"recipe": config.recipe, "name": config.name, "assemble": settings.assemble, "counts": counts, "stages": stages,
                    "teacher": teacher_from(config.teacher).identity(),
                    "cost_usd": sum(float(summary.get("cost_usd", 0.0)) for summary in stages.values()),
                    "seeds": [seed.model_dump(mode="json") for seed in config.seeds],
                    "settings": {"prompts": config.prompts.model_dump(mode="json"), "evolve": config.evolve.model_dump(mode="json") if config.evolve else None,
                                 "respond": config.respond.model_dump(mode="json"), "verify": config.verify.model_dump(mode="json"),
                                 "select": settings.model_dump(mode="json")}}
        write_json(workdir / MANIFEST, manifest)
        self.progress.update(1, 1)
        return {"rows": str(workdir / ROWS), "manifest": str(workdir / MANIFEST), "count": len(rows), "counts": counts,
                "cost_usd": manifest["cost_usd"], "trail": stages}
