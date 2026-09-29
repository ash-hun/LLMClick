"""Describe the shape of the evaluation panel next to the training data, from statistics only.

For each panel benchmark (and each sub-task of BBH and RAGTruth) and for each training source it reports: how many
rows, whether the state is plain text or named fields (and which), state length in characters, the question
instruction, the option keys and whether they carry descriptions, and the label balance. It never prints any state
text, so the report can guide data generation without exposing panel items."""

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from jeff.types import Example


def load(path: Path) -> list[Example]:
    return [json.loads(line) for line in path.read_text().split("\n") if line]


def panel_group(row: Example) -> str:
    """BBH rows by sub-task, RAGTruth rows by task type; other benchmarks as a whole."""
    if row["suite"] == "BBH":
        return "BBH / " + re.sub(r"-\d+$", "", row["family"]).removeprefix("panel-bbh-")
    if row["suite"] == "RAGTruth":
        return "RAGTruth / " + row["state"]["task"]
    return row["suite"]


def training_group(row: Example) -> str:
    """Synthetic rows by family, public rows by dataset."""
    if row["suite"] == "synthetic":
        return "synthetic / " + row["source"]["family"]
    return row["suite"]


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def profile(rows: list[Example]) -> dict[str, object]:
    lengths = [len(row["state"]) if isinstance(row["state"], str) else len(json.dumps(row["state"], ensure_ascii=False)) for row in rows]
    layouts = Counter("text" if isinstance(row["state"], str) else "fields: " + ", ".join(row["state"]) for row in rows)
    instructions = Counter(row["question"].get("instructions", "") for row in rows)
    option_sets = Counter(" | ".join(row["question"].get("criteria") or {}) for row in rows)
    described = sum(any(value is not None for value in (row["question"].get("criteria") or {}).values()) for row in rows)
    labels = Counter(str(row["label"]) for row in rows)
    return {"rows": len(rows), "layout": layouts.most_common(2), "chars_p10": percentile(lengths, .1),
            "chars_median": int(statistics.median(lengths)), "chars_p90": percentile(lengths, .9),
            "question_types": dict(Counter(row["question"]["type"] for row in rows)),
            "instructions": instructions.most_common(2), "distinct_instructions": len(instructions),
            "option_sets": option_sets.most_common(2), "options_described": described / len(rows),
            "labels": {label: round(count / len(rows), 3) for label, count in labels.most_common(8)}}


def groups(rows: list[Example], key) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[Example]] = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    return {name: profile(members) for name, members in sorted(grouped.items())}


def markdown(title: str, profiles: dict[str, dict[str, object]]) -> str:
    lines = [f"## {title}", "", "| Group | Rows | State | Chars p10 / median / p90 | Instruction | Options | Described | Labels |",
             "|---|---|---|---|---|---|---|---|"]
    for name, p in profiles.items():
        layout = "; ".join(f"{shape} ({count})" for shape, count in p["layout"])  # type: ignore[union-attr]
        instruction = p["instructions"][0][0][:80] + (f" (+{p['distinct_instructions'] - 1} more)" if p["distinct_instructions"] > 1 else "")  # type: ignore[index,operator]
        options = p["option_sets"][0][0][:60]  # type: ignore[index]
        labels = ", ".join(f"{label} {share:.0%}" for label, share in p["labels"].items())  # type: ignore[union-attr]
        lines.append(f"| {name} | {p['rows']} | {layout} | {p['chars_p10']} / {p['chars_median']} / {p['chars_p90']} | "
                     f"{instruction} | {options} | {p['options_described']:.0%} | {labels} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panel, train = load(args.panel), load(args.train)
    report = {"panel": groups(panel, panel_group), "train": groups(train, training_group)}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    args.output.write_text(f"# Panel shape next to the training data\n\nPanel: {args.panel}. Training data: {args.train}. "
                           "Statistics only; no state text is shown.\n\n"
                           + markdown("Panel", report["panel"]) + "\n" + markdown("Training data", report["train"]))


if __name__ == "__main__":
    main()
