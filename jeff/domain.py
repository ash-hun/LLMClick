"""Split a domain's training rows (for example an app's navigation decisions) into train, dev and calibration sets for
a domain fine-tune, keeping every family on one side so near-identical rows cannot sit on both sides of the split."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from jeff.data import validate, write_rows
from jeff.types import Example

SHARES = (("dev", 0.10), ("calibration", 0.05))  # the rest is training


def part(family: str, seed: int) -> str:
    value = int(hashlib.sha256(f"{seed}-{family}".encode()).hexdigest()[:12], 16) / 16 ** 12
    edge = 0.0
    for name, share in SHARES:
        edge += share
        if value < edge:
            return name
    return "train"


def split(rows: list[Example], seed: int) -> dict[str, list[Example]]:
    parts: dict[str, list[Example]] = {"train": [], "dev": [], "calibration": []}
    for row in rows:
        parts[part(row["family"], seed)].append(row)
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.rows.read_text().split("\n") if line]
    validate(rows)
    parts = split(rows, args.seed)
    report = {}
    for name, found in parts.items():
        if not found:
            raise ValueError(f"The {name} set is empty; the data has too few families")
        report[name] = {"rows": len(found), "families": len({r["family"] for r in found}),
                        "sha256": write_rows(args.out / f"{name}.jsonl", found)}
    report["labels_in_dev"] = dict(Counter(str(r["source"].get("kind")) for r in parts["dev"]))
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
