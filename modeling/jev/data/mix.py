"""Leak filter, panel layouts, escape options and hijack attempts, then the training arms: jeff.mix in-process."""

import json
from pathlib import Path
from typing import Any

from modeling.jev.config import MixConfig
from jeff import adversarial, escape, layout
from jeff.data import write_rows
from jeff.mix import build, build_public, read
from jeff.types import Example


def dress(rows: list[Example], config: MixConfig, seed: int) -> tuple[list[Example], dict[str, Any]]:
    report: dict[str, Any] = {}
    if config.panel_layout:
        rows, report["layouts"] = layout.rearrange(rows, seed)
    if config.escape:
        rows, report["escape_rows"] = escape.add(rows, seed)
    if config.adversarial:
        rows, report["adversarial_rows"] = adversarial.add(rows, seed)
    return rows, report


def run(public: Path, dev: Path, calibration: Path, protected: list[Path], synthetic: Path | None, config: MixConfig,
        seed: int, out: Path) -> dict[str, Any]:
    report_path = out / "report.json"
    if report_path.exists():
        cached: dict[str, Any] = json.loads(report_path.read_text())
        return cached
    panel = [row for path in protected for row in read(path)]
    public_rows, public_report = dress(read(public), config, seed)
    if synthetic is None:
        sets, report = build_public(public_rows, read(dev), read(calibration), panel, config.size, config.sweep_size, seed)
        report["dressing"] = {"public": public_report}
    else:
        synthetic_rows, synthetic_report = dress(read(synthetic), config, seed)
        sets, report = build(public_rows, synthetic_rows, read(dev), read(calibration), panel, config.size, config.sweep_size, seed)
        report["dressing"] = {"public": public_report, "synthetic": synthetic_report}
    report["sha256"] = {name: write_rows(out / f"{name}.jsonl", rows) for name, rows in sets.items()}
    report["train_file"] = str(out / ("public.jsonl" if synthetic is None else "combined.jsonl"))
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
