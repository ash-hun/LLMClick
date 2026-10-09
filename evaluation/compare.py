"""`evaluation_compare`: several checkpoints, the base model among them, on the same benchmarks and rows; one
table, the difference to the base with its uncertainty, a polygon chart, a Markdown summary."""

import math
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field, model_validator

from core.config.schema import Section
from core.stage import Outputs, Stage
from core.utils.files import write_json
from evaluation.benchmark import HARNESS, Benchmarks, HarnessSettings, harness_version
from evaluation.config import EvaluationData, SourceModel

REPORT = "report.json"
CHART = "axes.svg"
SUMMARY = "summary.md"
Z = 1.96  # two-sided 95% interval
PRIMARY = ("exact_match", "acc_norm", "acc", "prompt_level_strict_acc", "f1", "mrr", "accuracy", "reward")


class Run(Section):
    label: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    experiment: str
    checkpoint: str = "validate"

    @property
    def model(self) -> SourceModel:
        return SourceModel(experiment=self.experiment, checkpoint=self.checkpoint)


class CompareConfig(HarnessSettings):
    runs: list[Run] = Field(min_length=2, description="The checkpoints to compare; one of them `checkpoint: base`")
    benchmarks: Benchmarks | None = Field(default=None, description="Public benchmarks every run is scored on")
    rows: EvaluationData | None = Field(default=None, description="Rows every run is scored on with its recipe's metrics")
    axes: dict[str, list[str]] = Field(default_factory=dict, description="Chart axis -> benchmark names and `rows:<metric>` entries, each 0..1, averaged")

    @model_validator(mode="after")
    def _comparable(self) -> "CompareConfig":
        labels = [run.label for run in self.runs]
        if len(set(labels)) != len(labels):
            raise ValueError(f"run labels must differ: {labels}")
        if not any(run.checkpoint == "base" for run in self.runs):
            raise ValueError("one run must be `checkpoint: base`: a comparison without the starting weights is not interpretable")
        if self.benchmarks is None and self.rows is None:
            raise ValueError("give benchmarks, rows, or both")
        names = {task.name for task in self.benchmarks.resolved()} if self.benchmarks else set()
        for axis, entries in self.axes.items():
            unknown = [e for e in entries if not (e in names or (e.startswith("rows:") and self.rows is not None))]
            if unknown:
                raise ValueError(f"axis {axis!r} names {unknown}, which no benchmark or rows:<metric> provides")
        return self

    def paths(self) -> list[str]:
        own = [path for run in self.runs for path in run.model.paths()]
        return [*super().paths(), *own, *(self.rows.paths() if self.rows else [])]

    @property
    def base(self) -> Run:
        return next(run for run in self.runs if run.checkpoint == "base")


def primary(scores: dict[str, float]) -> str | None:
    """The metric a benchmark is summarized by: the first well-known one it reports, else its first number."""
    candidates = [key for key in scores if not key.endswith("_stderr")]
    return next((key for key in PRIMARY if key in candidates), candidates[0] if candidates else None)


def verdict(value: float, base: float, stderr: float | None, base_stderr: float | None) -> dict[str, Any]:
    """The difference to the base run and whether it is outside the 95% interval of the two measurements. Without
    standard errors (the recipes' own metrics report none) the difference is given and `decided` is null."""
    delta = value - base
    if stderr is None or base_stderr is None:
        return {"delta": delta, "decided": None}
    margin = Z * math.sqrt(stderr ** 2 + base_stderr ** 2)
    return {"delta": delta, "margin": margin, "decided": abs(delta) > margin}


def polygon_chart(axes: dict[str, dict[str, float]], labels: list[str]) -> str:
    """One polygon per run over the axes (values 0..1), as a small self-contained SVG."""
    names = list(axes)
    size, center, radius = 360, 180.0, 120.0
    colors = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#4b5563"]

    def point(index: int, value: float) -> tuple[float, float]:
        angle = -math.pi / 2 + 2 * math.pi * index / max(len(names), 1)
        return center + radius * value * math.cos(angle), center + radius * value * math.sin(angle)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size + 160} {size}" font-family="sans-serif" font-size="11">']
    for ring in (0.25, 0.5, 0.75, 1.0):
        ring_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(i, ring) for i in range(len(names))))
        parts.append(f'<polygon points="{ring_points}" fill="none" stroke="#9ca3af" stroke-width="0.5"/>')
    for index, name in enumerate(names):
        x, y = point(index, 1.0)
        lx, ly = point(index, 1.12)
        parts.append(f'<line x1="{center}" y1="{center}" x2="{x:.1f}" y2="{y:.1f}" stroke="#9ca3af" stroke-width="0.5"/>')
        parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" fill="#374151">{name}</text>')
    for order, label in enumerate(labels):
        color = colors[order % len(colors)]
        run_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(i, axes[name].get(label, 0.0)) for i, name in enumerate(names)))
        parts.append(f'<polygon points="{run_points}" fill="{color}" fill-opacity="0.12" stroke="{color}" stroke-width="1.5"/>')
        parts.append(f'<rect x="{size + 10}" y="{20 + order * 18}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{size + 28}" y="{31 + order * 18}" fill="#374151">{label}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


class CompareReport(Stage[CompareConfig]):
    """Reads every run's benchmark and row scores and writes the comparison."""
    name: ClassVar[str] = "report"
    benchmark_stages: dict[str, dict[str, str]] = {}  # label -> task -> stage name
    row_stages: dict[str, str] = {}                   # label -> stage name

    def dependencies(self) -> tuple[str, ...]:
        return (*(name for tasks in self.benchmark_stages.values() for name in tasks.values()), *self.row_stages.values())

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        config = self.config
        labels, base = [run.label for run in config.runs], config.base.label
        benchmarks: dict[str, dict[str, Any]] = {}
        for label, tasks in self.benchmark_stages.items():
            for task, stage in tasks.items():
                scores = inputs[stage]["scores"]
                metric = primary(scores)
                benchmarks.setdefault(task, {})[label] = {
                    "metric": metric, "value": scores.get(metric) if metric else None,
                    "stderr": scores.get(f"{metric}_stderr") if metric else None,
                    "n": inputs[stage]["n"].get(task), "num_fewshot": inputs[stage]["num_fewshot"], "limit": inputs[stage]["limit"],
                    "scores": scores}
        rows: dict[str, dict[str, float]] = {}
        for label, stage in self.row_stages.items():
            for metric, value in inputs[stage]["metrics"].items():
                rows.setdefault(metric, {})[label] = float(value)
        comparison: dict[str, dict[str, Any]] = {}
        for task, by_label in benchmarks.items():
            reference = by_label[base]
            if reference["value"] is not None:
                comparison[task] = {label: verdict(entry["value"], reference["value"], entry["stderr"], reference["stderr"])
                                    for label, entry in by_label.items() if label != base and entry["value"] is not None}
        for metric, by_label in rows.items():
            comparison[f"rows:{metric}"] = {label: verdict(value, by_label[base], None, None)
                                            for label, value in by_label.items() if label != base}
        axes: dict[str, dict[str, float]] = {}
        for axis, entries in config.axes.items():
            axes[axis] = {}
            for label in labels:
                values = [benchmarks[e][label]["value"] if not e.startswith("rows:") else rows[e[5:]].get(label)
                          for e in entries if (e in benchmarks and label in benchmarks[e]) or (e.startswith("rows:") and e[5:] in rows)]
                numbers = [min(max(float(v), 0.0), 1.0) for v in values if v is not None]
                axes[axis][label] = sum(numbers) / len(numbers) if numbers else 0.0
        report = {"runs": [{"label": run.label, "experiment": run.experiment, "checkpoint": run.checkpoint} for run in config.runs],
                  "base": base,
                  "settings": {"batch_size": config.evaluation.batch_size, "decoding": config.decoding.model_dump(mode="json"),
                               "chat_template": config.chat_template, "harness": {"name": HARNESS, "version": harness_version()}},
                  "benchmarks": benchmarks, "rows": rows, "comparison": comparison, "axes": axes,
                  "chart": str(workdir / CHART) if axes else None, "summary": str(workdir / SUMMARY), "contamination": None}
        if axes:
            (workdir / CHART).write_text(polygon_chart(axes, labels))
        (workdir / SUMMARY).write_text(self.summary(report))
        write_json(workdir / REPORT, report)
        return {"report": str(workdir / REPORT), "summary": str(workdir / SUMMARY), "chart": report["chart"],
                "comparison": comparison}

    def summary(self, report: dict[str, Any]) -> str:
        """A Markdown table per kind of score: runs as columns, the base first; a difference inside the interval of
        the two measurements is marked as not decided."""
        labels = [report["base"], *[run["label"] for run in report["runs"] if run["label"] != report["base"]]]
        lines = [f"# {self.config.name}", "", f"Base run: `{report['base']}`. Values are the primary metric, with the standard error "
                 "the harness reports; delta is against the base, marked `~` when inside the 95% interval of the two "
                 "measurements and `?` when no interval is available.", ""]

        def cell(label: str, entry: dict[str, Any] | None, task: str) -> str:
            if entry is None or entry.get("value") is None:
                return "-"
            text = f"{entry['value']:.4f}" + (f" ± {entry['stderr']:.4f}" if entry.get("stderr") is not None else "")
            if label == report["base"]:
                return text
            outcome = report["comparison"].get(task, {}).get(label)
            if outcome is None:
                return text
            mark = "" if outcome["decided"] else ("~" if outcome["decided"] is False else "?")
            return f"{text} ({outcome['delta']:+.4f}{mark})"

        if report["benchmarks"]:
            lines += ["## Benchmarks", "", "| benchmark | metric | " + " | ".join(labels) + " |", "|---|---|" + "---|" * len(labels)]
            for task, by_label in report["benchmarks"].items():
                metric = next((e["metric"] for e in by_label.values() if e["metric"]), "-")
                lines.append(f"| {task} | {metric} | " + " | ".join(cell(label, by_label.get(label), task) for label in labels) + " |")
            lines.append("")
        if report["rows"]:
            lines += ["## Rows", "", "| metric | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
            for metric, by_label in report["rows"].items():
                entries = {label: {"value": value} for label, value in by_label.items()}
                lines.append(f"| {metric} | " + " | ".join(cell(label, entries.get(label), f"rows:{metric}") for label in labels) + " |")
            lines.append("")
        if report["axes"]:
            lines += ["## Axes", "", f"![axes]({CHART})", "", "| axis | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
            for axis, by_label in report["axes"].items():
                lines.append(f"| {axis} | " + " | ".join(f"{by_label.get(label, 0.0):.3f}" for label in labels) + " |")
            lines.append("")
        return "\n".join(lines)
