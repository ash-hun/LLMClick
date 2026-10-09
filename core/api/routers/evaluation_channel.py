"""Evaluation Channel: the reports the evaluation recipes wrote under an output directory, and one report in full."""

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query

from core.api.guard import confined
from core.api.schema import EvaluationSummary
from core.utils.files import read_json

evaluation_channel_router = APIRouter(prefix="/api/evaluations", tags=["Evaluation Channel"])
REPORT = "report"


def headline(recipe: str, report: dict[str, Any]) -> dict[str, Any]:
    """The numbers a list can show: the recipes' own scores, benchmarks' primary values, tasks' main scores."""
    if recipe == "evaluation_benchmark":
        return {task: entry.get("scores") for task, entry in report.get("benchmarks", {}).items()}
    if recipe == "evaluation_embedding":
        return {task: entry.get("main_score") for task, entry in report.get("tasks", {}).items()}
    if recipe == "evaluation_compare":
        return {"comparison": report.get("comparison"), "axes": report.get("axes")}
    scores: dict[str, Any] = report.get("scores", {})
    return scores


def reports(output_dir: Path) -> list[dict[str, Any]]:
    found = []
    for manifest_path in sorted(output_dir.glob("*/manifest.json")):
        manifest = read_json(manifest_path)
        recipe = str(manifest.get("recipe", ""))
        entry = manifest.get("stages", {}).get(REPORT)
        if not recipe.startswith("evaluation_") or not entry:
            continue
        path = Path(entry["outputs"]["report"])
        if not path.is_file():
            continue
        report = read_json(path)
        found.append({"experiment": manifest["key"], "recipe": recipe, "directory": str(manifest_path.parent), "report": str(path),
                      "model": report.get("model") or {"runs": report.get("runs"), "base": report.get("base")},
                      "scores": headline(recipe, report), "contamination": report.get("contamination")})
    return found


@evaluation_channel_router.get("", response_model=list[EvaluationSummary])
def index(output_dir: Annotated[str, Query(description="Where the experiments are")] = "./output",
          recipe: Annotated[str | None, Query(description="Only this evaluation recipe")] = None,
          experiment: Annotated[str | None, Query(description="Only evaluations of this modeling experiment (directory)")] = None) -> list[EvaluationSummary]:
    confined(output_dir, "output_dir")
    found = reports(Path(output_dir))
    if recipe is not None:
        found = [entry for entry in found if entry["recipe"] == recipe]
    if experiment is not None:
        wanted = str(Path(experiment))

        def of(entry: dict[str, Any]) -> bool:
            model = entry["model"]
            runs = model.get("runs") or ([model] if "experiment" in model else [])
            return any(str(Path(run["experiment"])) == wanted for run in runs)

        found = [entry for entry in found if of(entry)]
    return [EvaluationSummary(**entry) for entry in found]


@evaluation_channel_router.get("/{experiment}")
def show(experiment: str, output_dir: Annotated[str, Query(description="Where the experiments are")] = "./output") -> dict[str, Any]:
    confined(output_dir, "output_dir")
    for entry in reports(Path(output_dir)):
        if entry["experiment"] == experiment:
            report: dict[str, Any] = read_json(Path(entry["report"]))
            return {**entry, "report": report}
    raise HTTPException(status_code=404, detail=f"No evaluation report for experiment {experiment!r} under {output_dir}")
