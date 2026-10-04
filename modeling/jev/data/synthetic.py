"""Synthetic questions from a local teacher: jeff-materials -> jeff-grounded -> jeff-generate run/finalize."""

from pathlib import Path
from typing import Any

from config import get_settings
from modeling.jev.registry import TEACHERS
from core.utils.proc import run_module
from jeff.families import BY_NAME, FOCUS, FOCUS_V2, FOCUS_V3, FOCUS_V4, FOCUS_V5, BBH_CODE
from jeff.generate import GROUNDED

LOG = "synthetic.log"  # child output goes here, not onto the progress bar
FOCUS_WEIGHTS = {"v1": FOCUS, "v2": FOCUS_V2, "v3": FOCUS_V3, "v4": FOCUS_V4, "v5": FOCUS_V5, "bbh-code": BBH_CODE,
                 "fallacies": {"formal_fallacies": 1}}


@TEACHERS.register("openai_compatible")
def openai_compatible(params: dict[str, Any]) -> dict[str, str]:
    """Environment for jeff's teacher client: `url` in the config or TEACHER_URL in environment/.env."""
    url = params.get("url") or get_settings().TEACHER_URL
    if not url:
        raise ValueError("data.synthetic.teacher.url or TEACHER_URL is required when synthetic data is enabled")
    return {"JEFF_TEACHER_URL": url}


def needs_grounded(focus: str | None) -> bool:
    weights = FOCUS_WEIGHTS[focus] if focus else {name: 1 for name in BY_NAME}
    return any(name in GROUNDED for name, weight in weights.items() if weight > 0)


def generate(out: Path, public_train: Path, slots: int, seed: int, focus: str | None, concurrency: int,
             env: dict[str, str]) -> Path:
    """Each jeff step keeps its own cache/outcomes, so a rerun resumes where the teacher stopped."""
    synthetic = out / "gen" / "synthetic.jsonl"
    if synthetic.exists():
        return synthetic
    materials = out / "materials.json"
    if not materials.exists():
        run_module("jeff.materials", ["--out", str(materials)], env=env, log=out / LOG)
    args = ["run", "--slots", str(slots), "--seed", str(seed), "--out", str(out / "gen"), "--materials", str(materials),
            "--concurrency", str(concurrency)]
    if focus:
        args += ["--focus", focus]
    if needs_grounded(focus):
        grounded = out / "grounded.jsonl"
        if not grounded.exists():
            run_module("jeff.grounded", ["--extra", str(public_train), "--out", str(grounded)], env=env, log=out / LOG)
        args += ["--grounded", str(grounded)]
    run_module("jeff.generate", args, env=env, log=out / LOG)
    run_module("jeff.generate", ["finalize", "--out", str(out / "gen")], env=env, log=out / LOG)
    return synthetic
