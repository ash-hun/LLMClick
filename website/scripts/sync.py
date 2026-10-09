"""Build website/docs from the repository: handwritten intros, sections of the README files, and reference pages
generated from the code. Running it twice writes the same files; nothing under docs/ is edited by hand."""

import re
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "website"
INTROS = SITE / "intros"
DOCS = SITE / "docs"
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
sys.path.insert(0, str(ROOT))

# page -> (intro file, [(source file, heading, own_only)]) ; None heading = the preamble before the first "## "
PAGES: dict[str, tuple[str, list[tuple[str, str | None, bool]]]] = {
    "index": ("index", [("README.md", None, False)]),
    "quick-start": ("quick-start", [("README.md", "03. Quick start", False)]),
    "project-structure": ("project-structure", [("README.md", "01. Project structure", False)]),
    "concepts/how-a-run-works": ("concepts/how-a-run-works", [("README.md", "02. How a run works", True)]),
    "concepts/stages": ("concepts/stages", [("README.md", "Stages", False)]),
    "concepts/idempotency": ("concepts/idempotency", []),
    "modeling/recipes": ("modeling/recipes", [("README.md", "Reusing sampled groups (`llm_grpo`, `llm_decision_cispo`)", False), ("README.md", "A reward for `llm_grpo`", False)]),
    "modeling/decision-recipes": ("modeling/decision-recipes", [("README.md", "Decision recipes", False)]),
    "modeling/extending": ("modeling/extending", [("README.md", "04. Extending", False)]),
    "evaluation/overview": ("evaluation/overview", [("evaluation/README.md", "Purpose", False), ("evaluation/README.md", "How it differs from `validate` and from `training.eval_every`", False)]),
    "evaluation/custom": ("evaluation/custom", [("evaluation/README.md", "Built: `evaluation_custom`", False)]),
    "evaluation/benchmark": ("evaluation/benchmark", [("evaluation/README.md", "Built: `evaluation_benchmark`", False)]),
    "evaluation/compare": ("evaluation/compare", [("evaluation/README.md", "Built: `evaluation_compare`", False)]),
    "evaluation/decision": ("evaluation/decision", [("evaluation/README.md", "Built: `evaluation_decision`", False)]),
    "evaluation/embedding": ("evaluation/embedding", [("evaluation/README.md", "Built: `evaluation_embedding`", False)]),
    "evaluation/contamination": ("evaluation/contamination", [("evaluation/README.md", "Built: contamination, tracker, API", False)]),
    "evaluation/design": ("evaluation/design", [("evaluation/README.md", "Design", False), ("evaluation/README.md", "Plan", False), ("evaluation/README.md", "Known starting points", False)]),
    "evaluation/survey": ("evaluation/survey", [("evaluation/README.md", "Survey: what to measure for 0.6B to 2B fine-tuned models", False), ("evaluation/README.md", "Sources", False)]),
    "data/overview": ("data/overview", [("data/README.md", "Purpose", False)]),
    "data/synthetic": ("data/synthetic", [("data/README.md", "Built: `data_synthetic`", False)]),
    "data/practices": ("data/practices", [("data/README.md", "Reference practices for synthetic quality", True), ("data/README.md", "What `data_synthetic` builds in", False)]),
    "data/lineages": ("data/lineages", [("data/README.md", "Two lineages in detail: Nemotron and WizardLM", False)]),
    "data/design": ("data/design", [("data/README.md", "Design", False), ("data/README.md", "Plan", False), ("data/README.md", "Known starting points", False)]),
    "data/survey": ("data/survey", [("data/README.md", "Survey", False), ("data/README.md", "Sources", False)]),
    "operations/jobs": ("operations/jobs", []),
    "operations/security": ("operations/security", []),
    "contributing/development": ("contributing/development", [("README.md", "05. Tests", False)]),
}
GENERATED = {"reference/recipes", "reference/config", "reference/api", "reference/cli", "reference/settings"}


def headings(lines: list[str]) -> list[tuple[int, int, str]]:
    """(line number, level, text) of every heading outside code fences."""
    found, fenced = [], False
    for number, line in enumerate(lines):
        if line.startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        match = HEADING.match(line)
        if match:
            found.append((number, len(match.group(1)), match.group(2).strip()))
    return found


def section(path: Path, heading: str | None, own_only: bool) -> str:
    """The body under `heading` (without the heading line) up to the next heading of the same or a higher level, or
    up to any next heading with `own_only`; the preamble before the first level-2 heading when `heading` is None."""
    lines = path.read_text().split("\n")
    marks = headings(lines)
    if heading is None:
        first = next((number for number, level, _ in marks if level == 2), len(lines))
        body = [line for line in lines[:first] if not line.startswith("# ")]
        return "\n".join(body).strip() + "\n"
    for index, (number, level, text) in enumerate(marks):
        if text == heading:
            end = len(lines)
            for later_number, later_level, _ in marks[index + 1:]:
                if own_only or later_level <= level:
                    end = later_number
                    break
            return promote("\n".join(lines[number + 1:end]).strip(), level) + "\n"
    raise KeyError(f"{path}: no heading {heading!r}")


def promote(body: str, level: int) -> str:
    """Headings inside an extracted section move up so the page's own level-2 headings come first in the TOC."""
    out, fenced = [], False
    for line in body.split("\n"):
        if line.startswith("```"):
            fenced = not fenced
        match = None if fenced else HEADING.match(line)
        if match and len(match.group(1)) > level:
            line = "#" * max(2, len(match.group(1)) - level + 1) + " " + match.group(2)
        out.append(line)
    return "\n".join(out)


def page(name: str) -> str:
    intro_name, sources = PAGES[name]
    intro = (INTROS / f"{intro_name}.md").read_text().strip()
    parts = [intro]
    for file, heading, own_only in sources:
        parts.append(section(ROOT / file, heading, own_only).strip())
    origin = ", ".join(sorted({file for file, _, _ in sources})) or "handwritten"
    return "\n\n".join(parts) + f"\n\n<p class=\"sourced\">Source: {origin}. This page is generated; edit the source and run <code>npm run sync</code>.</p>\n"


def table(rows: list[list[str]], header: list[str]) -> str:
    cell = lambda value: str(value).replace("|", "\\|").replace("\n", " ")  # noqa: E731
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def reference_recipes() -> str:
    from core.registry import catalogue
    rows = []
    for name, entry in catalogue().items():
        extras = {k: v for k, v in entry.items() if k != "stages"}
        rows.append([f"`{name}`", ", ".join(f"`{s}`" for s in entry["stages"]), "; ".join(f"{k}: {', '.join(f'`{x}`' for x in v)}" for k, v in extras.items())])
    intro = (INTROS / "reference/recipes.md").read_text().strip()
    return intro + "\n\n" + table(rows, ["Recipe", "Stages", "Registry keys the config may use"]) + "\n"


def schema_rows(schema: dict[str, Any], definitions: dict[str, Any], prefix: str = "", depth: int = 0) -> list[list[str]]:
    rows: list[list[str]] = []
    for key, prop in (schema.get("properties") or {}).items():
        resolved, kind = resolve(prop, definitions)
        default = prop.get("default", "")
        if isinstance(default, (dict, list)):
            default = json.dumps(default, ensure_ascii=False)
        required = key in (schema.get("required") or [])
        rows.append([f"`{prefix}{key}`", kind, "required" if required else (f"`{default}`" if default not in ("", None) else ""), prop.get("description", "")])
        if resolved is not None and depth < 3 and resolved.get("properties"):
            rows += schema_rows(resolved, definitions, f"{prefix}{key}.", depth + 1)
    return rows


def resolve(prop: dict[str, Any], definitions: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """The object schema a property points at (if any) and a short type name."""
    if "$ref" in prop:
        name = prop["$ref"].rsplit("/", 1)[-1]
        return definitions.get(name), name
    if "anyOf" in prop:
        names = []
        target = None
        for option in prop["anyOf"]:
            inner, kind = resolve(option, definitions)
            names.append(kind)
            target = target or inner
        return target, " | ".join(names)
    if prop.get("type") == "array":
        inner, kind = resolve(prop.get("items", {}), definitions)
        return inner, f"list[{kind}]"
    if prop.get("type") == "object" and prop.get("additionalProperties"):
        extra = prop["additionalProperties"]
        if isinstance(extra, dict):
            _, kind = resolve(extra, definitions)
            return None, f"dict[str, {kind}]"
        return prop if prop.get("properties") else None, "object (extra keys allowed)"
    if "enum" in prop:
        return None, " | ".join(f"`{v}`" for v in prop["enum"])
    if "const" in prop:
        return None, f"`{prop['const']}`"
    return None, str(prop.get("type", "any"))


def reference_config() -> str:
    from core.registry import RECIPES, load_channels
    load_channels()
    parts = [(INTROS / "reference/config.md").read_text().strip()]
    for name in RECIPES.names():
        config_class = RECIPES.get(name).config_class
        schema = config_class.model_json_schema()
        definitions = schema.get("$defs", {})
        parts.append(f"## `{name}`\n\n{(config_class.__doc__ or '').strip()}\n\n" + table(schema_rows(schema, definitions), ["Key", "Type", "Default", "Description"]))
    return "\n\n".join(parts) + "\n"


def reference_api() -> str:
    from core.api.app import app
    spec = app.openapi()
    parts = [(INTROS / "reference/api.md").read_text().strip()]
    for path, methods in spec["paths"].items():
        for method, operation in methods.items():
            parts.append(f"## `{method.upper()} {path}`\n\n{operation.get('summary', '')}\n\n{operation.get('description', '')}".strip())
            params = operation.get("parameters") or []
            if params:
                parts.append(table([[f"`{p['name']}`", p["in"], "yes" if p.get("required") else "no", p.get("description", p.get("schema", {}).get("description", ""))] for p in params],
                                   ["Parameter", "In", "Required", "Description"]))
            body = (operation.get("requestBody") or {}).get("content", {}).get("application/json", {}).get("schema", {})
            if body:
                parts.append(f"Request body: `{body.get('$ref', body.get('title', 'object')).rsplit('/', 1)[-1]}`")
            responses = operation.get("responses") or {}
            parts.append(table([[code, r.get("description", "")] for code, r in responses.items()], ["Status", "Meaning"]))
    definitions = spec.get("components", {}).get("schemas", {})
    parts.append("## Schemas")
    for name, schema in definitions.items():
        if name.startswith("HTTP") or name == "ValidationError":
            continue
        parts.append(f"### `{name}`\n\n" + table(schema_rows(schema, definitions), ["Field", "Type", "Default", "Description"]))
    return "\n\n".join(parts) + "\n"


def reference_cli() -> str:
    parts = [(INTROS / "reference/cli.md").read_text().strip()]
    for command in ([], ["run"], ["validate"], ["recipes"], ["serve"]):
        output = subprocess.run([sys.executable, "-m", "core.cli", *command, "--help"], capture_output=True, text=True, cwd=ROOT, check=True).stdout
        parts.append(f"## `llmclick {' '.join(command)}`".replace("`llmclick `", "`llmclick`") + "\n\n```text\n" + output.strip() + "\n```")
    return "\n\n".join(parts) + "\n"


def reference_settings() -> str:
    from core.settings import Settings
    rows = [[f"`{name}`", f"`{field.default}`" if field.default not in ("", None) else "", description_of(name)] for name, field in Settings.model_fields.items()]
    return (INTROS / "reference/settings.md").read_text().strip() + "\n\n" + table(rows, ["Variable", "Default", "What it does"]) + "\n"


def description_of(name: str) -> str:
    """The trailing comment on the setting's line in core/settings.py: the code is the documentation."""
    for line in (ROOT / "core" / "settings.py").read_text().split("\n"):
        if line.strip().startswith(f"{name}:") and "#" in line:
            return line.split("#", 1)[1].strip()
    return ""


def main() -> int:
    if DOCS.exists():
        shutil.rmtree(DOCS)
    for name in PAGES:
        target = DOCS / f"{name}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(page(name))
    generators = {"reference/recipes": reference_recipes, "reference/config": reference_config, "reference/api": reference_api,
                  "reference/cli": reference_cli, "reference/settings": reference_settings}
    for name, generate in generators.items():
        target = DOCS / f"{name}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(generate())
    print(f"wrote {len(PAGES) + len(generators)} pages under {DOCS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
