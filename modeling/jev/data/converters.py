"""Converters turn one raw dataset row into a jeff Example; `classification` and `boolean` need no code, only YAML."""

from typing import Any, cast

from jeff.types import Example
from modeling.jev.registry import CONVERTERS

Converter = Any  # (raw: dict, index: int, params: dict) -> Example | None

REQUIRED = ("id", "suite", "family", "state", "question", "label", "target", "source")


def state_text(raw: dict[str, Any], fields: list[str]) -> str:
    """One field is used verbatim; several are labelled lines, the layout jeff's converters and panel use."""
    if len(fields) == 1:
        return str(raw[fields[0]])
    return "\n".join(f"{field.replace('_', ' ').capitalize()}: {raw[field]}" for field in fields)


@CONVERTERS.register("example")
def example(raw: dict[str, Any], index: int, params: dict[str, Any]) -> Example | None:
    """The row already is an Example (for example a JSONL published by another run)."""
    missing = [key for key in REQUIRED if key not in raw]
    if missing:
        raise ValueError(f"Row {index} lacks Example keys {missing}")
    return raw  # type: ignore[return-value]


@CONVERTERS.register("classification")
def classification(raw: dict[str, Any], index: int, params: dict[str, Any]) -> Example | None:
    """params: suite, text_fields, label_field, criteria {key: description}, instructions, label_map?, family_field?"""
    label = str(raw[params["label_field"]])
    label = str(params.get("label_map", {}).get(label, label))
    criteria: dict[str, str | None] = params["criteria"]
    if label not in criteria:
        return None
    suite = params["suite"]
    identifier = f"{suite}-{index:07d}"
    return {
        "id": identifier, "suite": suite, "family": str(raw.get(params.get("family_field", ""), identifier)),
        "state": state_text(raw, params["text_fields"]),
        "question": {"type": "choice", "instructions": params.get("instructions"), "criteria": dict(criteria)},
        "label": label, "target": label,
        "source": {"dataset": params.get("dataset", suite), "converter": "classification", "index": index},
    }


@CONVERTERS.register("boolean")
def boolean(raw: dict[str, Any], index: int, params: dict[str, Any]) -> Example | None:
    """params: suite, text_fields, label_field, instructions, criteria {true: .., false: ..}?, true_values?"""
    value = raw[params["label_field"]]
    true_values = {str(v).lower() for v in params.get("true_values", [True, 1, "1", "true", "yes"])}
    label = str(value).lower() in true_values
    suite = params["suite"]
    identifier = f"{suite}-{index:07d}"
    question: dict[str, Any] = {"type": "noul", "instructions": params.get("instructions")}
    if params.get("criteria"):
        question["criteria"] = dict(params["criteria"])
    return cast(Example, {
        "id": identifier, "suite": suite, "family": str(raw.get(params.get("family_field", ""), identifier)),
        "state": state_text(raw, params["text_fields"]), "question": question,
        "label": label, "target": label,
        "source": {"dataset": params.get("dataset", suite), "converter": "boolean", "index": index},
    })
