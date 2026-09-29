"""Build the frozen evaluation panel from pinned public benchmarks; scoring only, never selection."""

import argparse
import json
import random
import re
import zipfile
from collections.abc import Callable
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from jeff.data import validate, write_rows
from jeff.model import decision_messages
from jeff.types import Example, JSONValue

SEED = 20260926
MAX_TOKENS = 8000  # Below the 8192 model limit to leave room for the answer-code prefix.
STUDENT = ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17")
SOURCES = {
    "BBH": ("lukaemon/bbh", "982bb89fd79532a8ac676a61fc42eb1aeec63f99"),
    "Financial PhraseBank": ("takala/financial_phrasebank", "8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964"),
    "JudgeBench": ("ScalerLab/JudgeBench", "57dd5e0b9817d07f05ec8f45a91b2ce1e310e308"),
    "RAGTruth": ("wandb/RAGTruth-processed", "eb4f4b9d1b68eb7092d3e1a61c0cd82d9808737b"),
    "WinoGrande": ("allenai/winogrande", "01e74176c63542e6b0bcb004dcdea22d94fb67b5"),
}
COUNTS = {"BBH": 750, "Financial PhraseBank": 999, "JudgeBench": 350, "RAGTruth": 1500, "WinoGrande": 1000}
# The 15 BBH tasks in AutoJev's published results, 50 rows each.
BBH_TASKS = ("boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa", "formal_fallacies",
             "hyperbaton", "logical_deduction_five_objects", "logical_deduction_seven_objects", "navigate",
             "penguins_in_a_table", "reasoning_about_colored_objects", "salient_translation_error_detection",
             "snarks", "temporal_sequences", "tracking_shuffled_objects_five_objects")
BBH_PER_TASK = 50
BBH_INSTRUCTIONS = "Solve the problem described in the state and choose the correct answer."
LETTERED = re.compile(r"^\(([A-Z])\) (.*)$")


def example(suite: str, key: str, family: str, state: JSONValue, question: dict[str, JSONValue],
            label: str | bool, source: dict[str, JSONValue]) -> Example:
    dataset, revision = SOURCES[suite]
    return {"id": f"panel-{key}", "suite": suite, "family": f"panel-{family}", "state": state,  # type: ignore[typeddict-item]
            "question": question, "label": label, "target": label,
            "source": {"dataset": dataset, "revision": revision, **source}}


def bbh_row(task: str, index: int, raw: dict[str, str]) -> Example:
    text, target = raw["input"], raw["target"].strip()
    if "\nOptions:\n" in text:
        state, block = text.split("\nOptions:\n", 1)
        criteria: dict[str, str | None] = {}
        for raw_line in block.strip().splitlines():
            line = raw_line.strip()
            if match := LETTERED.match(line):
                criteria[match.group(1)] = match.group(2).strip()
            elif line.startswith("- "):
                criteria[line[2:].strip()] = None
            else:
                raise ValueError(f"BBH {task} row {index}: unrecognised option line {line!r}")
        label = target[1:-1] if re.fullmatch(r"\([A-Z]\)", target) else target
    elif task == "boolean_expressions":
        state, criteria, label = text, {"True": None, "False": None}, target
    else:
        raise ValueError(f"BBH {task} row {index}: no options block and no known answer set")
    if label not in criteria:
        raise ValueError(f"BBH {task} row {index}: target {target!r} is not one of {list(criteria)}")
    question = {"type": "choice", "instructions": BBH_INSTRUCTIONS, "criteria": criteria}
    return example("BBH", f"bbh-{task}-{index}", f"bbh-{task}-{index}", state, question, label,  # type: ignore[arg-type]
                   {"task": task, "split": "test", "index": index})


FPB_CRITERIA = {
    "positive": "The news is likely to improve the company's financial position or share price.",
    "negative": "The news is likely to harm the company's financial position or share price.",
    "neutral": "The news has no clear positive or negative effect on the company.",
}


def fpb_row(index: int, line: str) -> Example:
    sentence, _, label = line.rstrip("\n").rpartition("@")
    if label not in FPB_CRITERIA or not sentence:
        raise ValueError(f"Financial PhraseBank line {index}: cannot parse {line!r}")
    question = {"type": "choice", "criteria": dict(FPB_CRITERIA),
                "instructions": "From an investor's point of view, classify the sentiment of this financial news sentence."}
    return example("Financial PhraseBank", f"fpb-{index}", f"fpb-{index}", sentence, question, label,  # type: ignore[arg-type]
                   {"config": "sentences_allagree", "index": index})


def judgebench_row(raw: dict[str, str]) -> Example:
    labels = {"A>B": "A", "B>A": "B"}
    if raw["label"] not in labels:
        raise ValueError(f"JudgeBench {raw['pair_id']}: unsupported label {raw['label']}")
    question = {"type": "choice", "instructions": "Which response answers the question more correctly?",
                "criteria": {"A": "Response A is more correct.", "B": "Response B is more correct."}}
    state = {"question": raw["question"], "response_A": raw["response_A"], "response_B": raw["response_B"]}
    return example("JudgeBench", f"judgebench-{raw['pair_id']}", f"judgebench-{raw['pair_id']}", state,  # type: ignore[arg-type]
                   question, labels[raw["label"]], {"split": "gpt", "pair_id": raw["pair_id"], "origin": raw["source"]})


def ragtruth_row(raw: dict[str, str]) -> Example:
    spans = json.loads(raw["hallucination_labels"])
    if not isinstance(spans, list):
        raise ValueError(f"RAGTruth {raw['id']}: hallucination_labels is not a list")
    question = {"type": "noul",
                "instructions": "Does the response contain any information that conflicts with the source or is not supported by it?",
                "criteria": {"true": "The response contains conflicting or unsupported information.",
                             "false": "Every claim in the response is supported by the source."}}
    state = {"task": raw["task_type"], "instruction": raw["query"], "source": raw["context"], "response": raw["output"]}
    return example("RAGTruth", f"ragtruth-{raw['id']}", f"ragtruth-{raw['id']}", state, question,  # type: ignore[arg-type]
                   bool(spans), {"split": "test", "original_id": raw["id"]})


def winogrande_row(index: int, raw: dict[str, str]) -> Example:
    if raw["answer"] not in {"1", "2"}:
        raise ValueError(f"WinoGrande row {index}: answer {raw['answer']!r}")
    question = {"type": "choice", "instructions": "Which option correctly fills the blank (_) in the sentence?",
                "criteria": {"1": raw["option1"], "2": raw["option2"]}}
    return example("WinoGrande", f"winogrande-{index}", f"winogrande-{index}", raw["sentence"], question,  # type: ignore[arg-type]
                   raw["answer"], {"config": "winogrande_xl", "split": "validation", "index": index})


def sample(rows: list[Example], count: int, seed: int, fits: Callable[[Example], bool]) -> tuple[list[Example], int]:
    """Drop rows over the token limit first, then draw a fixed-seed sample."""
    usable = [row for row in rows if fits(row)]
    if len(usable) < count:
        raise ValueError(f"{rows[0]['suite']}: only {len(usable)} rows fit the token limit, {count} needed")
    ordered = sorted(usable, key=lambda row: row["id"])
    random.Random(f"{seed}-{rows[0]['suite']}").shuffle(ordered)
    return sorted(ordered[:count], key=lambda row: row["id"]), len(rows) - len(usable)


def download(suite: str, filename: str) -> Path:
    repo, revision = SOURCES[suite]
    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))


def parquet(suite: str, filename: str) -> list[dict[str, str]]:
    return pq.read_table(download(suite, filename)).to_pylist()


def load_all() -> dict[str, list[Example]]:
    bbh: list[Example] = []
    for task in BBH_TASKS:
        bbh.extend(bbh_row(task, index, raw) for index, raw in enumerate(parquet("BBH", f"{task}/test-00000-of-00001.parquet")))
    with zipfile.ZipFile(download("Financial PhraseBank", "data/FinancialPhraseBank-v1.0.zip")) as archive:
        lines = archive.read("FinancialPhraseBank-v1.0/Sentences_AllAgree.txt").decode("latin-1").splitlines()
    judge = [json.loads(line) for line in download("JudgeBench", "data/gpt-00000-of-00001.jsonl").read_text().splitlines()]
    return {
        "BBH": bbh,
        "Financial PhraseBank": [fpb_row(index, line) for index, line in enumerate(lines) if line.strip()],
        "JudgeBench": [judgebench_row(raw) for raw in judge],
        "RAGTruth": [ragtruth_row(raw) for raw in parquet("RAGTruth", "data/test-00000-of-00001.parquet")],
        "WinoGrande": [winogrande_row(index, raw) for index, raw in
                       enumerate(parquet("WinoGrande", "winogrande_xl/validation-00000-of-00001.parquet"))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/panel.jsonl"))
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(STUDENT[0], revision=STUDENT[1])
    codes = [chr(ord("A") + index) for index in range(26)]

    def fits(row: Example) -> bool:
        text = tokenizer.apply_chat_template(decision_messages(row, codes), tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        return len(tokenizer.encode(text, add_special_tokens=False)) <= MAX_TOKENS

    loaded = load_all()
    rows: list[Example] = []
    dropped: dict[str, int] = {}
    for suite, count in COUNTS.items():
        if suite == "BBH":
            for task in BBH_TASKS:
                chosen, lost = sample([row for row in loaded[suite] if row["source"]["task"] == task], BBH_PER_TASK, SEED, fits)
                rows.extend(chosen)
                dropped[f"BBH/{task}"] = lost
        else:
            chosen, dropped[suite] = sample(loaded[suite], count, SEED, fits)
            rows.extend(chosen)
    validate(rows)
    if len(rows) != sum(COUNTS.values()):
        raise ValueError(f"Panel has {len(rows)} rows, expected {sum(COUNTS.values())}")
    digest = write_rows(args.output, rows)
    manifest = {"sha256": digest, "rows": len(rows), "seed": SEED, "max_tokens": MAX_TOKENS, "student_tokenizer": list(STUDENT),
                "sources": {suite: list(value) for suite, value in SOURCES.items()}, "counts": COUNTS,
                "dropped_for_length": dropped, "bbh_tasks": list(BBH_TASKS)}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
