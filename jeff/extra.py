"""Convert public Hugging Face datasets into the training-row (Example) format.

Every converter below is a pure function: it takes one raw record from the
source dataset and returns an `Example` (or `None` when the record has no
usable label, such as an SNLI row annotators never agreed on). Downloading,
sampling, validating and writing are handled by `main()`.
"""

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from jeff import panel
from jeff.data import validate, write_rows
from jeff.types import Example, JSONValue

SEED = 20260926
CAP = 5000
TRUTHFULQA_MAX_OPTIONS = 10

MULTIPLE_CHOICE_INSTRUCTIONS = "Choose the option that correctly answers the question."


@dataclass(frozen=True)
class Source:
    repository: str
    revision: str
    path: str
    license: str
    split: str
    config: str | None = None
    requested_revision: str | None = None
    note: str = ""


SOURCES: dict[str, Source] = {
    "snli": Source(
        repository="stanfordnlp/snli", revision="cdb5c3d5eed6ead6e5a341c8e56e669bb666725b",
        config="plain_text", split="train", path="plain_text/train-00000-of-00001.parquet",
        license="cc-by-sa-4.0"),
    "commonsense_qa": Source(
        repository="tau/commonsense_qa", revision="94630fe30dad47192a8546eb75f094926d47e155",
        split="train", path="data/train-00000-of-00001.parquet", license="mit"),
    "openbookqa": Source(
        repository="allenai/openbookqa", revision="388097ea7776314e93a529163e0fea805b8a6454",
        config="main", split="train", path="main/train-00000-of-00001.parquet", license="unknown"),
    "arc_challenge": Source(
        repository="allenai/ai2_arc", revision="210d026faf9955653af8916fad021475a3f00453",
        config="ARC-Challenge", split="train", path="ARC-Challenge/train-00000-of-00001.parquet",
        license="cc-by-sa-4.0"),
    "arc_easy": Source(
        repository="allenai/ai2_arc", revision="210d026faf9955653af8916fad021475a3f00453",
        config="ARC-Easy", split="train", path="ARC-Easy/train-00000-of-00001.parquet",
        license="cc-by-sa-4.0"),
    "social_iqa": Source(
        repository="allenai/social_i_qa", revision="537a2ec8ec565adc0b70b70752893e59e024df26",
        requested_revision="8835ceb9141d7896d9d968634a9b21ae440e3ec5",
        config="default", split="train", path="default/train/0000.parquet", license="cc-by-4.0",
        note=("The pinned revision only contains a legacy Python loading script "
              "(social_i_qa.py) that downloads from storage.googleapis.com; datasets 5.0.1 "
              "refuses to execute dataset loading scripts (\"Dataset scripts are no longer "
              "supported\"). Read instead from Hugging Face's own auto-converted Parquet "
              "mirror of that exact same script's output (ref refs/convert/parquet, commit "
              "537a2ec8), which is the only way this dataset's data can currently be fetched "
              "from the Hub.")),
    "cosmos_qa": Source(
        repository="allenai/cosmos_qa", revision="ed50fe83db62b355759053b1cd5eef344ee371e0",
        requested_revision="28d9d5e2aae025e73e11177891a88dba51190013",
        config="default", split="train", path="default/train/0000.parquet", license="cc-by-4.0",
        note=("Same situation as social_i_qa: the pinned revision is a Python loading script "
              "only (cosmos_qa.py, downloading from a GitHub raw URL). Read from Hugging "
              "Face's auto-converted Parquet mirror (ref refs/convert/parquet, commit "
              "ed50fe83) instead.")),
    "quartz": Source(
        repository="allenai/quartz", revision="28c1dbb56caf81799296cb17892fa73402e23464",
        split="train", path="data/train-00000-of-00001.parquet", license="cc-by-4.0"),
    "qasc": Source(
        repository="allenai/qasc", revision="a34ba204eb9a33b919c10cc08f4f1c8dae5ec070",
        split="train", path="data/train-00000-of-00001.parquet", license="cc-by-4.0"),
    "truthful_qa": Source(
        repository="truthfulqa/truthful_qa", revision="741b8276f2d1982aa3d5b832d3ee81ed3b896490",
        config="multiple_choice", split="validation",
        path="multiple_choice/validation-00000-of-00001.parquet", license="apache-2.0",
        note="multiple_choice has only a validation split; there is no train split to pin."),
    "twitter_financial": Source(
        repository="zeroshot/twitter-financial-news-sentiment",
        revision="ccbe24de388e287beb92dd393a335c376b350ac3",
        split="train", path="sent_train.csv", license="mit"),
    "liar2": Source(
        repository="chengxuphd/liar2", revision="12ad1a68e4256d0fab1701396e18873393cdc248",
        split="train", path="train.csv", license="apache-2.0"),
    "halueval_qa": Source(
        repository="pminervini/HaluEval", revision="12a856119f03975a94509091e8cada3e6be6ead7",
        config="qa_samples", split="data", path="qa_samples/data-00000-of-00001.parquet",
        license="apache-2.0", note="Each qa_samples/dialogue_samples/summarization_samples config has only a split named 'data'."),
    "halueval_dialogue": Source(
        repository="pminervini/HaluEval", revision="12a856119f03975a94509091e8cada3e6be6ead7",
        config="dialogue_samples", split="data", path="dialogue_samples/data-00000-of-00001.parquet",
        license="apache-2.0"),
    "halueval_summarization": Source(
        repository="pminervini/HaluEval", revision="12a856119f03975a94509091e8cada3e6be6ead7",
        config="summarization_samples", split="data",
        path="summarization_samples/data-00000-of-00001.parquet", license="apache-2.0"),
    "wikibio": Source(
        repository="potsawee/wiki_bio_gpt3_hallucination", revision="b3cfb73209a8c51582fa1d9b7fe7e45fec5529b2",
        split="evaluation", path="data/evaluation-00000-of-00001-e91191b8ff41afbe.parquet",
        license="cc-by-sa-3.0", note="This dataset has only an 'evaluation' split; there is no train split to pin."),
    # Training splits of two panel benchmarks. The panel scores their test (RAGTruth) and validation (WinoGrande)
    # splits only; these splits are disjoint from those, and the mix's leak filter still drops any overlap.
    "ragtruth_train": Source(
        repository="wandb/RAGTruth-processed", revision=panel.SOURCES["RAGTruth"][1],
        split="train", path="data/train-00000-of-00001.parquet", license="mit",
        note="Training split of the benchmark whose test split is in the panel. Licence as published for RAGTruth (MIT); confirm before release."),
    "winogrande_train": Source(
        repository="allenai/winogrande", revision=panel.SOURCES["WinoGrande"][1], config="winogrande_xl",
        split="train", path="winogrande_xl/train-00000-of-00001.parquet", license="cc-by",
        note="Training split of the benchmark whose validation split is in the panel. Licence as published (CC-BY); confirm before release."),
}
# Datasets allowed more rows than CAP: all of RAGTruth's and WinoGrande's training splits.
CAPS = {"ragtruth_train": 15090, "winogrande_train": 40398}

SKIPPED = {
    "banking77": (
        "PolyAI/banking77 @ 90d4e2ee5521c04fc1488f065b8b083658768c57 has no data files at all: "
        "the repository holds only a Python loading script (banking77.py, which downloads CSVs "
        "from a GitHub raw URL) and dataset_infos.json. Hugging Face has not auto-converted it "
        "to Parquet (the repo's /api/datasets/PolyAI/banking77/refs response lists "
        "\"converts\": []), and the installed datasets library refuses to execute loading "
        "scripts (\"Dataset scripts are no longer supported\", confirmed via the "
        "datasets-server API). Neither of the two sanctioned download methods "
        "(hf_hub_download of a data file, or `datasets` with revision=) can retrieve this "
        "dataset's rows, so conversion was skipped rather than fetching the data from GitHub "
        "directly."
    ),
}


def make_id(name: str, key: str) -> str:
    return f"extra-{name}-{key}"


def describe(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def download(source: Source) -> Path:
    return Path(hf_hub_download(source.repository, source.path, repo_type="dataset", revision=source.revision))


def parquet_rows(source: Source) -> list[dict[str, JSONValue]]:
    return cast(list[dict[str, JSONValue]], pq.read_table(download(source)).to_pylist())


def csv_rows(source: Source) -> list[dict[str, str]]:
    with download(source).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


# --- snli --------------------------------------------------------------

SNLI_CRITERIA = {
    "entailment": "The premise supports (entails) the hypothesis.",
    "neutral": "The premise neither supports nor contradicts the hypothesis.",
    "contradiction": "The premise contradicts the hypothesis.",
}
SNLI_LABELS = ["entailment", "neutral", "contradiction"]
SNLI_INSTRUCTIONS = "Does the premise entail the hypothesis, contradict it, or neither?"


def convert_snli(index: int, raw: dict[str, JSONValue]) -> Example | None:
    label_index = cast(int, raw["label"])
    if label_index not in (0, 1, 2):
        return None  # Annotators never agreed on a label (SNLI marks this -1).
    label = SNLI_LABELS[label_index]
    identifier = make_id("snli", str(index))
    state = f"Premise: {raw['premise']}\nHypothesis: {raw['hypothesis']}"
    return {
        "id": identifier, "suite": "snli", "family": identifier, "state": state,
        "question": {"type": "choice", "instructions": SNLI_INSTRUCTIONS, "criteria": dict(SNLI_CRITERIA)},
        "label": label, "target": label,
        "source": {"dataset": "snli", "hf_id": "stanfordnlp/snli", "revision": SOURCES["snli"].revision,
                   "split": "train", "license": SOURCES["snli"].license, "original_index": index},
    }


# --- commonsense_qa / openbookqa / arc_challenge / arc_easy -------------

def _multiple_choice_row(name: str, raw: dict[str, JSONValue], question_field: str) -> Example:
    original_id = str(raw["id"])
    identifier = make_id(name, original_id)
    choices = cast(dict[str, JSONValue], raw["choices"])
    criteria = dict(zip(cast(list[str], choices["label"]), cast(list[str], choices["text"])))
    answer_key = str(raw["answerKey"])
    if answer_key not in criteria:
        raise ValueError(f"{name} row {original_id}: answerKey {answer_key!r} is not one of {list(criteria)}")
    source = SOURCES[name]
    return {
        "id": identifier, "suite": name, "family": identifier, "state": cast(str, raw[question_field]),
        "question": {"type": "choice", "instructions": MULTIPLE_CHOICE_INSTRUCTIONS, "criteria": criteria},
        "label": answer_key, "target": answer_key,
        "source": {"dataset": name, "hf_id": source.repository, "revision": source.revision,
                   "config": source.config, "split": "train", "license": source.license,
                   "original_id": original_id},
    }


def convert_commonsense_qa(raw: dict[str, JSONValue]) -> Example:
    return _multiple_choice_row("commonsense_qa", raw, "question")


def convert_openbookqa(raw: dict[str, JSONValue]) -> Example:
    return _multiple_choice_row("openbookqa", raw, "question_stem")


def convert_arc_challenge(raw: dict[str, JSONValue]) -> Example:
    return _multiple_choice_row("arc_challenge", raw, "question")


def convert_arc_easy(raw: dict[str, JSONValue]) -> Example:
    return _multiple_choice_row("arc_easy", raw, "question")


# --- social_iqa ----------------------------------------------------------

def convert_social_iqa(index: int, raw: dict[str, JSONValue]) -> Example:
    identifier = make_id("social_iqa", str(index))
    label = str(raw["label"])
    criteria = {"1": cast(str, raw["answerA"]), "2": cast(str, raw["answerB"]), "3": cast(str, raw["answerC"])}
    if label not in criteria:
        raise ValueError(f"social_iqa row {index}: label {label!r} is not one of {list(criteria)}")
    source = SOURCES["social_iqa"]
    return {
        "id": identifier, "suite": "social_iqa", "family": identifier, "state": cast(str, raw["context"]),
        "question": {"type": "choice", "instructions": cast(str, raw["question"]), "criteria": criteria},
        "label": label, "target": label,
        "source": {"dataset": "social_iqa", "hf_id": "allenai/social_i_qa", "revision": source.revision,
                   "split": "train", "license": source.license, "original_index": index},
    }


# --- cosmos_qa -------------------------------------------------------------

def convert_cosmos_qa(raw: dict[str, JSONValue]) -> Example:
    original_id = str(raw["id"])
    identifier = make_id("cosmos_qa", original_id)
    label = str(raw["label"])
    criteria = {str(i): cast(str, raw[f"answer{i}"]) for i in range(4)}
    if label not in criteria:
        raise ValueError(f"cosmos_qa row {original_id}: label {label!r} is not one of {list(criteria)}")
    source = SOURCES["cosmos_qa"]
    return {
        "id": identifier, "suite": "cosmos_qa", "family": identifier, "state": cast(str, raw["context"]),
        "question": {"type": "choice", "instructions": cast(str, raw["question"]), "criteria": criteria},
        "label": label, "target": label,
        "source": {"dataset": "cosmos_qa", "hf_id": "allenai/cosmos_qa", "revision": source.revision,
                   "split": "train", "license": source.license, "original_id": original_id},
    }


# --- quartz ----------------------------------------------------------------

def convert_quartz(raw: dict[str, JSONValue]) -> Example:
    original_id = str(raw["id"])
    identifier = make_id("quartz", original_id)
    choices = cast(dict[str, JSONValue], raw["choices"])
    criteria = dict(zip(cast(list[str], choices["label"]), cast(list[str], choices["text"])))
    answer_key = str(raw["answerKey"])
    if answer_key not in criteria:
        raise ValueError(f"quartz row {original_id}: answerKey {answer_key!r} is not one of {list(criteria)}")
    source = SOURCES["quartz"]
    return {
        "id": identifier, "suite": "quartz", "family": identifier, "state": cast(str, raw["para"]),
        "question": {"type": "choice", "instructions": cast(str, raw["question"]), "criteria": criteria},
        "label": answer_key, "target": answer_key,
        "source": {"dataset": "quartz", "hf_id": source.repository, "revision": source.revision,
                   "split": "train", "license": source.license, "original_id": original_id},
    }


# --- qasc --------------------------------------------------------------------

def convert_qasc(raw: dict[str, JSONValue]) -> Example:
    original_id = str(raw["id"])
    identifier = make_id("qasc", original_id)
    choices = cast(dict[str, JSONValue], raw["choices"])
    criteria = dict(zip(cast(list[str], choices["label"]), cast(list[str], choices["text"])))
    answer_key = str(raw["answerKey"])
    if answer_key not in criteria:
        raise ValueError(f"qasc row {original_id}: answerKey {answer_key!r} is not one of {list(criteria)}")
    state = f"{raw['fact1']} {raw['fact2']}"
    source = SOURCES["qasc"]
    return {
        "id": identifier, "suite": "qasc", "family": identifier, "state": state,
        "question": {"type": "choice", "instructions": cast(str, raw["question"]), "criteria": criteria},
        "label": answer_key, "target": answer_key,
        "source": {"dataset": "qasc", "hf_id": source.repository, "revision": source.revision,
                   "split": "train", "license": source.license, "original_id": original_id},
    }


# --- truthful_qa ---------------------------------------------------------------

TRUTHFULQA_INSTRUCTIONS = "Choose the answer to the question that is factually correct."


def convert_truthful_qa(index: int, raw: dict[str, JSONValue]) -> Example:
    identifier = make_id("truthful_qa", str(index))
    targets = cast(dict[str, JSONValue], raw["mc1_targets"])
    choices = cast(list[str], targets["choices"])
    labels = cast(list[int], targets["labels"])
    correct_positions = [position for position, value in enumerate(labels) if value == 1]
    if len(correct_positions) != 1:
        raise ValueError(f"truthful_qa row {index}: expected exactly one correct mc1 choice, found {len(correct_positions)}")
    correct = correct_positions[0]
    order = list(range(len(choices)))
    if len(order) > TRUTHFULQA_MAX_OPTIONS:
        others = [position for position in order if position != correct]
        random.Random(f"{SEED}-truthful_qa-cap-{index}").shuffle(others)
        order = [correct] + others[:TRUTHFULQA_MAX_OPTIONS - 1]
    random.Random(f"{SEED}-truthful_qa-shuffle-{index}").shuffle(order)
    criteria = {str(position): choices[original] for position, original in enumerate(order)}
    label = str(order.index(correct))
    source = SOURCES["truthful_qa"]
    return {
        "id": identifier, "suite": "truthful_qa", "family": identifier, "state": cast(str, raw["question"]),
        "question": {"type": "choice", "instructions": TRUTHFULQA_INSTRUCTIONS, "criteria": criteria},
        "label": label, "target": label,
        "source": {"dataset": "truthful_qa", "hf_id": source.repository, "revision": source.revision,
                   "config": source.config, "split": source.split, "license": source.license,
                   "original_index": index, "options_before_cap": len(choices)},
    }


# --- twitter_financial -----------------------------------------------------------

TWITTER_CRITERIA = {
    "bearish": "The tweet expresses a negative outlook: the price or company is expected to fall or perform badly.",
    "bullish": "The tweet expresses a positive outlook: the price or company is expected to rise or perform well.",
    "neutral": "The tweet is informational and does not express a clearly positive or negative outlook.",
}
TWITTER_LABELS = ["bearish", "bullish", "neutral"]
TWITTER_INSTRUCTIONS = "From an investor's point of view, what is the sentiment of this financial tweet?"


def convert_twitter_financial(index: int, raw: dict[str, str]) -> Example:
    label_index = int(raw["label"])
    label = TWITTER_LABELS[label_index]
    identifier = make_id("twitter_financial", str(index))
    source = SOURCES["twitter_financial"]
    return {
        "id": identifier, "suite": "twitter_financial", "family": identifier, "state": raw["text"],
        "question": {"type": "choice", "instructions": TWITTER_INSTRUCTIONS, "criteria": dict(TWITTER_CRITERIA)},
        "label": label, "target": label,
        "source": {"dataset": "twitter_financial", "hf_id": source.repository, "revision": source.revision,
                   "split": "train", "license": source.license, "original_index": index},
    }


# --- liar2 -----------------------------------------------------------------------

LIAR2_LEVELS = ["pants_fire", "false", "barely_true", "half_true", "mostly_true", "true"]
LIAR2_CRITERIA = {
    "pants_fire": "The statement is false and makes a ridiculous claim.",
    "false": "The statement is false.",
    "barely_true": "The statement contains a little truth but ignores critical facts that would give a different impression.",
    "half_true": "The statement is partially accurate but leaves out important details or takes things out of context.",
    "mostly_true": "The statement is accurate but needs clarification or additional information.",
    "true": "The statement is accurate and there is nothing significant missing.",
}
LIAR2_INSTRUCTIONS = "How truthful is this statement, given the speaker and context if provided?"


def convert_liar2(raw: dict[str, str]) -> Example:
    original_id = raw["id"]
    identifier = make_id("liar2", original_id)
    label_index = int(raw["label"])
    label = LIAR2_LEVELS[label_index]
    parts = [raw["statement"]]
    if raw.get("speaker"):
        parts.append(f"Speaker: {raw['speaker']}")
    if raw.get("context"):
        parts.append(f"Context: {raw['context']}")
    source = SOURCES["liar2"]
    return {
        "id": identifier, "suite": "liar2", "family": identifier, "state": "\n".join(parts),
        "question": {"type": "choice", "instructions": LIAR2_INSTRUCTIONS, "criteria": dict(LIAR2_CRITERIA)},
        "label": label, "target": label,
        "source": {"dataset": "liar2", "hf_id": source.repository, "revision": source.revision,
                   "split": "train", "license": source.license, "original_id": original_id},
    }


# --- HaluEval (qa / dialogue / summarization) -------------------------------------

HALUEVAL_INSTRUCTIONS = "Does the response contain information that is not supported by the knowledge/document?"
HALUEVAL_CRITERIA = {
    "true": "The response contains information that is not supported by the knowledge or document.",
    "false": "The response is fully supported by the knowledge or document.",
}


def _halueval_row(name: str, index: int, state: str, hallucination: str) -> Example:
    if hallucination not in ("yes", "no"):
        raise ValueError(f"{name} row {index}: unexpected hallucination value {hallucination!r}")
    identifier = make_id(name, str(index))
    source = SOURCES[name]
    return {
        "id": identifier, "suite": name, "family": identifier, "state": state,
        "question": {"type": "noul", "instructions": HALUEVAL_INSTRUCTIONS, "criteria": dict(HALUEVAL_CRITERIA)},
        "label": hallucination == "yes", "target": hallucination == "yes",
        "source": {"dataset": name, "hf_id": source.repository, "revision": source.revision,
                   "config": source.config, "split": source.split, "license": source.license,
                   "original_index": index},
    }


def convert_halueval_qa(index: int, raw: dict[str, JSONValue]) -> Example:
    state = f"Knowledge:\n{raw['knowledge']}\n\nQuestion: {raw['question']}\n\nResponse: {raw['answer']}"
    return _halueval_row("halueval_qa", index, state, cast(str, raw["hallucination"]))


def convert_halueval_dialogue(index: int, raw: dict[str, JSONValue]) -> Example:
    state = (f"Knowledge:\n{raw['knowledge']}\n\nDialogue history: {raw['dialogue_history']}"
             f"\n\nResponse: {raw['response']}")
    return _halueval_row("halueval_dialogue", index, state, cast(str, raw["hallucination"]))


def convert_halueval_summarization(index: int, raw: dict[str, JSONValue]) -> Example:
    state = f"Document:\n{raw['document']}\n\nSummary: {raw['summary']}"
    return _halueval_row("halueval_summarization", index, state, cast(str, raw["hallucination"]))


# --- wiki_bio_gpt3_hallucination (one row per generated sentence) -----------------

WIKIBIO_INSTRUCTIONS = "Does this sentence contain information that is not supported by the reference biography?"
WIKIBIO_CRITERIA = {
    "true": "The sentence is inaccurate (minor or major) compared with the reference biography.",
    "false": "The sentence is accurate compared with the reference biography.",
}
WIKIBIO_INACCURATE = {"minor_inaccurate", "major_inaccurate"}


def convert_wikibio(raw: dict[str, JSONValue]) -> list[Example]:
    bio_index = raw["wiki_bio_test_idx"]
    family = make_id("wikibio", str(bio_index))
    sentences = cast(list[str], raw["gpt3_sentences"])
    annotations = cast(list[str], raw["annotation"])
    if len(sentences) != len(annotations):
        raise ValueError(f"wikibio bio {bio_index}: {len(sentences)} sentences but {len(annotations)} annotations")
    source = SOURCES["wikibio"]
    rows: list[Example] = []
    for sentence_index, (sentence, annotation) in enumerate(zip(sentences, annotations)):
        if annotation not in WIKIBIO_INACCURATE and annotation != "accurate":
            raise ValueError(f"wikibio bio {bio_index} sentence {sentence_index}: unexpected annotation {annotation!r}")
        identifier = make_id("wikibio", f"{bio_index}-{sentence_index}")
        state = f"Reference biography:\n{raw['wiki_bio_text']}\n\nSentence to check: {sentence}"
        label = annotation in WIKIBIO_INACCURATE
        rows.append({
            "id": identifier, "suite": "wikibio", "family": family, "state": state,
            "question": {"type": "noul", "instructions": WIKIBIO_INSTRUCTIONS, "criteria": dict(WIKIBIO_CRITERIA)},
            "label": label, "target": label,
            "source": {"dataset": "wikibio", "hf_id": source.repository, "revision": source.revision,
                       "split": source.split, "license": source.license,
                       "wiki_bio_test_idx": bio_index, "sentence_index": sentence_index},
        })
    return rows


# --- training splits of panel benchmarks, in the panel's own format ----------------

def as_training(row: Example, name: str, key: str) -> Example:
    """A panel-format row relabelled as training data from `name`'s training split."""
    source = SOURCES[name]
    identifier = make_id(name, key)
    return {**row, "id": identifier, "suite": name, "family": identifier,
            "source": {"dataset": name, "hf_id": source.repository, "revision": source.revision, "config": source.config,
                       "split": source.split, "license": source.license, "original_key": key}}


def convert_ragtruth_train(raw: dict[str, str]) -> Example:
    return as_training(panel.ragtruth_row(raw), "ragtruth_train", str(raw["id"]))


def convert_winogrande_train(index: int, raw: dict[str, str]) -> Example:
    return as_training(panel.winogrande_row(index, raw), "winogrande_train", str(index))


# --- driver ------------------------------------------------------------------------

@dataclass
class BuildResult:
    rows: list[Example] = field(default_factory=list)
    rows_read: int = 0
    dropped: Counter[str] = field(default_factory=Counter)


def build_snli() -> BuildResult:
    result = BuildResult()
    raw_rows = parquet_rows(SOURCES["snli"])
    result.rows_read = len(raw_rows)
    for index, raw in enumerate(raw_rows):
        row = convert_snli(index, raw)
        if row is None:
            result.dropped["label -1: annotators did not reach consensus"] += 1
            continue
        result.rows.append(row)
    return result


def _simple_build(name: str, rows: list[Example]) -> BuildResult:
    return BuildResult(rows=rows, rows_read=len(rows), dropped=Counter())


def build_commonsense_qa() -> BuildResult:
    raw_rows = parquet_rows(SOURCES["commonsense_qa"])
    return _simple_build("commonsense_qa", [convert_commonsense_qa(raw) for raw in raw_rows])


def build_openbookqa() -> BuildResult:
    raw_rows = parquet_rows(SOURCES["openbookqa"])
    return _simple_build("openbookqa", [convert_openbookqa(raw) for raw in raw_rows])


def build_arc_challenge() -> BuildResult:
    raw_rows = parquet_rows(SOURCES["arc_challenge"])
    return _simple_build("arc_challenge", [convert_arc_challenge(raw) for raw in raw_rows])


def build_arc_easy() -> BuildResult:
    raw_rows = parquet_rows(SOURCES["arc_easy"])
    return _simple_build("arc_easy", [convert_arc_easy(raw) for raw in raw_rows])


def build_social_iqa() -> BuildResult:
    raw_rows = parquet_rows(SOURCES["social_iqa"])
    return _simple_build("social_iqa", [convert_social_iqa(index, raw) for index, raw in enumerate(raw_rows)])


def build_cosmos_qa() -> BuildResult:
    raw_rows = parquet_rows(SOURCES["cosmos_qa"])
    return _simple_build("cosmos_qa", [convert_cosmos_qa(raw) for raw in raw_rows])


def build_quartz() -> BuildResult:
    raw_rows = parquet_rows(SOURCES["quartz"])
    return _simple_build("quartz", [convert_quartz(raw) for raw in raw_rows])


def build_qasc() -> BuildResult:
    raw_rows = parquet_rows(SOURCES["qasc"])
    return _simple_build("qasc", [convert_qasc(raw) for raw in raw_rows])


def build_truthful_qa() -> BuildResult:
    raw_rows = parquet_rows(SOURCES["truthful_qa"])
    return _simple_build("truthful_qa", [convert_truthful_qa(index, raw) for index, raw in enumerate(raw_rows)])


def build_twitter_financial() -> BuildResult:
    raw_rows = csv_rows(SOURCES["twitter_financial"])
    return _simple_build("twitter_financial", [convert_twitter_financial(index, raw) for index, raw in enumerate(raw_rows)])


def build_liar2() -> BuildResult:
    raw_rows = csv_rows(SOURCES["liar2"])
    return _simple_build("liar2", [convert_liar2(raw) for raw in raw_rows])


def build_halueval_qa() -> BuildResult:
    raw_rows = parquet_rows(SOURCES["halueval_qa"])
    return _simple_build("halueval_qa", [convert_halueval_qa(index, raw) for index, raw in enumerate(raw_rows)])


def build_halueval_dialogue() -> BuildResult:
    raw_rows = parquet_rows(SOURCES["halueval_dialogue"])
    return _simple_build("halueval_dialogue", [convert_halueval_dialogue(index, raw) for index, raw in enumerate(raw_rows)])


def build_halueval_summarization() -> BuildResult:
    raw_rows = parquet_rows(SOURCES["halueval_summarization"])
    return _simple_build("halueval_summarization",
                          [convert_halueval_summarization(index, raw) for index, raw in enumerate(raw_rows)])


def build_wikibio() -> BuildResult:
    raw_rows = parquet_rows(SOURCES["wikibio"])
    rows: list[Example] = []
    for raw in raw_rows:
        rows.extend(convert_wikibio(raw))
    return BuildResult(rows=rows, rows_read=len(raw_rows), dropped=Counter())


BUILDERS = {
    "snli": build_snli,
    "commonsense_qa": build_commonsense_qa,
    "openbookqa": build_openbookqa,
    "arc_challenge": build_arc_challenge,
    "arc_easy": build_arc_easy,
    "social_iqa": build_social_iqa,
    "cosmos_qa": build_cosmos_qa,
    "quartz": build_quartz,
    "qasc": build_qasc,
    "truthful_qa": build_truthful_qa,
    "twitter_financial": build_twitter_financial,
    "liar2": build_liar2,
    "halueval_qa": build_halueval_qa,
    "halueval_dialogue": build_halueval_dialogue,
    "halueval_summarization": build_halueval_summarization,
    "wikibio": build_wikibio,
    "ragtruth_train": lambda: _simple_build("ragtruth_train", [convert_ragtruth_train(raw)  # type: ignore[arg-type]
                                                              for raw in parquet_rows(SOURCES["ragtruth_train"])]),
    "winogrande_train": lambda: _simple_build("winogrande_train", [convert_winogrande_train(index, raw)  # type: ignore[arg-type]
                                                                  for index, raw in enumerate(parquet_rows(SOURCES["winogrande_train"]))]),
}


def sample(rows: list[Example], name: str, seed: int = SEED, cap: int = CAP) -> list[Example]:
    """Deterministically keep at most `cap` rows: sort by id, shuffle with a named seed, cut, re-sort."""
    ordered = sorted(rows, key=lambda row: row["id"])
    random.Random(f"{seed}-{name}").shuffle(ordered)
    return sorted(ordered[:cap], key=lambda row: row["id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/extra"))
    parser.add_argument("--only", nargs="+", choices=sorted(BUILDERS), help="Build only these datasets (into --out)")
    args = parser.parse_args()
    names = args.only or list(BUILDERS)

    manifest: dict[str, object] = {"seed": SEED, "cap_per_dataset": CAP, "datasets": {}, "skipped": SKIPPED}
    dataset_reports = cast(dict[str, object], manifest["datasets"])
    combined: list[Example] = []

    for name in names:
        result = BUILDERS[name]()
        validate(result.rows)
        chosen = sample(result.rows, name, cap=CAPS.get(name, CAP))
        validate(chosen)
        sha256 = write_rows(args.out / f"{name}.jsonl", chosen)
        source = SOURCES[name]
        dataset_reports[name] = {
            "hf_id": source.repository, "revision": source.revision,
            "requested_revision": source.requested_revision, "config": source.config, "split": source.split,
            "license": source.license, "note": source.note,
            "rows_read": result.rows_read, "rows_dropped": dict(result.dropped),
            "rows_converted": len(result.rows), "rows_written": len(chosen), "sha256": sha256,
        }
        combined.extend(chosen)
        print(f"{name}: read {result.rows_read}, dropped {sum(result.dropped.values())}, wrote {len(chosen)}",
              flush=True)

    validate(combined)
    combined_sha256 = write_rows(args.out / "train.jsonl", combined)
    manifest["combined_rows"] = len(combined)
    manifest["combined_sha256"] = combined_sha256
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
