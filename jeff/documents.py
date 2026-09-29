"""Long real documents with human labels, as decision rows: ContractNLI (non-disclosure agreements, CC BY 4.0),
ConditionalQA (UK government guidance pages; release under BSD-2, pages under the Open Government Licence) and CUAD
(commercial contracts annotated by lawyers for clause types, CC BY 4.0).

- ContractNLI: is a statement about the contract entailed, contradicted, or not mentioned by it?
- ConditionalQA: given a person's scenario and a guidance page, is the answer to their question yes, no, does it depend
  on a condition the scenario does not settle, or does the page not answer it? Free-text answers are left out.
- MAUD: merger-agreement excerpts with lawyer-chosen answers to deal-point questions ("type of consideration: all cash,
  all stock, mixed..."). Every answer a question kind has becomes an option; kinds whose answers are combinations
  (more than eight distinct answers) are left out, as are MAUD's shortened duplicate rows.
- CUAD: does an excerpt of a contract contain a clause of a given type? Excerpts are windows of the contract; the
  answer comes from the lawyers' marked spans. CUAD has no split of its own, so 15% of contracts (by a hash of the
  title) form the check set.

Half the rows (by a hash of the id) put the whole document in the state and the question in the instructions, as
JevBench's long-document items do; the other half use named fields. Rows longer than the model's input limit are left
out and counted, because the trainer never truncates. Each dataset's own dev split becomes a held-out check set."""

import argparse
import csv
import hashlib
import html
import json
import random
import re
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from jeff.data import validate, write_rows
from jeff.types import Example

CONTRACT_NLI = ("https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip",
                "data/docs-raw/contract-nli.zip")
CONDITIONAL_QA = ("https://github.com/haitian-sun/ConditionalQA", "77bd295952daf415548b3244db10880d3d55cfe0",
                  "data/docs-raw/ConditionalQA")
CUAD = ("theatticusproject/cuad", "a3c393f5d103fd0c516374e4fdff676c8176dcb1", "CUAD_v1/CUAD_v1.json")
# Metadata fields rather than clauses: every contract has them, so they teach little about reading clauses.
CUAD_SKIPPED = {"Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date", "Governing Law"}
CUAD_WINDOW = 12000
CUAD_CRITERIA = {"true": "Yes, the excerpt contains a clause of this type.", "false": "No, the excerpt contains no clause of this type."}
MAUD = ("theatticusproject/maud", "37d5c3b95d18dcd8404cc5ce3fd5069be062392f")
MAUD_PER_KIND = {"train": 40, "dev": 10}
MAUD_MAX_ANSWERS = 8
TOKENIZER = ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17")
MAX_TOKENS = 7000  # leaves room within the 8,192-token limit for the question, options and chat template
STATEMENTS_PER_CONTRACT = 6

CONTRACT_CRITERIA = {"entailed": "The contract states or clearly implies the statement.",
                     "contradicted": "The contract states the opposite of the statement.",
                     "not_mentioned": "The contract does not address the statement."}
CONTRACT_LABELS = {"Entailment": "entailed", "Contradiction": "contradicted", "NotMentioned": "not_mentioned"}
GUIDANCE_CRITERIA = {"yes": "Yes, for this person.", "no": "No, for this person.",
                     "depends": "It depends on a condition that the scenario does not settle.",
                     "not_answered": "The guidance does not answer this question."}


def fraction(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16) / 16 ** 12


def as_text(pieces: list[str]) -> str:
    """Guidance page fragments (HTML) as plain text, one block per line."""
    return "\n".join(html.unescape(re.sub(r"<[^>]+>", "", piece)).strip() for piece in pieces if piece.strip())


def row(identifier: str, suite: str, family: str, fields: dict[str, str], document_field: str, instructions: str,
        field_instructions: str, criteria: dict[str, str], label: str, source: dict) -> Example:
    """Named fields, or (for half the rows) the document as the state with the rest folded into the instructions."""
    if fraction(f"layout-{identifier}") < 0.5:
        state: object = fields
        question_text = field_instructions
    else:
        state = fields[document_field]
        others = "\n".join(f"{name.capitalize()}: {value}" for name, value in fields.items() if name != document_field)
        question_text = f"{others}\n{instructions}"
    return {"id": identifier, "suite": suite, "family": family, "state": state,  # type: ignore[typeddict-item]
            "question": {"type": "choice", "instructions": question_text, "criteria": dict(criteria)},
            "label": label, "target": label, "source": source}


def contract_rows(split: dict, name: str, seed: int) -> list[Example]:
    hypotheses = {key: value["hypothesis"] for key, value in split["labels"].items()}
    rows: list[Example] = []
    for document in split["documents"]:
        annotations = document["annotation_sets"][0]["annotations"]
        keys = sorted(annotations)
        rng = random.Random(f"{seed}-contract-{document['id']}")
        contradicted = [k for k in keys if annotations[k]["choice"] == "Contradiction"]
        others = [k for k in keys if k not in contradicted]
        chosen = contradicted + rng.sample(others, max(0, min(len(others), STATEMENTS_PER_CONTRACT - len(contradicted))))
        for key in sorted(chosen):
            identifier = f"contractnli-{name}-{document['id']}-{key}"
            rows.append(row(identifier, "contract_nli", f"contractnli-{document['id']}",
                            {"contract": document["text"], "statement": hypotheses[key]}, "contract",
                            "Using only the contract above, is the statement entailed, contradicted, or not mentioned?",
                            "Using only the contract, is the statement entailed, contradicted, or not mentioned?",
                            CONTRACT_CRITERIA, CONTRACT_LABELS[annotations[key]["choice"]],
                            {"dataset": "contract_nli", "split": name, "license": "CC-BY-4.0", "document": document["id"],
                             "hypothesis": key}))
    return rows


def guidance_label(item: dict) -> str | None:
    if item["not_answerable"]:
        return "not_answered"
    answers = {answer[0].strip().lower() for answer in item["answers"]}
    conditional = any(answer[1] for answer in item["answers"])
    if not answers <= {"yes", "no"}:
        return None  # a free-text answer
    return "depends" if conditional or answers == {"yes", "no"} else answers.pop()


def guidance_rows(items: list[dict], pages: dict[str, dict], name: str) -> list[Example]:
    rows: list[Example] = []
    for item in items:
        label = guidance_label(item)
        if label is None:
            continue
        page = pages[item["url"]]
        identifier = f"conditionalqa-{item['id']}"
        rows.append(row(identifier, "conditional_qa", f"conditionalqa-{item['url']}",
                        {"guidance": f"{page['title']}\n{as_text(page['contents'])}", "scenario": item["scenario"],
                         "question": item["question"]}, "guidance",
                        "Using only the guidance above, what is the answer to this person's question?",
                        "Using only the guidance, what is the answer to this person's question in their scenario?",
                        GUIDANCE_CRITERIA, label,
                        {"dataset": "conditional_qa", "split": name, "license": "BSD-2-Clause (release); OGL (gov.uk pages)",
                         "url": item["url"], "original_id": item["id"]}))
    return rows


def cuad_rows(data: list[dict], seed: int) -> dict[str, list[Example]]:
    """Up to three excerpts with a marked clause, two for clause types the contract lacks, and one for a type the contract
    has outside the excerpt, per contract."""
    rows: dict[str, list[Example]] = {"train": [], "dev": []}
    for contract in data:
        split = "dev" if fraction(f"cuad-split-{contract['title']}") < 0.15 else "train"
        context = contract["paragraphs"][0]["context"]
        spans: dict[str, list[tuple[int, int]]] = {}
        details: dict[str, str] = {}
        for qa in contract["paragraphs"][0]["qas"]:
            category = re.search(r'related to "(.+?)"', qa["question"]).group(1)  # type: ignore[union-attr]
            if category in CUAD_SKIPPED:
                continue
            details[category] = qa["question"].split("Details: ", 1)[1].strip()
            spans[category] = [(a["answer_start"], a["answer_start"] + len(a["text"])) for a in qa["answers"]]
        rng = random.Random(f"{seed}-cuad-{contract['title']}")
        present = sorted(c for c, found in spans.items() if found)
        absent = sorted(c for c, found in spans.items() if not found)
        size = min(CUAD_WINDOW, len(context))

        def window(start: int) -> tuple[int, int]:
            start = max(0, min(start, len(context) - size))
            return start, start + size

        cases: list[tuple[str, tuple[int, int], bool, str]] = []
        for category in rng.sample(present, min(3, len(present))):
            begin, end = rng.choice(spans[category])
            if end - begin < size:
                cases.append((category, window(begin - rng.randint(0, size - (end - begin))), True, "marked clause"))
        for category in rng.sample(absent, min(2, len(absent))):
            cases.append((category, window(rng.randrange(0, max(1, len(context) - size + 1))), False, "type absent from contract"))
        for category in rng.sample(present, len(present)):  # the first type with a window free of all its clauses
            low, high = window(rng.randrange(0, max(1, len(context) - size + 1)))
            if all(end <= low or begin >= high for begin, end in spans[category]):
                cases.append((category, (low, high), False, "type elsewhere in contract"))
                break
        for category, (low, high), label, kind in cases:
            if label and not any(low <= begin and end <= high for begin, end in spans[category]):
                continue  # the clause did not fit inside the window
            excerpt = re.sub(r"[ \t]+", " ", context[low:high]).strip()
            identifier = (f"cuad-{hashlib.sha256(contract['title'].encode()).hexdigest()[:12]}-"
                          f"{re.sub(r'[^a-z]+', '_', category.lower())}-{'in' if label else kind.split()[1]}")
            if fraction(f"layout-{identifier}") < 0.5:
                state: object = {"contract_excerpt": excerpt, "clause_type": f"{category}: {details[category]}"}
                instructions = "Does the contract excerpt contain a clause of the given type?"
            else:
                state = excerpt
                instructions = f"Does this contract excerpt contain a clause about {category}? {details[category]}"
            rows[split].append({"id": identifier, "suite": "cuad", "family": f"cuad-{contract['title']}", "state": state,  # type: ignore[typeddict-item]
                                "question": {"type": "noul", "instructions": instructions, "criteria": dict(CUAD_CRITERIA)},
                                "label": label, "target": label,
                                "source": {"dataset": "cuad", "license": "CC-BY-4.0", "contract": contract["title"],
                                           "clause_type": category, "kind": kind, "window": [low, high]}})
    return rows


def maud_rows(rows: dict[str, list[dict]], seed: int) -> dict[str, list[Example]]:
    """Multiple-choice rows per MAUD question kind; the options are all answers the kind has in the training split."""
    kinds: dict[tuple[str, str], list[str]] = {}
    for item in rows["train"]:
        answers = kinds.setdefault((item["question"], item["subquestion"]), [])
        if item["answer"] not in answers:
            answers.append(item["answer"])
    usable = {kind: sorted(answers) for kind, answers in kinds.items() if 2 <= len(answers) <= MAUD_MAX_ANSWERS}
    result: dict[str, list[Example]] = {}
    for split, cap in MAUD_PER_KIND.items():
        chosen: list[Example] = []
        for kind, answers in sorted(usable.items()):
            matching = [i for i in rows[split] if (i["question"], i["subquestion"]) == kind and i["data_type"] != "abridged"
                        and i["answer"] in answers]
            items = list({(i["data_type"], i["id"], i["text"]): i for i in matching}.values())  # MAUD repeats some rows exactly
            question, sub = kind
            topic = re.sub(r"[-:]?\s*Answer$", "", question).strip() + ("" if sub == "<NONE>" else f" ({sub})")
            keys = {answer: f"option_{index + 1}" for index, answer in enumerate(answers)}
            for item in random.Random(f"{seed}-maud-{split}-{question}-{sub}").sample(items, min(cap, len(items))):
                identifier = (f"maud-{split}-{item['data_type']}-{item['id']}-"
                              f"{hashlib.sha256((topic + item['text']).encode()).hexdigest()[:10]}")
                chosen.append({"id": identifier, "suite": "maud", "family": f"maud-{item['contract_name']}", "state": item["text"],
                               "question": {"type": "choice",
                                            "instructions": f"This is an excerpt from a merger agreement. Which option describes its {topic}?",
                                            "criteria": {keys[a]: a for a in answers}},
                               "label": keys[item["answer"]], "target": keys[item["answer"]],
                               "source": {"dataset": "maud", "license": "CC-BY-4.0", "split": split, "contract": item["contract_name"],
                                          "question": question, "subquestion": sub, "original_id": item["id"]}})
        result[split] = chosen
    return result


def fits(rows: list[Example], tokenizer) -> tuple[list[Example], int]:  # type: ignore[no-untyped-def]
    kept = [r for r in rows if len(tokenizer.encode(json.dumps(r["state"], ensure_ascii=False) + r["question"]["instructions"]))
            <= MAX_TOKENS]
    return kept, len(rows) - len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/documents"))
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    checkout = Path(CONDITIONAL_QA[2])
    commit = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if commit != CONDITIONAL_QA[1]:
        raise ValueError(f"{checkout} is at {commit}, not the pinned {CONDITIONAL_QA[1]}")
    archive = Path(CONTRACT_NLI[1])
    with zipfile.ZipFile(archive) as bundle:
        contracts = {name: json.loads(bundle.read(f"contract-nli/{name}.json")) for name in ("train", "dev")}
    pages = {page["url"]: page for page in json.loads((checkout / "v1_0" / "documents.json").read_text())}
    guidance = {name: json.loads((checkout / "v1_0" / f"{name}.json").read_text()) for name in ("train", "dev")}
    cuad_path = Path(hf_hub_download(CUAD[0], CUAD[2], repo_type="dataset", revision=CUAD[1]))
    cuad = cuad_rows(json.loads(cuad_path.read_text())["data"], args.seed)
    maud_raw = {split: list(csv.DictReader(Path(hf_hub_download(MAUD[0], f"MAUD_v1/MAUD_{split}.csv", repo_type="dataset",
                                                                 revision=MAUD[1])).open(encoding="utf-8")))
                for split in ("train", "dev")}
    maud = maud_rows(maud_raw, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    manifest: dict[str, object] = {"contract_nli": CONTRACT_NLI[0], "contract_nli_zip_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                                   "conditional_qa": CONDITIONAL_QA[0], "conditional_qa_commit": commit, "cuad": f"{CUAD[0]}@{CUAD[1]}", "maud": f"{MAUD[0]}@{MAUD[1]}",
                                   "max_tokens": MAX_TOKENS}
    for name, output in (("train", "train"), ("dev", "check")):
        rows = contract_rows(contracts[name], name, args.seed) + guidance_rows(guidance[name], pages, name) + cuad[name] + maud[name]
        rows, too_long = fits(rows, tokenizer)
        validate(rows)
        manifest[output] = {"rows": len(rows), "left_out_too_long": too_long, "sha256": write_rows(args.out / f"{output}.jsonl", rows),
                            "labels": {suite: dict(Counter(str(r["label"]) for r in rows if r["suite"] == suite))
                                       for suite in ("contract_nli", "conditional_qa", "cuad", "maud")}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
