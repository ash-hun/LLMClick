from jeff import legalbench
from jeff.data import validate

README = """# hearsay

### Classify if a particular piece of evidence qualifies as hearsay.
---

**License**: [CC By 4.0](https://creativecommons.org/licenses/by/4.0/)

## Task description
Hearsay is an out-of-court statement introduced to prove the truth of the matter asserted.

## Other
"""
TABLE = "answer\tindex\ttext\n" + "\n".join(f"{'Yes' if i % 2 else 'No'}\t{i}\tEvidence number {i}." for i in range(100)) + "\n"


def test_licences_are_read_and_non_commercial_ones_refused() -> None:
    assert legalbench.licence(README) is not None
    assert legalbench.licence(README.replace("CC By 4.0", "CC BY-NC-SA 4.0")) is None
    assert legalbench.licence("**License**: Creative Commons Attribution-NonCommercial License") is None
    assert legalbench.licence("**License**: MIT") == "MIT"


def test_a_task_becomes_choice_rows_with_its_description_as_background() -> None:
    split = legalbench.task_rows("hearsay", README, TABLE)
    assert split is not None and len(split["train"]) + len(split["check"]) == 100 and 5 <= len(split["check"]) <= 30
    row = split["train"][0]
    assert row["question"]["criteria"] == {"no": "No", "yes": "Yes"}
    assert row["question"]["instructions"].startswith("Hearsay is an out-of-court statement")
    assert row["question"]["instructions"].endswith("Task: Classify if a particular piece of evidence qualifies as hearsay.")
    assert row["state"].startswith("Evidence number")
    validate(split["train"] + split["check"])
    assert legalbench.task_rows("cuad_audit_rights", README, TABLE) is None  # built from CUAD: used directly elsewhere
    open_ended = "answer\ttext\n" + "\n".join(f"answer {i}\tq{i}" for i in range(20)) + "\n"
    assert legalbench.task_rows("rule_qa", README, open_ended) is None  # 20 distinct answers: not a decision


def test_hint_columns_are_dropped_and_multiple_choice_uses_the_choice_texts() -> None:
    table = "answer\tindex\ttext\tslice\tparties_are_diverse\n" + "".join(f"{'Yes' if i % 2 else 'No'}\t{i}\tFacts {i}.\thint\tTrue\n" for i in range(40))
    split = legalbench.task_rows("diversity_1", README, table)
    assert split is not None and all(isinstance(r["state"], str) and r["state"].startswith("Facts ") for r in split["train"])
    assert not any("hint" in str(r["state"]) or "True" in str(r["state"]) for r in split["train"] + split["check"])
    scalr = "index\tquestion\tchoice_0\tchoice_1\tanswer\n" + "".join(f"{i}\tWhich holding?\tholding one\tholding two\t{i % 2}\n" for i in range(40))
    rows = legalbench.task_rows("scalr", README, scalr)
    assert rows is not None
    row = rows["train"][0]
    assert row["question"]["criteria"] == {"choice_0": "holding one", "choice_1": "holding two"} and row["state"] == "Which holding?"
    assert row["question"]["criteria"][row["label"]] == ("holding two" if row["id"].endswith(("1", "3", "5", "7", "9")) else "holding one")
    listed = "answer\ttext\n" + "".join(f"['a', 'b']\tx{i}\n['a']\ty{i}\n" for i in range(10))
    assert legalbench.task_rows("successor_liability", README, listed) is None
