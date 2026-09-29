from jeff import documents
from jeff.data import validate


def test_guidance_answers_map_to_four_decisions() -> None:
    assert documents.guidance_label({"not_answerable": True, "answers": []}) == "not_answered"
    assert documents.guidance_label({"not_answerable": False, "answers": [["yes", []]]}) == "yes"
    assert documents.guidance_label({"not_answerable": False, "answers": [["No", []]]}) == "no"
    assert documents.guidance_label({"not_answerable": False, "answers": [["yes", ["if you are over 18"]]]}) == "depends"
    assert documents.guidance_label({"not_answerable": False, "answers": [["yes", []], ["no", ["x"]]]}) == "depends"
    assert documents.guidance_label({"not_answerable": False, "answers": [["£100", []]]}) is None  # free text: left out


def test_contract_rows_keep_every_contradiction_and_cap_the_rest() -> None:
    labels = {f"nda-{i}": {"hypothesis": f"Statement {i}."} for i in range(10)}
    choices = ["Contradiction", "Contradiction"] + ["Entailment"] * 4 + ["NotMentioned"] * 4
    split = {"labels": labels, "documents": [{"id": 3, "text": "The contract text.",
             "annotation_sets": [{"annotations": {f"nda-{i}": {"choice": c} for i, c in enumerate(choices)}}]}]}
    rows = documents.contract_rows(split, "train", seed=1)
    assert len(rows) == documents.STATEMENTS_PER_CONTRACT
    assert sum(r["label"] == "contradicted" for r in rows) == 2
    for r in rows:
        if isinstance(r["state"], dict):
            assert r["state"]["contract"] == "The contract text." and r["state"]["statement"].startswith("Statement")
        else:
            assert r["state"] == "The contract text." and "Statement: Statement" in r["question"]["instructions"]
    validate(rows)


def test_guidance_pages_become_plain_text() -> None:
    assert documents.as_text(["<h1>Overview</h1>", "<p>You can &amp; must apply.</p>", " "]) == "Overview\nYou can & must apply."


def test_cuad_labels_follow_the_marked_spans() -> None:
    text = "x" * 30000 + " The Licensee shall not compete. " + "y" * 30000
    start = text.index("The Licensee")
    contract = {"title": "Acme Supply Agreement", "paragraphs": [{"context": text, "qas": [
        {"question": 'Highlight the parts (if any) of this contract related to "Non-Compete" that should be reviewed by a lawyer. Details: Is there a non-compete?',
         "answers": [{"text": "The Licensee shall not compete.", "answer_start": start}]},
        {"question": 'Highlight the parts (if any) of this contract related to "Insurance" that should be reviewed by a lawyer. Details: Is insurance required?',
         "answers": []},
        {"question": 'Highlight the parts (if any) of this contract related to "Parties" that should be reviewed by a lawyer. Details: Who?',
         "answers": [{"text": "x", "answer_start": 0}]}]}]}
    rows = [r for split in documents.cuad_rows([contract], seed=1).values() for r in split]
    for r in rows:
        low, high = r["source"]["window"]
        inside = low <= start and start + 31 <= high
        assert r["label"] is (r["source"]["clause_type"] == "Non-Compete" and inside)
        assert r["source"]["clause_type"] != "Parties"  # metadata fields are skipped
    assert any(r["label"] is True for r in rows) and any(r["label"] is False for r in rows)
    validate(rows)


def test_maud_options_are_the_answers_each_question_kind_has() -> None:
    def item(i: int, answer: str, kind: str = "Type of Consideration-Answer", data_type: str = "main") -> dict:
        return {"id": str(i), "contract_name": f"contract_{i}", "text": f"Excerpt {i}.", "answer": answer, "question": kind,
                "subquestion": "<NONE>", "data_type": data_type}
    train = [item(0, "All Cash"), item(1, "All Stock"), item(2, "All Cash", data_type="abridged")]
    train += [item(10 + i, f"Combination {i}", kind="Types of R&Ws") for i in range(9)]  # too many answers: left out
    rows = documents.maud_rows({"train": train, "dev": [item(5, "All Stock")]}, seed=1)
    assert len(rows["train"]) == 2 and len(rows["dev"]) == 1  # the abridged duplicate and the combination kind are gone
    for r in rows["train"] + rows["dev"]:
        assert set(r["question"]["criteria"].values()) == {"All Cash", "All Stock"}
        assert r["question"]["instructions"].endswith("describes its Type of Consideration?")
    assert rows["dev"][0]["question"]["criteria"][rows["dev"][0]["label"]] == "All Stock"
    validate(rows["train"])
