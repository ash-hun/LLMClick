import pytest

from jeff import panel


def test_bbh_lettered_options() -> None:
    raw = {"input": "Which statement is sarcastic?\nOptions:\n(A) He is kind\n(B) He is terrible, obviously", "target": "(B)"}
    row = panel.bbh_row("snarks", 3, raw)
    assert row["suite"] == "BBH"
    assert row["state"] == "Which statement is sarcastic?"
    assert row["question"] == {"type": "choice", "instructions": panel.BBH_INSTRUCTIONS,
                               "criteria": {"A": "He is kind", "B": "He is terrible, obviously"}}
    assert row["label"] == row["target"] == "B"
    assert row["source"]["task"] == "snarks"


def test_bbh_dash_options() -> None:
    raw = {"input": "Is the argument valid?\nOptions:\n- valid\n- invalid", "target": "invalid"}
    row = panel.bbh_row("formal_fallacies", 0, raw)
    assert row["question"]["criteria"] == {"valid": None, "invalid": None}
    assert row["label"] == "invalid"


def test_bbh_dash_options_with_trailing_spaces() -> None:
    raw = {"input": "Is the argument valid?\nOptions:\n- valid \n- invalid", "target": "valid"}
    row = panel.bbh_row("formal_fallacies", 3, raw)
    assert row["question"]["criteria"] == {"valid": None, "invalid": None}
    assert row["label"] == "valid"


def test_bbh_boolean_expressions_has_true_false() -> None:
    row = panel.bbh_row("boolean_expressions", 0, {"input": "not ( True ) and ( True ) is", "target": "False"})
    assert row["question"]["criteria"] == {"True": None, "False": None}
    assert row["label"] == "False"


def test_bbh_target_not_in_options_raises() -> None:
    raw = {"input": "Q\nOptions:\n(A) x\n(B) y", "target": "(C)"}
    with pytest.raises(ValueError, match="snarks.*7"):
        panel.bbh_row("snarks", 7, raw)


def test_fpb_line() -> None:
    row = panel.fpb_row(5, "Sales doubled to EUR131m .@positive")
    assert row["suite"] == "Financial PhraseBank"
    assert row["state"] == "Sales doubled to EUR131m ."
    assert list(row["question"]["criteria"]) == ["positive", "negative", "neutral"]
    assert row["label"] == "positive"


def test_judgebench_labels() -> None:
    raw = {"pair_id": "p1", "question": "Q?", "response_A": "a", "response_B": "b", "label": "B>A", "source": "mmlu-pro-law"}
    row = panel.judgebench_row(raw)
    assert row["state"] == {"question": "Q?", "response_A": "a", "response_B": "b"}
    assert row["label"] == "B"
    with pytest.raises(ValueError, match="A=B"):
        panel.judgebench_row({**raw, "label": "A=B"})


def test_ragtruth_noul() -> None:
    raw = {"id": "24", "query": "Summarize", "context": "ctx", "output": "out", "task_type": "Summary",
           "hallucination_labels": '[{"start": 1, "end": 4, "label_type": "Evident Conflict"}]'}
    row = panel.ragtruth_row(raw)
    assert row["question"]["type"] == "noul"
    assert row["label"] is True and row["target"] is True
    assert panel.ragtruth_row({**raw, "hallucination_labels": "[]"})["label"] is False


def test_winogrande() -> None:
    row = panel.winogrande_row(2, {"sentence": "Sarah beat Maria so _ won.", "option1": "Sarah", "option2": "Maria", "answer": "1"})
    assert row["question"]["criteria"] == {"1": "Sarah", "2": "Maria"}
    assert row["label"] == "1"


def test_long_rows_excluded_before_sampling() -> None:
    rows = [panel.fpb_row(i, f"s{i} .@neutral") for i in range(10)]
    chosen, dropped = panel.sample(rows, 5, seed=1, fits=lambda row: row["id"] not in {rows[0]["id"], rows[1]["id"]})
    assert dropped == 2
    assert len(chosen) == 5
    assert all(row["id"] not in {rows[0]["id"], rows[1]["id"]} for row in chosen)


def test_sample_raises_when_too_few_fit() -> None:
    rows = [panel.fpb_row(i, f"s{i} .@neutral") for i in range(3)]
    with pytest.raises(ValueError, match="Financial PhraseBank"):
        panel.sample(rows, 5, seed=1, fits=lambda row: True)
