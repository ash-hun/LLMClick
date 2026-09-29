from core.modules.data.materialize import convert_rows

PARAMS = {"suite": "s", "text_fields": ["premise", "hypothesis"], "label_field": "label",
          "label_map": {"0": "yes", "1": "no"}, "criteria": {"yes": "Yes.", "no": "No."}, "instructions": "Q?"}


def test_classification_converter_maps_labels_and_drops_unknown() -> None:
    raw = [{"premise": "a", "hypothesis": "b", "label": 0}, {"premise": "c", "hypothesis": "d", "label": 1},
           {"premise": "e", "hypothesis": "f", "label": -1}]
    rows = convert_rows(raw, "classification", PARAMS)
    assert [row["label"] for row in rows] == ["yes", "no"]
    assert rows[0]["state"] == "Premise: a\nHypothesis: b"
    assert rows[0]["question"]["criteria"] == {"yes": "Yes.", "no": "No."}


def test_boolean_converter() -> None:
    rows = convert_rows([{"t": "x", "ok": "yes"}, {"t": "y", "ok": 0}], "boolean",
                        {"suite": "b", "text_fields": ["t"], "label_field": "ok", "instructions": "Ok?"})
    assert [row["label"] for row in rows] == [True, False]
    assert rows[0]["question"]["type"] == "noul"
