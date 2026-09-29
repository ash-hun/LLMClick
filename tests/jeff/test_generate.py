import asyncio
import json
from pathlib import Path

import pytest

from jeff import generate
from jeff.families import BY_NAME, batches, make_slots
from jeff.teacher import MalformedOutput


# Every part any family requires, so fake texts pass the structure check whatever the family.
PARTS = (" Premise: p Hypothesis: h Passage: p Question: q Source: s Instruction: i Response: r Response A: a Response B: b"
         " _ Option 1: x Option 2: y A: a B: b C: c D: d Rules: r Statement: s Claim: c Source A: a Source B: b"
         " Statement 1: s Statement 2: t Document: d Summary: s Policy: p Request: r\nCompare: x vs y\nEvents: e / f")


# Families whose checks need parts at a line start or at the very end, keyed by the family name inside the slot id.
TAILS = {"translation_error": "\nSource: s\nTranslation: t", "disambiguation": "\nSentence: s\nA: a\nB: b\nC: Ambiguous",
         "causal_judgement": "\nDid x cause y?"}


def fake_text(identifier: str, prefix: str = "text") -> str:
    tail = next((text for family, text in TAILS.items() if f"-{family}-" in identifier), "")
    return f"{prefix} for {identifier}{PARTS}{tail}"


def grounded_text(slot: dict) -> str:
    right = slot["required_label"]
    answers = {right: slot["correct_answer"], "B" if right == "A" else "A": slot["wrong_answer"]}
    return (f"text for {slot['id']}\nQuestion: {slot['question']}\nResponse A: reasoning.\nFinal answer: {answers['A']}\n"
            f"Response B: reasoning.\nFinal answer: {answers['B']}")


class FakeTeacher:
    """Writes 'text for <id>'; repairs to 'fixed text for <id>'. The verifier labels original texts with verify_label and
    repaired texts with fixed_label (defaults to verify_label)."""

    def __init__(self, verify_label, write_error: Exception | None = None, drop: set[str] = frozenset(),
                 fixed_label=None, review=lambda text: "pass") -> None:
        self.verify_label, self.write_error, self.drop = verify_label, write_error, drop
        self.fixed_label = fixed_label or verify_label
        self.review = review
        self.calls = 0
        self.requests: list[dict] = []

    async def complete(self, *, system, user, schema, temperature, max_tokens):
        self.calls += 1
        request = json.loads(user)
        self.requests.append(request)
        if "slots" in request:
            if self.write_error:
                raise self.write_error
            return {"examples": [{"id": s["id"], "state": grounded_text(s) if "correct_answer" in s else fake_text(s["id"])}
                                 for s in request["slots"] if s["id"] not in self.drop]}
        if "examples" not in request:  # semantic review of one row, label included
            return {"reasoning": f"checked {request['text']}", "overall_verdict": self.review(request["text"])}
        if any("required_label" in e for e in request["examples"]):
            return {"examples": [{"id": e["id"], "state": fake_text(e["id"], "fixed text")} for e in request["examples"]]}
        # The verifier never sees the intended label or the previous reviewer's opinion.
        assert "required_label" not in json.dumps(request) and "reviewer" not in json.dumps(request)
        return {"examples": [{"id": e["id"], "unambiguous": True, "reason": "because",
                              "label": (self.fixed_label if e["text"].startswith("fixed") else self.verify_label)(e["id"])}
                             for e in request["examples"]]}


GROUNDED_ITEMS = [{"id": f"gsm8k-{i}", "question": f"What is {i} + {i}?", "correct_answer": str(2 * i), "wrong_answer": str(2 * i + 1)}
                  for i in range(20)]


def dressed(count: int):
    from jeff import grounded
    from jeff.materials import dress
    from tests.jeff.test_materials import pools

    return grounded.dress(dress(make_slots(count, seed=1), pools(), seed=1), GROUNDED_ITEMS, seed=1)


def first_batch(family: str = "support_routing"):
    return next(b for b in batches(make_slots(200, seed=1)) if b[0]["family"] == family)


def test_agreeing_rows_are_kept_as_examples() -> None:
    batch = first_batch()
    labels = {s["id"]: s["label"] for s in batch}
    outcomes = asyncio.run(generate.process_batch(FakeTeacher(lambda i: labels[i]), batch))
    assert [o["reason"] for o in outcomes] == ["kept"] * len(batch)
    row = outcomes[0]["row"]
    assert row["id"] == batch[0]["id"] and row["suite"] == "synthetic"
    assert row["question"] == BY_NAME["support_routing"].question()
    assert row["label"] == row["target"] == batch[0]["label"]
    assert row["family"].startswith("syn-support_routing-")


def test_noul_labels_become_booleans() -> None:
    batch = first_batch("sarcasm")
    labels = {s["id"]: s["label"] for s in batch}
    outcomes = asyncio.run(generate.process_batch(FakeTeacher(lambda i: labels[i]), batch))
    for outcome, slot in zip(outcomes, batch):
        assert outcome["row"]["label"] is (slot["label"] == "true")


def test_disagreement_is_rejected() -> None:
    batch = first_batch()
    outcomes = asyncio.run(generate.process_batch(FakeTeacher(lambda i: "other"), batch))
    reasons = {o["reason"] for o in outcomes}
    assert reasons <= {"kept", "label_mismatch_after_fix"} and "label_mismatch_after_fix" in reasons
    assert all(o["row"] is None for o in outcomes if not o["kept"])


def test_malformed_and_missing_become_rejections() -> None:
    batch = first_batch()
    malformed = asyncio.run(generate.process_batch(FakeTeacher(lambda i: "x", write_error=MalformedOutput("cut")), batch))
    assert {o["reason"] for o in malformed} == {"malformed_write"}
    labels = {s["id"]: s["label"] for s in batch}
    missing = asyncio.run(generate.process_batch(FakeTeacher(lambda i: labels[i], drop={batch[0]["id"]}), batch))
    assert missing[0]["reason"] == "missing_from_write"
    assert all(o["reason"] == "kept" for o in missing[1:])


def test_resume_skips_finished_batches(tmp_path: Path) -> None:
    slots = dressed(40)
    labels = {s["id"]: s["label"] for s in slots}
    teacher = FakeTeacher(lambda i: labels[i])
    asyncio.run(generate.run_slots(teacher, slots, tmp_path, concurrency=4))
    first_calls = teacher.calls
    asyncio.run(generate.run_slots(teacher, slots, tmp_path, concurrency=4))
    assert teacher.calls == first_calls
    outcomes = generate.load_outcomes(tmp_path / "outcomes.jsonl")
    assert sorted(o["slot"] for o in outcomes) == sorted(labels)


def test_torn_outcome_line_raises(tmp_path: Path) -> None:
    path = tmp_path / "outcomes.jsonl"
    path.write_text('{"slot": "a"}\n{"slot": ')
    with pytest.raises(ValueError, match="line 2"):
        generate.load_outcomes(path)


def test_finalize_writes_rows_and_report(tmp_path: Path) -> None:
    slots = dressed(40)
    labels = {s["id"]: s["label"] for s in slots}
    asyncio.run(generate.run_slots(FakeTeacher(lambda i: labels[i]), slots, tmp_path, concurrency=4))
    report = generate.finalize(tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "synthetic.jsonl").read_text().splitlines()]
    assert len(rows) == 40 == report["kept"]
    assert report["by_family"]["support_routing"]["kept"] == sum(s["family"] == "support_routing" for s in slots)
    assert sum(family["kept"] for family in report["by_family"].values()) == 40


def test_every_family_has_its_own_writer_prompt() -> None:
    from jeff.families import FAMILIES
    from jeff.writer_prompts import WRITER

    teacher_written = {family.name for family in FAMILIES if family.name not in generate.PROGRAMMATIC}
    assert teacher_written <= set(WRITER)
    for family in FAMILIES:
        if family.name not in WRITER:
            continue
        system = generate.write_system(family)
        assert WRITER[family.name] in system
        assert "required_label" in system


def test_rejected_rows_are_repaired_and_verified_again() -> None:
    batch = first_batch()
    labels = {s["id"]: s["label"] for s in batch}
    teacher = FakeTeacher(lambda i: "other", fixed_label=lambda i: labels[i])
    outcomes = asyncio.run(generate.process_batch(teacher, batch))
    for slot, outcome in zip(batch, outcomes):
        if slot["label"] == "other":
            assert outcome["reason"] == "kept" and outcome["row"]["source"]["repaired"] is False
        else:
            assert outcome["reason"] == "kept_after_fix"
            assert outcome["row"]["state"].startswith(f"fixed text for {slot['id']} ")
            assert outcome["row"]["source"]["repaired"] is True
    fix_request = next(r for r in teacher.requests if "examples" in r and "required_label" in r["examples"][0])
    assert all("'other'" in e["problem"] and "because" in e["problem"] for e in fix_request["examples"])


def test_leaked_scratch_work_is_rejected_before_verification() -> None:
    assert generate.leaked("Mara has the atlas. No, wait, let's re-read the clues.")
    assert not generate.leaked("Response A: the answer is 12 because 3 x 4 = 12.")
    assert generate.leaked("Wait, the slot requires Label D, so I will change the clue.")
    assert generate.leaked("The last clue settles it. So B is true.")
    assert generate.leaked("Response: the policy covers it. Label is 'false'.")
    assert not generate.leaked("The time slot for the meeting moved, and the label on the box said fragile.")
    assert not generate.leaked("So the committee agreed the budget was correct.")
    batch = first_batch()
    labels = {s["id"]: s["label"] for s in batch}
    teacher = FakeTeacher(lambda i: labels[i])
    original = teacher.complete

    async def leaky(**kw):
        reply = await original(**kw)
        if "slots" in json.loads(kw["user"]):
            reply["examples"][0]["state"] = "The required label is billing."
        return reply

    teacher.complete = leaky
    outcomes = asyncio.run(generate.process_batch(teacher, batch))
    assert outcomes[0]["reason"] == "leaked_reasoning"
    assert all(o["reason"] == "kept" for o in outcomes[1:])


def test_object_tracking_is_built_in_code_without_the_teacher() -> None:
    from jeff.materials import dress
    from tests.jeff.test_materials import pools

    slots = dress(make_slots(200, seed=1), pools(), seed=1)
    batch = next(b for b in batches(slots) if b[0]["family"] == "object_tracking")
    teacher = FakeTeacher(lambda i: pytest.fail("the teacher must not be called"))
    outcomes = asyncio.run(generate.process_batch(teacher, batch))
    assert teacher.calls == 0
    assert all(o["kept"] and o["row"]["source"]["teacher"] == "program" for o in outcomes)


def test_semantic_review_failure_is_repaired_with_the_reviewers_reasoning() -> None:
    batch = first_batch()
    labels = {s["id"]: s["label"] for s in batch}
    teacher = FakeTeacher(lambda i: labels[i], review=lambda text: "pass" if text.startswith("fixed") else "fail")
    outcomes = asyncio.run(generate.process_batch(teacher, batch))
    assert {o["reason"] for o in outcomes} == {"kept_after_fix"}
    fix_request = next(r for r in teacher.requests if "examples" in r and "required_label" in r["examples"][0])
    assert all(e["problem"].startswith(f"checked text for {e['id']} ") for e in fix_request["examples"])
    reviewed = [r for r in teacher.requests if "examples" not in r and "slots" not in r]
    assert all(r["label"] in labels.values() and "label_meaning" in r for r in reviewed)


def test_row_failing_review_twice_is_discarded_with_the_reason() -> None:
    batch = first_batch()
    labels = {s["id"]: s["label"] for s in batch}
    outcomes = asyncio.run(generate.process_batch(FakeTeacher(lambda i: labels[i], review=lambda text: "fail"), batch))
    assert {o["reason"] for o in outcomes} == {"review_failed_after_fix"}
    assert all(o["row"] is None and o["detail"].startswith("checked fixed text") for o in outcomes)


def test_structure_check_names_the_missing_parts() -> None:
    assert generate.missing_parts("pronoun_resolution", "Ana thanked Ben because _ helped.\nOption 1: Ana\nOption 2: Ben") == []
    assert generate.missing_parts("pronoun_resolution", "Ana thanked Ben because he helped.\nOption 1: Ana") == \
        ["a blank written as _", "an 'Option 2:' line"]
    assert generate.missing_parts("numeric_comparison", "Facts.\nCompare: the rent vs the fee") == []
    assert generate.missing_parts("sarcasm", "anything") == []


def test_text_missing_structure_is_repaired_without_verification_first() -> None:
    batch = first_batch("paraphrase")
    labels = {s["id"]: s["label"] for s in batch}
    teacher = FakeTeacher(lambda i: labels[i])
    original = teacher.complete

    async def bare_first_draft(**kw):
        reply = await original(**kw)
        if "slots" in json.loads(kw["user"]):
            for e in reply["examples"]:
                e["state"] = "Just one statement here."
        return reply

    teacher.complete = bare_first_draft
    outcomes = asyncio.run(generate.process_batch(teacher, batch))
    assert {o["reason"] for o in outcomes} == {"kept_after_fix"}
    fix_request = next(r for r in teacher.requests if "examples" in r and "required_label" in r["examples"][0])
    assert all("Statement 1:" in e["problem"] and "Statement 2:" in e["problem"] for e in fix_request["examples"])
    verified = [r for r in teacher.requests if "examples" in r and "required_label" not in r["examples"][0]]
    assert all("Just one statement" not in json.dumps(r) for r in verified)


def test_batches_are_interleaved_across_families() -> None:
    from jeff.families import FAMILIES

    n = len(FAMILIES)
    order = [batch[0]["family"] for batch in generate.interleaved(batches(make_slots(n * 24, seed=1)))]
    assert len(set(order[:n])) == n  # the first round of batches covers every family once
    assert order[:n] == order[n:2 * n]


def test_grounded_rows_skip_the_blind_check_and_keep_the_proven_label() -> None:
    batch = next(b for b in batches(dressed(200)) if b[0]["family"] == "grounded_pairwise")
    teacher = FakeTeacher(lambda i: pytest.fail("grounded rows must not be relabelled blind"))
    outcomes = asyncio.run(generate.process_batch(teacher, batch))
    assert all(o["reason"] == "kept" and o["row"]["label"] == s["label"] for o, s in zip(outcomes, batch))


def test_grounded_wrong_final_answer_goes_to_repair_with_the_reason() -> None:
    batch = next(b for b in batches(dressed(200)) if b[0]["family"] == "grounded_pairwise")
    slot = batch[0]
    text = grounded_text({**slot, "required_label": slot["label"]}).replace(f"Final answer: {slot['correct_answer']}", "Final answer: 999999")
    assert f"must end with 'Final answer: {slot['correct_answer']}'" in generate.row_problems(slot, text)


def test_outcomes_with_unicode_line_separators_load(tmp_path: Path) -> None:
    path = tmp_path / "outcomes.jsonl"
    path.write_text(json.dumps({"slot": "a", "text": "first\u2028second"}, ensure_ascii=False) + "\n")
    assert generate.load_outcomes(path) == [{"slot": "a", "text": "first\u2028second"}]
