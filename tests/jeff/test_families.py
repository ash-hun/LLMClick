from collections import Counter

from jeff.families import BY_NAME, CONDITIONS, FAMILIES, POSITIONAL, batches, make_slots


def test_unique_families_with_valid_labels() -> None:
    assert len(FAMILIES) == 38 == len(BY_NAME)
    for family in FAMILIES:
        assert family.labels, family.name
        if family.kind == "noul":
            assert family.labels == ["true", "false"]
            assert set(family.criteria) == {"true", "false"}
            assert family.question()["type"] == "noul"
        else:
            assert family.question()["criteria"] == family.criteria


def test_slots_are_deterministic_and_unique() -> None:
    first, second = make_slots(400, seed=7), make_slots(400, seed=7)
    assert first == second
    assert len({slot["id"] for slot in first}) == 400
    assert make_slots(400, seed=8) != first


def test_labels_balanced_outside_skewed_condition() -> None:
    slots = make_slots(20 * 200, seed=1)
    for family in FAMILIES:
        if family.name in POSITIONAL:
            continue  # positional labels cycle over every slot; test_positional_labels_balanced_over_all_slots checks them
        counts = Counter(s["label"] for s in slots if s["family"] == family.name and s["condition"] != "skewed")
        assert max(counts.values()) - min(counts.values()) <= 1, (family.name, counts)
        skewed = [s["label"] for s in slots if s["family"] == family.name and s["condition"] == "skewed"]
        if family.name in POSITIONAL:
            counts = Counter(skewed)
            assert max(counts.values()) - min(counts.values()) <= 1, (family.name, counts)
        else:
            assert set(skewed) == {family.labels[0]}


def test_positional_labels_balanced_over_all_slots() -> None:
    """Counted over every slot: the correct answer must not favour any position (it once sat at A 60% of the time)."""
    slots = make_slots(25 * 200, seed=1)
    for name in POSITIONAL:
        counts = Counter(s["label"] for s in slots if s["family"] == name)
        assert max(counts.values()) - min(counts.values()) <= 1, (name, counts)


def test_every_condition_used() -> None:
    assert {s["condition"] for s in make_slots(200, seed=1)} == set(CONDITIONS)


def test_batches_single_family_and_bounded() -> None:
    for batch in batches(make_slots(1000, seed=3)):
        assert 1 <= len(batch) <= 8
        assert len({slot["family"] for slot in batch}) == 1


def test_even_plan_is_unchanged_by_weighting_support() -> None:
    even = make_slots(400, seed=7)
    assert [s["family"] for s in even[:len(FAMILIES)]] == [f.name for f in FAMILIES]
    assert make_slots(400, seed=7, weights={f.name: 1 for f in FAMILIES}) == even


def test_weights_scale_families_and_zero_excludes() -> None:
    weights = {"pairwise_answer_quality": 3, "pronoun_resolution": 2, "logical_deduction": 1}
    counts = Counter(s["family"] for s in make_slots(600, seed=1, weights=weights))
    assert counts == {"pairwise_answer_quality": 300, "pronoun_resolution": 200, "logical_deduction": 100}
