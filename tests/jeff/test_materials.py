import asyncio

import pytest

from jeff import materials
from jeff.families import DOMAINS, make_slots


class FakeTeacher:
    """Returns PER_CALL distinct items per call, numbered so every call's items are unique."""

    def __init__(self, duplicates: bool = False) -> None:
        self.calls = 0
        self.duplicates = duplicates

    async def complete(self, *, system, user, schema, temperature, max_tokens):
        self.calls += 1
        base = 0 if self.duplicates else self.calls * 1000
        return {"items": [f"item {base + i}" for i in range(materials.PER_CALL)]}


def pools() -> materials.Materials:
    return {"people": [f"Person {i}" for i in range(600)],
            "organisations": {d: [f"{d} org {i}" for i in range(40)] for d in DOMAINS},
            "topics": {d: [f"{d} topic {i}" for i in range(60)] for d in DOMAINS}}


def test_build_collects_unique_pools() -> None:
    teacher = FakeTeacher()
    built = asyncio.run(materials.build(teacher))
    assert teacher.calls == len(materials.REGIONS) + 2 * len(DOMAINS)
    assert len(built["people"]) == len(materials.REGIONS) * materials.PER_CALL
    assert set(built["topics"]) == set(DOMAINS) == set(built["organisations"])


def test_build_raises_when_deduplication_leaves_too_few() -> None:
    with pytest.raises(ValueError, match="unique people"):
        asyncio.run(materials.build(FakeTeacher(duplicates=True)))


def test_dress_is_deterministic_and_draws_from_the_slot_domain() -> None:
    slots = make_slots(100, seed=5)
    first, second = materials.dress(slots, pools(), seed=5), materials.dress(slots, pools(), seed=5)
    assert first == second
    for slot in first:
        assert slot["topic"] in pools()["topics"][slot["domain"]]
        assert slot["organisation"] in pools()["organisations"][slot["domain"]]
        assert len(set(slot["people"].split(", "))) == materials.PEOPLE_PER_SLOT
    assert materials.dress(slots, pools(), seed=6) != first
