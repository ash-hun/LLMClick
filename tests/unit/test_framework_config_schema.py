"""`framework` core/config/config_schema.py: a key added later stays out of the dump while it holds its default."""

from typing import ClassVar

from core.config.config_schema import Section


class Settings(Section):
    added_keys: ClassVar[frozenset[str]] = frozenset({"late", "tags"})
    early: int = 1
    late: bool = False
    tags: list[str] = []


def test_added_keys_are_dumped_only_when_set() -> None:
    assert Settings().model_dump() == {"early": 1}
    assert Settings(late=True, tags=["a"]).model_dump() == {"early": 1, "late": True, "tags": ["a"]}
    assert Settings(early=2, late=False).model_dump(mode="json") == {"early": 2}
