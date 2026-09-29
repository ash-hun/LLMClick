"""Named registries: every external dependency is a string key in the YAML config that resolves here."""

from collections.abc import Callable
from typing import Any


class Registry:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, Any] = {}

    def register(self, name: str) -> Callable[[Any], Any]:
        def decorate(item: Any) -> Any:
            existing = self._items.get(name)
            if existing is not None and existing is not item:
                raise ValueError(f"{self.kind} {name!r} is already registered")
            self._items[name] = item
            return item
        return decorate

    def get(self, name: str) -> Any:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(f"Unknown {self.kind} {name!r}; registered: {self.names()}") from None

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items


BUILDERS = Registry("builder")        # data builders: params -> Example JSONL
CONVERTERS = Registry("converter")    # raw dataset row -> Example
BACKBONES = Registry("backbone")      # model family -> jeff-train arguments
TEACHERS = Registry("teacher")        # synthetic-data teacher -> environment for jeff-generate
BENCHMARKS = Registry("benchmark")    # frozen evaluation sets: params -> Example JSONL
ALL = {r.kind: r for r in (BUILDERS, CONVERTERS, BACKBONES, TEACHERS, BENCHMARKS)}


def catalogue() -> dict[str, list[str]]:
    from core import modules  # noqa: F401  (importing fills the registries)
    return {kind: registry.names() for kind, registry in ALL.items()}
