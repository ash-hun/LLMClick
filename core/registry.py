"""Named registries: a string key in the YAML config resolves to an implementation here."""

import importlib
from collections.abc import Callable
from typing import Any

CHANNELS = ("modeling",)  # packages whose import registers their recipes; Data and Evaluation join here


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


RECIPES = Registry("recipe")  # `pipeline.recipe` in the YAML -> Pipeline subclass


def load_channels() -> None:
    for channel in CHANNELS:
        importlib.import_module(channel)


def catalogue() -> dict[str, dict[str, list[str]]]:
    """Per recipe: its stages and the keys its own registries accept."""
    load_channels()
    return {name: RECIPES.get(name).catalogue() for name in RECIPES.names()}
