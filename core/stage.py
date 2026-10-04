"""One unit of pipeline work: named, fingerprinted by its inputs, and safe to run twice in the same directory."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar

from core.progress import Progress
from core.config.schema import BaseConfig

ConfigT = TypeVar("ConfigT", bound=BaseConfig)
Outputs = dict[str, Any]


class Stage(ABC, Generic[ConfigT]):
    name: ClassVar[str]
    requires: ClassVar[tuple[str, ...]] = ()   # stages whose outputs this one reads
    sections: ClassVar[tuple[str, ...]] = ()   # config sections that decide the result
    version: ClassVar[int] = 1                 # bump when the code changes what the same inputs produce

    def __init__(self, config: ConfigT, progress: Progress) -> None:
        self.config = config
        self.progress = progress

    def identity(self) -> Any:
        """Everything besides upstream stages that decides the result; override to add e.g. hashes of input files."""
        return [self.config.section(name) for name in self.sections]

    @abstractmethod
    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        """Write into `workdir` and return JSON-safe outputs. `workdir` may hold files of an interrupted earlier
        run: reuse or replace them, never fail on them."""
