"""Config keys every recipe shares; a recipe subclasses BaseConfig and adds its own sections."""

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field


class Section(BaseModel):
    """A config section: an unknown key is an error, so a typo cannot silently fall back to a default."""
    model_config = ConfigDict(extra="forbid")


class BaseConfig(Section):
    # Keys that say how to run, not what to build: they never change the experiment directory or a stage fingerprint.
    identity_exclude: ClassVar[frozenset[str]] = frozenset({"stages", "output_dir"})

    recipe: str = Field(description="Which pipeline builds this model; see `llmclick recipes`")
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    seed: int = 20260920
    output_dir: str = "./output"
    device: Literal["cuda", "mps", "cpu"] | None = None
    stages: list[str] | None = Field(default=None, description="Stages to run, in the recipe's order; default all")

    def section(self, name: str) -> Any:
        """A JSON-safe view of one section, used to fingerprint stages."""
        value = getattr(self, name)
        return value.model_dump(mode="json") if isinstance(value, BaseModel) else value

    def identity(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude=set(self.identity_exclude))
