"""jev's swappable parts: every external dependency is a `name:` key in the YAML that resolves here."""

from core.registry import Registry

BUILDERS = Registry("builder")        # data builders: params -> Example JSONL
CONVERTERS = Registry("converter")    # raw dataset row -> Example
BACKBONES = Registry("backbone")      # model family -> jeff-train arguments
TEACHERS = Registry("teacher")        # synthetic-data teacher -> environment for jeff-generate
BENCHMARKS = Registry("benchmark")    # frozen evaluation sets: params -> Example JSONL
ALL = {r.kind: r for r in (BUILDERS, CONVERTERS, BACKBONES, TEACHERS, BENCHMARKS)}


def fill() -> None:
    """Import the modules whose decorators register the built-in entries."""
    from modeling.jev.data import builders, converters, synthetic  # noqa: F401
    from modeling.jev.evaluation import benchmarks  # noqa: F401
    from modeling.jev.model import backbones  # noqa: F401


def names() -> dict[str, list[str]]:
    fill()
    return {kind: registry.names() for kind, registry in ALL.items()}
