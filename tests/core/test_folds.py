import pytest

from core.modules.data.folds import carve
from tests.core.conftest import example


def test_carve_is_disjoint_and_deterministic() -> None:
    rows = [example("x", i) for i in range(40)]
    a = carve(rows, dev=5, temperature=3, seed=1)
    b = carve(rows, dev=5, temperature=3, seed=1)
    assert a == b
    ids = [row["id"] for fold in a.values() for row in fold]
    assert len(ids) == len(set(ids)) == 40
    assert len(a["dev"]) == 5 and len(a["temperature"]) == 3


def test_carve_refuses_to_eat_everything() -> None:
    with pytest.raises(ValueError):
        carve([example("x", i) for i in range(5)], dev=3, temperature=2, seed=1)
