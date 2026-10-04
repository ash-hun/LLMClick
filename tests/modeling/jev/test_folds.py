import pytest

from modeling.jev.data.folds import carve
from tests.conftest import example

SIZES = {"dev": 5, "temperature": 3, "validation": 4}


def test_carve_is_disjoint_and_deterministic() -> None:
    rows = [example("x", i) for i in range(40)]
    a = carve(rows, SIZES, seed=1)
    assert a == carve(rows, SIZES, seed=1)
    ids = [row["id"] for fold in a.values() for row in fold]
    assert len(ids) == len(set(ids)) == 40
    assert {name: len(a[name]) for name in SIZES} == SIZES


def test_carve_refuses_to_eat_everything() -> None:
    with pytest.raises(ValueError):
        carve([example("x", i) for i in range(5)], {"dev": 3, "temperature": 1, "validation": 1}, seed=1)
