"""`framework` modeling/modeling_stages.py: validation bounds name every metric that is missing, out of bounds or not finite."""

from modeling.modeling_config import ValidationConfig
from modeling.modeling_stages import violations


def test_violations() -> None:
    bounds = ValidationConfig(min={"accuracy": 0.8}, max={"ece": 0.1, "nll": 1.0})
    assert violations({"accuracy": 0.9, "ece": 0.05, "nll": 0.5}, bounds) == []
    assert violations({"accuracy": 0.7, "ece": 0.2}, bounds) == [
        "nll was not measured", "accuracy=0.7000 is below the minimum 0.8", "ece=0.2000 is above the maximum 0.1"]
    broken = violations({"accuracy": float("nan"), "ece": 0.05, "nll": float("inf")}, bounds)
    assert broken == ["accuracy is nan, not a finite number", "nll is inf, not a finite number"]  # NaN passes no bound
    assert violations({"loss": float("nan")}, ValidationConfig()) == ["loss is nan, not a finite number"]
