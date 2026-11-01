"""`llm_merge` modeling/merge/merge_rule.py: what each rule computes on small tensors."""

import pytest
import torch


def tensors(models: list[list[float]], weights: list[float] | None = None, base: list[float] | None = None,
            t: float = 0.5, density: float = 1.0, seed: int = 0) -> object:
    from modeling.merge.merge_rule import Tensors

    return Tensors([torch.tensor(m) for m in models], weights or [1.0] * len(models),
                   torch.tensor(base) if base is not None else None, t, density, torch.Generator().manual_seed(seed))


def test_linear_and_task_arithmetic_weigh_the_models() -> None:
    from modeling.merge.merge_rule import RULES

    assert torch.allclose(RULES.get("linear")(tensors([[0.0, 4.0], [4.0, 0.0]], [1.0, 3.0])), torch.tensor([3.0, 1.0]))
    merged = RULES.get("task_arithmetic")(tensors([[2.0, 1.0], [1.0, 3.0]], [0.5, 1.0], base=[1.0, 1.0]))
    assert torch.allclose(merged, torch.tensor([1.5, 3.0]))


def test_ties_keeps_the_changes_that_agree_with_the_elected_sign() -> None:
    from modeling.merge.merge_rule import RULES, trimmed

    merged = RULES.get("ties")(tensors([[2.0, -1.0, 0.1], [1.0, 3.0, -0.2]], base=[0.0, 0.0, 0.0]))
    assert torch.allclose(merged, torch.tensor([1.5, 3.0, -0.2]))
    assert torch.equal(trimmed(torch.tensor([0.1, -5.0, 2.0, 0.3]), 0.5), torch.tensor([0.0, -5.0, 2.0, 0.0]))


def test_dare_drops_the_same_entries_for_the_same_seed_and_keeps_the_mean() -> None:
    from modeling.merge.merge_rule import RULES

    change = [[1.0] * 20000]
    first = RULES.get("dare_linear")(tensors(change, base=[0.0] * 20000, density=0.25, seed=7))
    again = RULES.get("dare_linear")(tensors(change, base=[0.0] * 20000, density=0.25, seed=7))
    assert torch.equal(first, again) and set(first.unique().tolist()) == {0.0, 4.0}
    assert float(first.mean()) == pytest.approx(1.0, abs=0.05)


def test_slerp_runs_from_the_first_model_to_the_second_and_keeps_the_norm() -> None:
    from modeling.merge.merge_rule import RULES

    a, b = [1.0, 0.0], [0.0, 1.0]
    assert torch.allclose(RULES.get("slerp")(tensors([a, b], t=0.0)), torch.tensor(a), atol=1e-6)
    assert torch.allclose(RULES.get("slerp")(tensors([a, b], t=1.0)), torch.tensor(b), atol=1e-6)
    assert float(RULES.get("slerp")(tensors([a, b], t=0.5)).norm()) == pytest.approx(1.0)
    assert torch.allclose(RULES.get("slerp")(tensors([[1.0, 1.0], [2.0, 2.0]], t=0.5)), torch.tensor([1.5, 1.5]))


def test_merge_keeps_the_dtype_and_passes_integer_buffers_through() -> None:
    from modeling.merge.merge_rule import Tensors, merge_tensor

    halves = [torch.tensor([1.0, 2.0], dtype=torch.bfloat16), torch.tensor([3.0, 4.0], dtype=torch.bfloat16)]
    merged = merge_tensor("linear", Tensors(halves, [1.0, 1.0], None, 0.5, 1.0, torch.Generator()))
    assert merged.dtype == torch.bfloat16 and torch.equal(merged, torch.tensor([2.0, 3.0], dtype=torch.bfloat16))
    ids = [torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])]
    assert torch.equal(merge_tensor("ties", Tensors(ids, [1.0, 1.0], torch.tensor([0, 1, 2]), 0.5, 1.0, torch.Generator())), ids[0])
