"""Merge rules `llm_merge` offers: a `merge.rule` resolves here to a function that merges one tensor of every model."""

from dataclasses import dataclass, replace

import torch

from core.recipe_registry import Registry

RULES = Registry("merge rule")
TASK_RULES = {"task_arithmetic", "ties", "dare_linear", "dare_ties"}  # these add changes to `model`, the base
COLINEAR = 0.9995  # beyond this cosine the two directions are one and slerp falls back to linear interpolation


@dataclass(frozen=True)
class Tensors:
    """One tensor of each merged model, the base model's (for the task rules) and what the rule reads."""
    models: list[torch.Tensor]
    weights: list[float]
    base: torch.Tensor | None
    t: float
    density: float
    generator: torch.Generator


def trimmed(change: torch.Tensor, density: float) -> torch.Tensor:
    """TIES' trim: the largest `density` share of a change by magnitude, the rest set to zero."""
    keep = max(1, round(change.numel() * density))
    if keep >= change.numel():
        return change
    threshold = change.abs().flatten().kthvalue(change.numel() - keep + 1).values
    return torch.where(change.abs() >= threshold, change, torch.zeros_like(change))


def dropped(change: torch.Tensor, density: float, generator: torch.Generator) -> torch.Tensor:
    """DARE: each entry kept with probability `density` and rescaled by 1 / density, so the expected change is the same."""
    if density >= 1:
        return change
    kept = torch.rand(change.shape, generator=generator) < density
    return torch.where(kept, change / density, torch.zeros_like(change))


def elected(changes: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    """TIES' sign election and disjoint mean: per entry, the weighted mean of the changes that agree with the sign of
    their weighted sum; changes of the other sign do not cancel it out."""
    scale = torch.tensor(weights, dtype=changes[0].dtype).view(-1, *[1] * changes[0].dim())
    stacked = torch.stack(changes) * scale
    agree = (torch.sign(stacked) == torch.sign(stacked.sum(dim=0))) & (stacked != 0)
    return (stacked * agree).sum(dim=0) / (agree * scale).sum(dim=0).clamp(min=1e-12)


def weighted_sum(values: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    scale = torch.tensor(weights, dtype=values[0].dtype).view(-1, *[1] * values[0].dim())
    return (torch.stack(values) * scale).sum(dim=0)


def changes(tensors: Tensors) -> list[torch.Tensor]:
    assert tensors.base is not None
    return [model - tensors.base for model in tensors.models]


@RULES.register("linear")
def linear(tensors: Tensors) -> torch.Tensor:
    """The weighted mean of the models (model soup)."""
    return weighted_sum(tensors.models, tensors.weights) / sum(tensors.weights)


@RULES.register("slerp")
def slerp(tensors: Tensors) -> torch.Tensor:
    """Spherical interpolation between two models at `t` (0 is the first, 1 the second), along the arc between their
    directions rather than the chord, so the merged tensor keeps their norm."""
    first, second = tensors.models
    a, b = first.flatten(), second.flatten()
    cosine = float(torch.dot(a, b) / (a.norm() * b.norm()).clamp(min=1e-12))
    if abs(cosine) > COLINEAR:
        return (1 - tensors.t) * first + tensors.t * second
    angle = torch.arccos(torch.tensor(cosine))
    sine = torch.sin(angle)
    merged: torch.Tensor = (torch.sin((1 - tensors.t) * angle) / sine) * first + (torch.sin(tensors.t * angle) / sine) * second
    return merged


@RULES.register("task_arithmetic")
def task_arithmetic(tensors: Tensors) -> torch.Tensor:
    """The base plus the weighted sum of every model's change from it."""
    assert tensors.base is not None
    return tensors.base + weighted_sum(changes(tensors), tensors.weights)


@RULES.register("ties")
def ties(tensors: Tensors) -> torch.Tensor:
    """TIES-merging: each change trimmed to its largest `density` share, a sign elected per entry, the agreeing
    changes averaged."""
    assert tensors.base is not None
    return tensors.base + elected([trimmed(change, tensors.density) for change in changes(tensors)], tensors.weights)


@RULES.register("dare_linear")
def dare_linear(tensors: Tensors) -> torch.Tensor:
    """DARE then task arithmetic: random entries of each change dropped, the rest rescaled, the changes summed."""
    assert tensors.base is not None
    return tensors.base + weighted_sum([dropped(change, tensors.density, tensors.generator) for change in changes(tensors)],
                                       tensors.weights)


@RULES.register("dare_ties")
def dare_ties(tensors: Tensors) -> torch.Tensor:
    """DARE's random drop in place of TIES' trim, then TIES' sign election."""
    assert tensors.base is not None
    return tensors.base + elected([dropped(change, tensors.density, tensors.generator) for change in changes(tensors)],
                                  tensors.weights)


def merge_tensor(rule: str, tensors: Tensors) -> torch.Tensor:
    """One merged tensor in the models' own dtype; computed in FP32 so bf16 weights lose nothing on the way."""
    dtype = tensors.models[0].dtype
    if not dtype.is_floating_point:
        return tensors.models[0]  # integer buffers (position ids and the like) are equal across fine-tunes of one base
    merged: torch.Tensor = RULES.get(rule)(replace(tensors, models=[model.float() for model in tensors.models],
                                                   base=tensors.base.float() if tensors.base is not None else None))
    return merged.to(dtype)
