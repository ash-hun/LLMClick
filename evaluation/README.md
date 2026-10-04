# Evaluation channel

Status: planned. This directory holds no code yet.

## Purpose

Pipelines that score a trained model beyond the validation gate: on custom evaluation sets, on public benchmarks,
and across several models at once, with a report a person can read (per-axis scores drawn as a polygon chart).

## How it differs from `validate`

The Modeling channel's `validate` stage is a gate: one checkpoint, the rows held out from its own training data,
pass or fail against `validation.min` / `validation.max`. Evaluation is a measurement: any checkpoint, any number of
evaluation sets, results kept for comparison.

## What it will read from Modeling

A checkpoint directory. `output/<experiment>/validate/checkpoint` is the one to use: it exists only when the model
passed validation.

## How it plugs in

A channel is a package that registers recipes. It reuses the framework in `core/` as it is:

- `Stage` and `Pipeline` for fingerprinted, rerunnable steps, and `Experiment` for the output layout
- `Progress` for the CLI bars and the API job state
- `core/registry.py`: add `"evaluation"` to `CHANNELS`, and register each pipeline with `@recipe`

## Known starting points

- Each training method already has an `evaluate` that returns its metrics on a set of rows
  (`modeling/tuning/method.py`); an evaluation pipeline can call it on rows that were never part of training.
- The removed jeff package carried benchmark builders (a five-benchmark panel and JevBench) for decision models;
  they are in the git history before the commit "vendored jeff 와 jev 레시피 제거".
