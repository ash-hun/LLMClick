# Data channel

Status: planned. This directory holds no code yet.

## Purpose

Pipelines that produce the rows the Modeling channel trains on: collecting and converting public datasets, generating
synthetic rows with a teacher model, and filtering what must not reach training (duplicates, rows that leak a
held-out or benchmark item).

## What it will hand to Modeling

JSONL files in the row format of a recipe (see the "Stages" table in the root README), for example Jev records
(`state`, `questions`) for the decision recipes or `query`/`positive`/`negative` for contrastive learning. Modeling
reads them through `data.sources` and never needs to know how they were built.

## How it plugs in

A channel is a package that registers recipes. It reuses the framework in `core/` as it is:

- `Stage` and `Pipeline` for fingerprinted, rerunnable steps, and `Experiment` for the output layout
- `Progress` for the CLI bars and the API job state
- `core/registry.py`: add `"data"` to `CHANNELS`, and register each pipeline with `@recipe`

Users then pick a data pipeline the same way they pick a model recipe: `pipeline.recipe` in a YAML, run with
`llmclick run`.

## Known starting points

- Row sources (`local_jsonl`, `huggingface`) live in `modeling/tuning/sources.py` today; they belong here once this
  channel exists.
- The vendored jeff package that was removed carried a data pipeline (27 public datasets converted into decision
  rows, teacher-written synthetic questions, a near-duplicate leak filter). It is in the git history before the
  commit "vendored jeff 와 jev 레시피 제거" and is the reference for the first decision-data pipeline.
