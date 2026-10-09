---
title: Comparing checkpoints
description: evaluation_compare puts several checkpoints and the base model through the same measurements.
---

# Comparing checkpoints: `evaluation_compare`

A single score means little; a difference against the base model, with its uncertainty, is what tells you whether
training helped. This recipe takes several runs (the base model is required), scores them on the same benchmarks,
rows, decision questions or embedding tasks, and reports every difference with a verdict: decided when it lies
outside the 95% interval of the two measurements, not decided when inside, and open when no interval exists.

Nothing is measured twice: each run's stages are the ones the single-model recipes use, so results already computed
are reused and adding a run scores only that run. The output is a table, a polygon chart over the axes you define,
and a Markdown summary you can paste into a report.
