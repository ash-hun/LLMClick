---
title: Evaluation overview
description: Measure what you trained, against the base model, with the leaks reported.
---

# Evaluation overview

Training tells you the loss went down. Evaluation tells you whether the model got better at what you care about, and
whether that number can be trusted. The Evaluation channel answers three questions about any checkpoint:

- How does it score on **my rows**, with the metrics of the recipe that trained it (`evaluation_custom`)?
- How does it score on **public benchmarks**, with the standard harness, and how does that compare with the **base
  model** under the same settings (`evaluation_benchmark`, `evaluation_compare`)?
- Did any of the evaluation items **leak** into training? Every report carries the overlap, so an inflated score is
  visible as such.

Decision models and embedding models have their own recipes, because their metrics are different (accuracy with
calibration and latency; retrieval scores on MTEB tasks).

Everything is a recipe on the same runner as training: results are cached per checkpoint, benchmark and settings,
so comparing three checkpoints scores each benchmark once per checkpoint and a benchmark added later is the only
one that runs. Reports land in `report.json` and are listed by the API.
