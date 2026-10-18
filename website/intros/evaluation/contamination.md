---
title: Contamination, tracking and the API
description: Every report says how much of the evaluation set the model saw in training, and where reports go.
---

# Contamination, tracking and the API

A model that trained on the test set scores well and means nothing. Rather than trusting that this never happened,
every evaluation report measures it, and every report can be read by a person, a dashboard or a script.

| Feature | What it gives |
|---|---|
| contamination | exact matches and near-duplicates between the evaluation items and the rows the experiment trained on, next to the scores |
| tracker | the report's numbers in Weights and Biases, once per report |
| API | `GET /api/evaluations` lists every report under an output directory |

Contamination is a measurement, not a filter: the Data channel is where leaks are removed before training.

## Details
