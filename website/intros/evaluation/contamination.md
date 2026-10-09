---
title: Contamination, tracking and the API
description: Every report says how much of the evaluation set the model saw in training.
---

# Contamination, tracking and the API

A model that trained on the test set scores well and means nothing. Rather than trusting that this never happened,
every evaluation report measures it: exact matches and near-duplicates between the evaluation items and the rows the
experiment trained on. The number sits next to the scores, and the Data channel is where leaks are removed before
training.

Reports can also go to Weights and Biases with one setting, and the HTTP API lists every report under an output
directory so a dashboard or a script can read them.
