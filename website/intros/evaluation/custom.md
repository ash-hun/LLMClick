---
title: Your own rows
description: evaluation_custom scores a checkpoint on any rows with the metrics of the recipe that trained it.
---

# Your own rows: `evaluation_custom`

The quickest useful evaluation, for any LLM, decision or embedding model Modeling trained: point at a finished experiment, give it rows in the
same format it trained on, and get the metrics its `validate` reports (loss and perplexity for SFT, accuracy and
calibration for a decision model, ranking accuracy for an embedding model), now on data of your choosing. The speech
recipes have their own measurements: `tts_piper`'s validation and `evaluation_wakeword`.

| `model.checkpoint` | Measures |
|---|---|
| `validate` | the checkpoint that passed validation |
| `train` | the last trained weights, also when validation failed |
| `base` | the weights the experiment started from: the baseline |
| a directory | any checkpoint the experiment's recipe can load |

Because the experiment's own config says how the model loads and is measured, there is nothing to configure about
the model.

## Config and details
