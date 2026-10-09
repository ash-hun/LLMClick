---
title: Your own rows
description: evaluation_custom scores a checkpoint on any rows with the metrics of the recipe that trained it.
---

# Your own rows: `evaluation_custom`

The quickest useful evaluation: point at a finished experiment, give it rows in the same format it trained on, and
get the same metrics `validate` reports, now on data of your choosing. Because the experiment's own config says how
the model loads and is measured, there is nothing to configure about the model; `checkpoint: base` measures the
weights the experiment started from, which is the baseline you compare against.
