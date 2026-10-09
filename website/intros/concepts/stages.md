---
title: Stages and metrics
description: The three stages of every tuning recipe, the row format each recipe expects, and the metrics validation can check.
---

# Stages and metrics

Every tuning recipe is `data` then `train` then `validate`. The tables here are the ones to keep open while writing a
config: which keys your rows need, and which metric names `validation.min` and `validation.max` may refer to.
