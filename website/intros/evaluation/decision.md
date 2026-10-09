---
title: Decision models
description: evaluation_decision measures accuracy, calibration and latency, without and with reasoning.
---

# Decision models: `evaluation_decision`

Decision models are judged on three things: whether they pick the right option, whether their confidence means what
it says (calibration), and how fast they answer. This recipe measures all three on Jev rows or on the public JevBench
tiers, which it downloads on first use, and does it twice when asked: without reasoning and after the model reasons
first, so you can see what thinking buys and what it costs.
