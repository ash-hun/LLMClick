---
title: Decision models
description: evaluation_decision measures accuracy, calibration and latency, without and with reasoning.
---

# Decision models: `evaluation_decision`

A decision model is judged on three things: whether it picks the right option, whether its confidence means what it
says, and how fast it answers. This recipe measures all three, on your Jev rows or on the public JevBench tiers, and
does it twice when asked: without reasoning and after the model reasons first, so you can see what thinking buys and
what it costs.

| Number | Tells you |
|---|---|
| `accuracy` (per tier against chance when rows carry tiers) | whether it picks the right option |
| `nll`, `ece` | whether a 0.9 means right nine times in ten; calibrated with the checkpoint's temperature |
| latency p50, p95 | seconds a question at the batch size used |
| reasoning tokens, closed chains | what thinking costs (with `think: on` or `both`) |

## Config and details
