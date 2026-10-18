---
title: Decision recipes
description: Models that answer typed questions about a state with calibrated probabilities, trained in one or two stages.
---

# Decision recipes

A decision model does not write text. Given a state (a request, a ticket, a document, an event) and a typed question
(pick one of these options, yes or no, a level on a scale), it returns a probability per option. That makes it one
forward pass, easy to threshold, and measurable with accuracy and calibration instead of judges. A router that
decides whether an agent thinks before answering is one ([experiment](../../experiments/decoder-llm/jarvis-router)).

| Recipe | Stage | What it adds | Validation metrics |
|---|---|---|---|
| [`llm_decision_sft`](./sft) | 1 | a head that reads the options, trained with cross-entropy; optionally reasoning chains as context | `accuracy`, `nll`, `ece` |
| [`llm_decision_cispo`](./cispo) | 2, from a stage-1 checkpoint | reasoning before the decision, reinforced when it leads the head to the right option | the same, plus `think_accuracy` |

Both build on the LLM base models (`transformer`, `hybrid`), fit a temperature at the end so the probabilities are
calibrated, and write it next to the head (`head.json`). The row format follows the Jev request format: a state
with one or more questions.

```json
{"state": "Order total: $62. Free shipping applies to orders of $70 or more.",
 "questions": {"free_shipping": {"type": "noul", "instructions": "Does free shipping apply to this order?", "label": false},
               "tier": {"type": "choice", "instructions": "Which customer tier does the order total fall into?",
                        "criteria": {"bronze": "under $30", "silver": "$30 to $69", "gold": "$70 or more"}, "label": "silver"}}}
```

Question types: `choice` (one of `criteria`), `noul` (yes or no), `score` (a level on a scale). One question per row
(`state`, `question`, `label`) is accepted too. The heads and prompt markers follow the open jeff and Jeeves
projects, reimplemented on this framework's loop, so their released checkpoints do not load here.

## How the heads and stages work
