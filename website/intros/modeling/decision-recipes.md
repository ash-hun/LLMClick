---
title: Decision recipes
description: Models that answer typed questions about a state with calibrated probabilities.
---

# Decision recipes

A decision model does not write text. Given a state (a document, a ticket, an event) and a typed question (pick one
of these options, yes or no, a level on a scale), it returns a probability per option. That makes it cheap to run,
easy to threshold, and measurable with accuracy and calibration instead of judges.

LLMClick builds them on top of a language model by adding a small head that reads the options, trained in two
possible stages: `llm_decision_sft` teaches the head to decide, optionally with a reasoning chain in front as
context; `llm_decision_cispo` continues from such a checkpoint and reinforces reasoning that leads the head to the
right option. Both fit a temperature at the end so the probabilities are calibrated, and validation reports accuracy,
negative log-likelihood and expected calibration error.

The row format follows the Jev request format (a state with one or more questions); the heads and prompt markers
follow the open jeff and Jeeves projects, reimplemented on this framework's loop.
