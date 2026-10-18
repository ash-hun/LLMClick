---
title: wakeword
sidebar_label: wakeword
description: Trains the detector that fires when the agent hears its name, and tests it on name-then-command streams.
---

# `wakeword`: the agent's name

A voice agent listens all the time, but it should act only after hearing its name, such as "Jarvis". This recipe
trains the small detector that listens for that name; the command that follows it ("wake up", "search the web") is
heard by speech recognition afterwards. It uses openWakeWord's frozen speech features and trains only a small scoring
head on top of them, so a few hundred synthetic recordings are enough. The exported model loads in openWakeWord's
runtime.

Validation plays the agent's first moment back to the detector: the user says the name and then something else, with
noise underneath, and the detector must fire exactly once, on the name and not on what follows.

## Rows

Two sets: `positive` recordings say the name and nothing else; `negative` recordings never say it (near misses such
as "Travis", the commands said alone, ordinary speech). Rows are `audio` and an optional `text`.

## Config

<!-- include: configs/wakeword/jarvis.yaml -->

## Use

```bash
uv run llmclick run configs/data/wake_phrase.yaml      # the name, read by 300 speakers
uv run llmclick run configs/data/wake_negatives.yaml   # near misses, commands, ordinary lines
uv run llmclick run configs/data/wake_nearmiss.yaml    # more near misses, more speakers
uv run llmclick run configs/wakeword/jarvis.yaml       # replace each <hash> first
```

Measure a detector on voices and hours of audio it never heard with
[`evaluation_wakeword`](../../evaluation/wakeword).

## How it works
