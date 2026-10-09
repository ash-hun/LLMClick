---
title: Data overview
description: Make the rows a recipe trains on, with the filtering the published recipes use.
---

# Data overview

Most of the work in a good fine-tune is the data. The Data channel produces rows in the exact format a training
recipe expects, two ways: by converting open datasets, and by generating rows with a teacher model. Both end with
the same filters (duplicates out, evaluation items out) and the same manifest (what came from where, how much was
dropped at each step, what it cost).

Today the synthetic route is built: `data_synthetic` takes seeds (topics, personas, documents, existing rows), writes
instructions, optionally makes them harder, answers them several times with a teacher, verifies and judges the
answers, and assembles training rows. Teachers can be the Anthropic API, any OpenAI-format server, a local Ollama
model, or a checkpoint LLMClick trained. Converting open datasets (`data_open`) is designed and listed under Design.

Everything here is a recipe on the same runner, so a generation run that crashes or runs out of budget continues
where it stopped, and a second run of the same config costs nothing.
