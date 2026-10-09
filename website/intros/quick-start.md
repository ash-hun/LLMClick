---
title: Quick start
description: Install, run a pilot config, and read what it wrote.
---

# Quick start

A pilot run takes a few minutes on a laptop: the shipped configs train a few steps on the sample rows so you can see
every stage end to end before pointing them at real data. You need Python 3.12 and `uv`; a GPU is optional (Apple
silicon trains on MPS).

What you will see after `llmclick run`:

- a progress bar per stage and one for the steps inside the running stage;
- an experiment directory `output/<name>-<hash>/` with `config.yaml`, `manifest.json`, `pipeline.log` and a link per
  stage;
- the validation report at `output/<name>-<hash>/validate/validation.json` and, if it passed, the checkpoint link at
  `validate/checkpoint`.

Run the same command again and every stage reports `cached`. Edit `training.lr` and only `train` and `validate` run
again; the data stage is shared.
