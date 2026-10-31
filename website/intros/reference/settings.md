---
title: Settings
description: Process-level settings read from the environment and environment/.env.
---

# Settings

These are the few things that belong to the machine rather than to an experiment, so they live in the environment
(or `environment/.env`, copied from `.env.sample`) instead of a config. Everything about a model or a run is in its
YAML.

Logging is read by `core/logger.py` before any setting is loaded: `LOG_FORMAT` is `console` (the default, for people)
or `json` (one object per line, what `compose.yaml` sets), and `LOG_LEVEL` defaults to `INFO`. Both
the terminal and each experiment's `pipeline.log` use that format.
