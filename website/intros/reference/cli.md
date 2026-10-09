---
title: CLI
description: The llmclick command and its subcommands, as the program prints them.
---

# CLI

`llmclick run` runs the stages a config asks for, `validate` checks a config and prints its experiment key and stage
plan without running anything, `recipes` lists what this installation can build, and `serve` starts the HTTP API.
Paths in a config and `environment/.env` are read relative to the working directory; `LLMCLICK_ENV` names another
settings file.
