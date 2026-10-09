---
title: Config schema
description: Every key of every recipe's YAML, with its type, default and meaning. Generated from the config classes.
---

# Config schema

A config is one YAML with a `pipeline` block (`recipe`, `name`, `seed`, `output_dir`, `device`, `stages`) and one
section per concern. Every section rejects keys it does not know, so a typo is an error, not a silently ignored
setting. The tables below come straight from the Pydantic classes; nested keys are written with dots.
