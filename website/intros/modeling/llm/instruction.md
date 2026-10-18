---
title: llm_instruction
sidebar_label: llm_instruction
description: Instruction tuning on instruction, input, output records; the loss is on the output only.
---

# `llm_instruction`: instruction records

Instruction tuning fits datasets shaped as a task, an optional input and the expected output (Alpaca, most
classification and extraction sets). The instruction and input become the user turn, an optional system prompt goes
in front, and only the output is learned.

## Rows

```json
{"instruction": "Add the two numbers and reply with the sum only.", "input": "9 and 2", "output": "11"}
```

`input` may be left out. A Hub dataset with other column names maps them with `fields` (see the commented source in
the config).

## Config

<!-- include: configs/llm/instruction.yaml -->

## Use

```bash
uv run llmclick run configs/llm/instruction.yaml
```

- **Gate**: `validation.max: {loss: ...}` on the held-out outputs; `perplexity` is reported too.
- **Settings**: `method.system` is the system prompt in front of every instruction; leave it out for none.
- **Output**: `output/instruction-qwen3.5-0.8b-<hash>/validate/checkpoint`.
