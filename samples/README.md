# Sample rows

Small rows per recipe (eighty arithmetic rows, or 120 policy records for the decision recipes), written by hand-rolled code and free of any licence. They exist so every
config under `configs/llm/` and `configs/embedding/` runs offline in minutes; they are not training data worth keeping.

| File | Recipe | Row shape |
|---|---|---|
| `llm_sft.jsonl` | `llm_sft` | `messages` ending with an assistant turn |
| `llm_tool_sft.jsonl` | `llm_sft` | `tools`, `messages` with `tool_calls` and `tool` turns (three made-up functions; one row in four answers without a call) |
| `llm_instruction.jsonl` | `llm_instruction` | `instruction`, `input`, `output` |
| `llm_dpo.jsonl` | `llm_dpo`, `llm_orpo`, `llm_simpo`, `llm_reward` | `prompt`, `chosen`, `rejected` |
| `llm_kto.jsonl` | `llm_kto` | `prompt`, `completion`, `label` (the first forty DPO pairs, each split into a desirable and an undesirable row) |
| `llm_grpo.jsonl` | `llm_grpo`, `llm_distill` | `prompt`, `answer` (read by the `exact_match` reward; `llm_distill` reads the prompt only) |
| `llm_decision.jsonl` | `llm_decision_sft`, `llm_decision_cispo` | `state`, `questions` (120 records, 360 questions of type choice, noul and score) |
| `embedding_contrastive.jsonl` | `embedding_contrastive` | `query`, `positive`, `negative` |

`evaluation/` holds evaluation sets made for an experiment. They stay local (not committed: the repository holds
recipes and the framework) and apart from the rows above, because the tests build their tiny model's tokenizer from
the sample files `tests/conftest.py` lists in `CORPUS`; a file added here leaves that model as it is. `configs/evaluation/route.yaml` reads
`evaluation/route_boundary.jsonl`: 19 requests routed by hand, listed in the jarvis-router experiment page.
