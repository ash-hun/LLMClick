# Sample rows

Eighty small arithmetic rows per recipe, written by hand-rolled code and free of any licence. They exist so every
config under `configs/llm/` and `configs/embedding/` runs offline in minutes; they are not training data worth keeping.

| File | Recipe | Row shape |
|---|---|---|
| `llm_sft.jsonl` | `llm_sft` | `messages` ending with an assistant turn |
| `llm_instruction.jsonl` | `llm_instruction` | `instruction`, `input`, `output` |
| `llm_dpo.jsonl` | `llm_dpo` | `prompt`, `chosen`, `rejected` |
| `llm_grpo.jsonl` | `llm_grpo` | `prompt`, `answer` (read by the `exact_match` reward) |
| `embedding_contrastive.jsonl` | `embedding_contrastive` | `query`, `positive`, `negative` |
