# Sample rows

Small rows per recipe (eighty arithmetic rows, or 120 policy records for the decision recipes), written by hand-rolled code and free of any licence. They exist so every
config under `configs/llm/` and `configs/embedding/` runs offline in minutes; they are not training data worth keeping.

| File | Recipe | Row shape |
|---|---|---|
| `llm_sft.jsonl` | `llm_sft` | `messages` ending with an assistant turn |
| `llm_instruction.jsonl` | `llm_instruction` | `instruction`, `input`, `output` |
| `llm_dpo.jsonl` | `llm_dpo` | `prompt`, `chosen`, `rejected` |
| `llm_grpo.jsonl` | `llm_grpo` | `prompt`, `answer` (read by the `exact_match` reward) |
| `llm_decision.jsonl` | `llm_decision_sft`, `llm_decision_cispo` | `state`, `questions` (120 records, 360 questions of type choice, noul and score) |
| `embedding_contrastive.jsonl` | `embedding_contrastive` | `query`, `positive`, `negative` |

`evaluation/` holds evaluation sets made for an experiment. They stay local (not committed: the repository holds
recipes and the framework) and apart from the rows above, because the tests build their tiny model's tokenizer from
every `samples/*.jsonl`: a new file there changes that model. `configs/evaluation/route.yaml` reads
`evaluation/route_boundary.jsonl`: 19 requests routed by hand, listed in the jarvis-router experiment page.
