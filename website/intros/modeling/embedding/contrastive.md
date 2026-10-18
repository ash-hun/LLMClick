---
title: embedding_contrastive
sidebar_label: embedding_contrastive
description: Contrastive learning (InfoNCE) of an embedding model with in-batch and hard negatives.
---

# `embedding_contrastive`: retrieval embeddings

Contrastive learning pulls a query towards the text that answers it and away from the others in the batch, plus an
optional hard negative per row. Use it to adapt an embedding model to your domain's queries and documents: search,
retrieval for RAG, deduplication.

| `model.architecture` | Model | Pooling |
|---|---|---|
| `bi_encoder` | Qwen3-Embedding-0.6B | one tower for queries and documents, last-token pooling, unit vectors |

## Rows

```json
{"query": "sum of 9 and 2", "positive": "9 plus 2 equals 11.", "negative": "9 plus 2 equals 10."}
```

`negative` is optional and may be a list. Texts over `training.max_length` are truncated, as usual for retrieval.

## Config

<!-- include: configs/embedding/contrastive.yaml -->

## Use

```bash
uv run llmclick run configs/embedding/contrastive.yaml
```

- **Gate**: `accuracy` (the positive ranked first among the held-out documents) and `mrr`.
- **Settings**: `method.temperature` sharpens the softmax over candidates; `method.query_instruction` is prefixed to
  every query, as Qwen3-Embedding expects.
- **Output**: `output/contrastive-qwen3-embedding-0.6b-<hash>/validate/checkpoint`. Measure it on MTEB tasks with
  [`evaluation_embedding`](../../evaluation/embedding).
