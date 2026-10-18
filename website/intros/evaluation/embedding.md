---
title: Embedding models
description: evaluation_embedding runs MTEB tasks through the mteb package.
---

# Embedding models: `evaluation_embedding`

Retrieval quality is measured on MTEB tasks, the standard for embedding models, which covers retrieval, sentence
similarity, classification, clustering and reranking. This recipe wraps the experiment's backbone as an MTEB encoder,
so the pooling and instructions are exactly the ones training used, and caches one stage per task. Any MTEB task runs
by name; `python -c "import mteb; print(mteb.get_tasks())"` lists them.

## Config and details
