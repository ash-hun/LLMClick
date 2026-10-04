"""Contrastive learning (InfoNCE): a query must score its positive above the batch's other documents."""

import torch
from pydantic import Field

from modeling.embedding.models.base import EmbeddingBackbone
from modeling.tuning.method import Row, TrainingMethod, chunks
from core.config.schema import Section
from core.progress import Progress


def negatives(row: Row) -> list[str]:
    value = row.get("negative") or []
    return [value] if isinstance(value, str) else list(value)


class Contrastive(TrainingMethod[EmbeddingBackbone]):
    """Rows: {"query": str, "positive": str, "negative"?: str | [str]}. Other rows' documents are negatives too."""

    class Config(Section):
        temperature: float = Field(default=0.05, gt=0)
        query_instruction: str | None = Field(default=None, description="Task description prepended to every query")

    def query(self, row: Row) -> str:
        instruction = self.config.query_instruction
        return f"Instruct: {instruction}\nQuery:{row['query']}" if instruction else str(row["query"])

    def check(self, row: Row) -> None:
        if not all(isinstance(text, str) and text for text in (row["query"], row["positive"], *negatives(row))):
            raise ValueError("query, positive and negative must be non-empty strings")

    def scores(self, backbone: EmbeddingBackbone, rows: list[Row]) -> tuple[torch.Tensor, list[str]]:
        """Similarity of every query to every document of the batch; row i's positive is document i."""
        documents = [row["positive"] for row in rows] + [text for row in rows for text in negatives(row)]
        queries = backbone.embed([self.query(row) for row in rows], self.training.max_length)
        return queries @ backbone.embed(documents, self.training.max_length).T, documents

    def loss(self, backbone: EmbeddingBackbone, rows: list[Row]) -> torch.Tensor:
        scores, _ = self.scores(backbone, rows)
        labels = torch.arange(len(rows), device=scores.device)
        self.metrics = {"accuracy": float((scores.argmax(dim=1) == labels).float().mean())}
        return torch.nn.functional.cross_entropy(scores / self.config.temperature, labels)

    @torch.no_grad()
    def evaluate(self, backbone: EmbeddingBackbone, rows: list[Row], batch_size: int, progress: Progress) -> dict[str, float]:
        """Retrieval over all held-out documents: is the positive ranked first (accuracy), and how high (mrr)."""
        documents = list(dict.fromkeys(text for row in rows for text in (row["positive"], *negatives(row))))
        batches = chunks([{"text": text} for text in documents], batch_size)
        vectors = []
        for index, batch in enumerate(batches):
            progress.update(index, len(batches), "documents")
            vectors.append(backbone.embed([item["text"] for item in batch], self.training.max_length))
        matrix = torch.cat(vectors)
        hits, reciprocal = 0, 0.0
        for batch in chunks(rows, batch_size):
            ranked = (backbone.embed([self.query(row) for row in batch], self.training.max_length) @ matrix.T).argsort(
                dim=1, descending=True).tolist()
            for row, order in zip(batch, ranked, strict=True):
                rank = order.index(documents.index(row["positive"])) + 1
                hits, reciprocal = hits + (rank == 1), reciprocal + 1 / rank
        progress.update(len(batches), len(batches))
        return {"accuracy": hits / len(rows), "mrr": reciprocal / len(rows), "documents": float(len(documents))}
