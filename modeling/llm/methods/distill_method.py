"""On-policy distillation: the student samples its own answers and a larger teacher grades every token of them."""

from pathlib import Path
from typing import Any

import torch
from pydantic import Field, model_validator

from modeling.llm.methods.llm_method import Encoded, LLMMethod, completion, conversation, token_log_probabilities
from modeling.llm.models.backbone_config import LLMBackboneConfig
from modeling.tuning.tuning_config import is_commit
from modeling.llm.models.llm_backbone import LLMBackbone
from modeling.tuning.training_method import Row, chunks
from core.config.config_schema import Section
from core.progress_reporter import Progress
from modeling.llm.models import LLM_BACKBONES


class TeacherModel(Section):
    architecture: str = Field(description="Key of the LLM model catalog, like model.architecture")
    name: str = Field(description="Hugging Face model ID, or a local directory")
    revision: str | None = Field(default=None, description="40-character commit; required unless `name` is a directory")

    @model_validator(mode="after")
    def _pinned(self) -> "TeacherModel":
        LLM_BACKBONES.check(self.architecture)
        if not Path(self.name).is_dir() and not is_commit(self.revision):
            raise ValueError("method.teacher.revision must be a 40-character commit hash (a branch name is not reproducible)")
        return self


class Distill(LLMMethod):
    """Rows: {"prompt": str | messages}. Per token of a sampled answer the student is pushed by how much less likely
    the teacher finds it than the student does: the reverse KL, estimated on the tokens the student actually chose,
    so the student learns where its own answers go wrong rather than imitating text it would not write."""

    class Config(Section):
        teacher: TeacherModel
        samples: int = Field(default=1, ge=1, description="Answers sampled per prompt")
        max_new_tokens: int = Field(default=64, ge=1)
        temperature: float = Field(default=1.0, gt=0, description="Sampling temperature of the student's answers")

    def __init__(self, config: Any, training: Any) -> None:
        super().__init__(config, training)
        self.teacher: LLMBackbone | None = None

    def check(self, row: Row) -> None:
        if not isinstance(row["prompt"], (str, list)):
            raise ValueError("prompt must be a string or messages")

    def lengths(self, backbone: LLMBackbone, row: Row) -> list[int]:
        return [len(backbone.render(conversation(row["prompt"]), True)) + self.config.max_new_tokens + 1]

    def load_teacher(self, backbone: LLMBackbone) -> LLMBackbone:
        """The teacher on the student's device. It scores the student's token ids, so both must share one vocabulary."""
        spec = self.config.teacher
        teacher: LLMBackbone = LLM_BACKBONES.get(spec.architecture)(LLMBackboneConfig(
            architecture=spec.architecture, name=spec.name, revision=spec.revision, template=backbone.config.template))
        teacher.load(backbone.device)
        if teacher.tokenizer.get_vocab() != backbone.tokenizer.get_vocab():
            teacher.release()
            raise ValueError(f"teacher {spec.name!r} has another vocabulary than the student; on-policy distillation "
                             "needs both to tokenize alike (e.g. Qwen3-0.6B with Qwen3-1.7B)")
        teacher.model.eval()
        teacher.model.requires_grad_(False)
        return teacher

    def prepare(self, backbone: LLMBackbone, rows: list[Row], workdir: Path, progress: Progress) -> list[Row]:
        """Loads the teacher, which stays beside the student until `finish`."""
        progress.update(0, None, "loading teacher")
        self.teacher = self.load_teacher(backbone)
        return rows

    def finish(self, backbone: LLMBackbone, progress: Progress) -> dict[str, float]:
        if self.teacher is not None:
            self.teacher.release()
            self.teacher = None
        return {}

    def answers(self, backbone: LLMBackbone, rows: list[Row], samples: int, temperature: float) -> list[Encoded]:
        prompts = [backbone.render(conversation(row["prompt"]), True) for row in rows]
        groups = backbone.generate(prompts, self.config.max_new_tokens, samples, temperature)
        return [completion(backbone, prompt, tokens, self.config.max_new_tokens)
                for prompt, group in zip(prompts, groups, strict=True) for tokens in group]

    @torch.no_grad()
    def teacher_scores(self, teacher: LLMBackbone, encoded: list[Encoded]) -> torch.Tensor:
        return token_log_probabilities(teacher, encoded)[0]

    def loss(self, backbone: LLMBackbone, rows: list[Row]) -> torch.Tensor:
        assert self.teacher is not None, "prepare loads the teacher before training"
        encoded = self.answers(backbone, rows, self.config.samples, self.config.temperature)
        student, graded = token_log_probabilities(backbone, encoded)
        gap = (student.detach() - self.teacher_scores(self.teacher, encoded).to(student.device)) * graded
        tokens = graded.sum().clamp(min=1)
        # the gradient of the reverse KL on the sampled tokens: each token's log-probability, weighted by its own gap
        loss: torch.Tensor = (gap * student).sum() / tokens
        self.metrics = {"reverse_kl": float(gap.sum() / tokens), "answer_tokens": float(graded.sum() / len(encoded))}
        return loss

    @torch.no_grad()
    def evaluate(self, backbone: LLMBackbone, rows: list[Row], batch_size: int, progress: Progress) -> dict[str, float]:
        """On the student's greedy answers: the teacher's mean token log-probability (`teacher_logp`, higher means the
        teacher would rather say it) and the mean gap between student and teacher (`reverse_kl`)."""
        owned = self.teacher is None  # the validate stage measures without `prepare`, so it loads the teacher here
        teacher = self.load_teacher(backbone) if owned else self.teacher
        assert teacher is not None
        try:
            teacher_total, gap_total, count, batches = 0.0, 0.0, 0.0, chunks(rows, batch_size)
            for index, batch in enumerate(batches):
                progress.update(index, len(batches), "teacher log-probabilities")
                encoded = self.answers(backbone, batch, 1, 0.0)
                student, graded = token_log_probabilities(backbone, encoded)
                scored = self.teacher_scores(teacher, encoded).to(student.device)
                teacher_total += float((scored * graded).sum())
                gap_total += float(((student - scored) * graded).sum())
                count += float(graded.sum())
            progress.update(len(batches), len(batches))
        finally:
            if owned:
                teacher.release()
        return {"teacher_logp": teacher_total / max(count, 1.0), "reverse_kl": gap_total / max(count, 1.0)}
