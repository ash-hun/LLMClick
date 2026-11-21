"""On-policy distillation: the student samples its own answers and a larger teacher grades every token of them."""

from pathlib import Path
from typing import Any

import torch
from pydantic import Field, model_validator

from modeling.llm.methods.llm_method import Encoded, LLMMethod, completion, conversation, token_log_probabilities
from modeling.llm.models.backbone_config import LLMBackboneConfig
from modeling.llm.models.llm_backbone import LLMBackbone
from modeling.tuning.training_method import Row, chunks
from modeling.tuning.tuning_config import is_commit
from core.config.config_schema import Section
from modeling.llm.models import LLM_BACKBONES
from core.progress_reporter import Progress

ALL = ""  # the domain of every row when a single teacher grades them all


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


class OnPolicyDistillation(LLMMethod):
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
        self.teachers: dict[str, LLMBackbone] = {}

    def teacher_specs(self) -> dict[str, TeacherModel]:
        """The teacher of each domain."""
        return {ALL: self.config.teacher}

    def domain(self, row: Row) -> str:
        return ALL

    def check(self, row: Row) -> None:
        if not isinstance(row["prompt"], (str, list)):
            raise ValueError("prompt must be a string or messages")

    def lengths(self, backbone: LLMBackbone, row: Row) -> list[int]:
        return [len(backbone.render(conversation(row["prompt"]), True)) + self.config.max_new_tokens + 1]

    def load_teacher(self, backbone: LLMBackbone, spec: TeacherModel) -> LLMBackbone:
        """A teacher on the student's device. It scores the student's token ids, so both must share one vocabulary."""
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

    def load_teachers(self, backbone: LLMBackbone) -> dict[str, LLMBackbone]:
        """Every teacher at once; a teacher that fails to load releases the ones loaded before it."""
        # ponytail: every teacher stays on the device, so memory is their sum; offload idle teachers to the CPU when
        # one device cannot hold them all
        teachers: dict[str, LLMBackbone] = {}
        try:
            for domain, spec in self.teacher_specs().items():
                teachers[domain] = self.load_teacher(backbone, spec)
        except BaseException:
            release(teachers)
            raise
        return teachers

    def prepare(self, backbone: LLMBackbone, rows: list[Row], workdir: Path, progress: Progress) -> list[Row]:
        """Loads the teachers, which stay beside the student until `finish`."""
        progress.update(0, None, "loading teacher")
        self.teachers = self.load_teachers(backbone)
        return rows

    def finish(self, backbone: LLMBackbone, progress: Progress) -> dict[str, float]:
        release(self.teachers)
        self.teachers = {}
        return {}

    def answers(self, backbone: LLMBackbone, rows: list[Row], samples: int, temperature: float) -> list[Encoded]:
        prompts = [backbone.render(conversation(row["prompt"]), True) for row in rows]
        groups = backbone.generate(prompts, self.config.max_new_tokens, samples, temperature)
        return [completion(backbone, prompt, tokens, self.config.max_new_tokens)
                for prompt, group in zip(prompts, groups, strict=True) for tokens in group]

    @torch.no_grad()
    def teacher_scores(self, teachers: dict[str, LLMBackbone], rows: list[Row], encoded: list[Encoded],
                       width: int) -> torch.Tensor:
        """Each answer's token log-probabilities under its row's teacher; `encoded` holds the answers of each row in
        turn, and `width` is the student's (tokens - 1)."""
        repeats = len(encoded) // len(rows)
        domains = [self.domain(row) for row in rows for _ in range(repeats)]
        scores = torch.zeros(len(encoded), width)
        for domain in dict.fromkeys(domains):
            picked = [index for index, owner in enumerate(domains) if owner == domain]
            scored = token_log_probabilities(teachers[domain], [encoded[index] for index in picked])[0].float().cpu()
            scores[picked, :scored.shape[1]] = scored  # right-padded, so a narrower batch fills the left columns
        return scores

    def advantages(self, backbone: LLMBackbone, rows: list[Row], encoded: list[Encoded]) -> torch.Tensor:
        """A per-answer push added to every one of its tokens on top of the teacher's; none here."""
        return torch.zeros(len(encoded))

    def loss(self, backbone: LLMBackbone, rows: list[Row]) -> torch.Tensor:
        assert self.teachers, "prepare loads the teachers before training"
        encoded = self.answers(backbone, rows, self.config.samples, self.config.temperature)
        student, graded = token_log_probabilities(backbone, encoded)
        scored = self.teacher_scores(self.teachers, rows, encoded, student.shape[1]).to(student.device, student.dtype)
        gap = (student.detach() - scored) * graded
        tokens = graded.sum().clamp(min=1)
        self.metrics = {"reverse_kl": float(gap.sum() / tokens), "answer_tokens": float(graded.sum() / len(encoded))}
        advantage = self.advantages(backbone, rows, encoded).to(student.device, student.dtype)
        # the gradient of the reverse KL on the sampled tokens: each token's log-probability, weighted by its own gap
        loss: torch.Tensor = ((gap - advantage[:, None] * graded) * student).sum() / tokens
        return loss

    @torch.no_grad()
    def evaluate(self, backbone: LLMBackbone, rows: list[Row], batch_size: int, progress: Progress) -> dict[str, float]:
        """On the student's greedy answers: the teacher's mean token log-probability (`teacher_logp`, higher means the
        teacher would rather say it) and the mean gap between student and teacher (`reverse_kl`), also per domain
        when several teachers share the rows."""
        owned = not self.teachers  # the validate stage measures without `prepare`, so it loads the teachers here
        teachers = self.load_teachers(backbone) if owned else self.teachers
        totals: dict[str, list[float]] = {}  # domain -> teacher log-probability, gap, tokens
        try:
            batches = chunks(rows, batch_size)
            for index, batch in enumerate(batches):
                progress.update(index, len(batches), "teacher log-probabilities")
                encoded = self.answers(backbone, batch, 1, 0.0)
                student, graded = token_log_probabilities(backbone, encoded)
                scored = self.teacher_scores(teachers, batch, encoded, student.shape[1]).to(student.device, student.dtype)
                for row, mine, theirs, mask in zip(batch, student, scored, graded, strict=True):
                    total = totals.setdefault(self.domain(row), [0.0, 0.0, 0.0])
                    total[0] += float((theirs * mask).sum())
                    total[1] += float(((mine - theirs) * mask).sum())
                    total[2] += float(mask.sum())
            progress.update(len(batches), len(batches))
        finally:
            if owned:
                release(teachers)
        overall = [sum(total[column] for total in totals.values()) for column in range(3)]
        metrics = {"teacher_logp": overall[0] / max(overall[2], 1.0), "reverse_kl": overall[1] / max(overall[2], 1.0)}
        for domain, total in sorted(totals.items()):
            if domain != ALL:
                metrics[f"teacher_logp/{domain}"] = total[0] / max(total[2], 1.0)
                metrics[f"reverse_kl/{domain}"] = total[1] / max(total[2], 1.0)
        return metrics


def release(teachers: dict[str, LLMBackbone]) -> None:
    for teacher in teachers.values():
        teacher.release()
