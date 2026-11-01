"""`llm_distill` modeling/llm/methods/distill_method.py: the teacher's pin and vocabulary, the reverse-KL gradient."""

import shutil
from pathlib import Path

import pytest


def test_teacher_must_be_pinned_and_registered(tiny_model: Path) -> None:
    from modeling.llm.methods.distill_method import TeacherModel

    assert TeacherModel(architecture="transformer", name=str(tiny_model)).revision is None
    with pytest.raises(ValueError, match="40-character commit"):
        TeacherModel(architecture="transformer", name="Qwen/Qwen3-1.7B", revision="main")
    with pytest.raises(ValueError, match="Unknown"):
        TeacherModel(architecture="mamba", name=str(tiny_model))


def test_teacher_with_another_vocabulary_is_refused(tiny_model: Path, tmp_path: Path) -> None:
    from transformers import AutoTokenizer
    from modeling.llm.methods.distill_method import Distill
    from modeling.llm.models.backbone_config import LLMBackboneConfig
    from modeling.llm.models.transformer_backbone import TransformerBackbone
    from modeling.tuning.tuning_config import TrainingConfig

    other = tmp_path / "other"
    shutil.copytree(tiny_model, other)
    tokenizer = AutoTokenizer.from_pretrained(other)
    tokenizer.add_tokens(["zebra"])
    tokenizer.save_pretrained(other)
    student = TransformerBackbone(LLMBackboneConfig(architecture="transformer", name=str(tiny_model)))
    student.load("cpu")
    method = Distill(Distill.Config(teacher={"architecture": "transformer", "name": str(other)}), TrainingConfig())
    with pytest.raises(ValueError, match="another vocabulary"):
        method.load_teacher(student)
    same = Distill(Distill.Config(teacher={"architecture": "transformer", "name": str(tiny_model)}), TrainingConfig())
    assert same.load_teacher(student).tokenizer.get_vocab() == student.tokenizer.get_vocab()


def test_a_student_that_is_its_teacher_has_no_gap_and_no_gradient(tiny_model: Path) -> None:
    from modeling.llm.methods.distill_method import Distill
    from modeling.llm.models.backbone_config import LLMBackboneConfig
    from modeling.llm.models.transformer_backbone import TransformerBackbone
    from modeling.tuning.tuning_config import TrainingConfig
    from core.progress_reporter import Progress

    student = TransformerBackbone(LLMBackboneConfig(architecture="transformer", name=str(tiny_model)))
    student.load("cpu")
    method = Distill(Distill.Config(teacher={"architecture": "transformer", "name": str(tiny_model)}, max_new_tokens=4,
                                    samples=2), TrainingConfig())
    rows = [{"prompt": "What is 1 plus 1?"}, {"prompt": "What is 2 plus 2?"}]
    method.prepare(student, rows, Path("."), Progress())
    loss = method.loss(student, rows)
    loss.backward()
    assert method.metrics["reverse_kl"] == pytest.approx(0.0, abs=1e-5)
    assert all(float(p.grad.abs().max()) < 1e-6 for p in student.trainable() if p.grad is not None)
    method.finish(student, Progress())
    assert method.teacher is None
