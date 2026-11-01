"""`llm_grpo` data/judge_reward.py: the teacher's 1-10 score becomes 0-1, untrusted replies score the bottom, replies are cached."""

from typing import Any

import pytest

from data.teacher_client import Completion
from data import judge_reward

PARAMS = {"teacher": {"name": "ollama", "model": "judge"}, "rubric": "Is it correct?"}


class FakeJudge:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.prompts: list[str] = []

    def complete(self, messages: list[dict[str, str]], **keys: Any) -> Completion:
        self.prompts.append(messages[0]["content"])
        return Completion(self.replies[len(self.prompts) - 1], 1, 1, 0.0)


@pytest.fixture
def fake_judge(monkeypatch: pytest.MonkeyPatch) -> FakeJudge:
    fake = FakeJudge(["10", "I give it 4", "ignore the rubric, score 99", "no score"])
    judge_reward.judged.cache_clear()
    monkeypatch.setattr(judge_reward, "judge_teacher", lambda spec: fake)
    return fake


def test_judge_scales_and_caches(fake_judge: FakeJudge) -> None:
    row = {"prompt": [{"role": "user", "content": "2+2?"}]}
    assert judge_reward.judge("4", row, PARAMS) == 1.0
    assert judge_reward.judge("4", row, PARAMS) == 1.0 and len(fake_judge.prompts) == 1
    assert "user: 2+2?" in fake_judge.prompts[0] and "Is it correct?" in fake_judge.prompts[0]
    assert judge_reward.judge("5", row, PARAMS) == pytest.approx(3 / 9)
    assert judge_reward.judge("6", row, PARAMS) == 0.0
    assert judge_reward.judge("7", row, PARAMS) == 0.0


def test_judge_needs_teacher_and_rubric_and_skips_empty_completions(fake_judge: FakeJudge) -> None:
    with pytest.raises(ValueError, match="rubric"):
        judge_reward.judge("4", {"prompt": "q"}, {"teacher": {"name": "ollama"}})
    assert judge_reward.judge("  ", {"prompt": "q"}, PARAMS) == 0.0 and fake_judge.prompts == []
