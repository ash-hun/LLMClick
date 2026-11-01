"""Reward `judge`: a teacher model scores a completion against a rubric; registered from here so modeling never imports data."""

import json
from functools import lru_cache
from typing import Any

from data.teacher_client import Teacher, teacher_from
from modeling.llm.methods.grpo_reward import REWARDS
from modeling.tuning.training_method import Row
from modeling.modeling_config import Keyed
from data.verify_stage import JUDGE, SCORE

LOWEST, HIGHEST = 1.0, 10.0
CACHED_SCORES = 4096  # a GRPO run asks about each completion once per reuse; this keeps one run's recent groups


@lru_cache(maxsize=8)
def judge_teacher(spec: str) -> Teacher:
    """One teacher per distinct `teacher` entry for the life of the process, so a local judge loads once."""
    return teacher_from(Keyed(**json.loads(spec)))


@lru_cache(maxsize=CACHED_SCORES)
def judged(spec: str, rubric: str, prompt: str, completion: str, max_tokens: int) -> float:
    reply = judge_teacher(spec).complete([{"role": "user", "content": JUDGE.format(rubric=rubric, instruction=prompt, text=completion)}],
                                         max_tokens=max_tokens, temperature=0.0, seed=0)
    # the reply is untrusted model text: only a number inside the scale counts. One outside it (a completion that
    # talked the judge into "99") scores the bottom rather than being clamped to the top.
    match = SCORE.search(reply.text)
    score = float(match.group()) if match else LOWEST
    score = score if LOWEST <= score <= HIGHEST else LOWEST
    return (score - LOWEST) / (HIGHEST - LOWEST)


def prompt_text(row: Row) -> str:
    prompt = row["prompt"]
    return prompt if isinstance(prompt, str) else "\n\n".join(f"{turn['role']}: {turn['content']}" for turn in prompt)


@REWARDS.register("judge")
def judge(completion: str, row: Row, params: dict[str, Any]) -> float:
    """0 to 1 from the `teacher`'s 1-10 score of the completion under `rubric`. Same text, same score: replies are
    greedy and cached in memory. `check` scores an empty completion first, which makes one call per row."""
    teacher, rubric = params.get("teacher"), params.get("rubric")
    if not isinstance(teacher, dict) or "name" not in teacher or not isinstance(rubric, str) or not rubric.strip():
        raise ValueError("the judge reward needs `teacher` ({name, ...}) and a non-empty `rubric`")
    if not completion.strip():
        return 0.0  # nothing to judge, and the row check then costs no call
    return judged(json.dumps(teacher, sort_keys=True), rubric, prompt_text(row), completion, int(params.get("max_tokens", 8)))
