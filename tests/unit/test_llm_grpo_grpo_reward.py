"""`llm_grpo` modeling/llm/methods/grpo_reward.py: the rule rewards, and a weighted sum of them."""

import pytest

from modeling.llm.methods.grpo_reward import (contains, exact_match, last_number_match, length_limit, regex_match,
                                              reward_names, weighted)


def test_answer_rewards() -> None:
    row = {"answer": "12"}
    assert exact_match(" 12 ", row, {}) == 1.0 and exact_match("12 .", row, {}) == 0.0
    assert contains("It is 12 .", row, {}) == 1.0 and contains("13", row, {}) == 0.0
    worked = {"answer": "She sells 16 - 7 = 9 eggs for 9 * 2 = $18.\n#### 1,018"}
    assert last_number_match("So 9 * 2 gives 1018.", worked, {}) == 1.0
    assert last_number_match("The answer is 1018 eggs, not 9", worked, {}) == 0.0
    assert last_number_match("no digits here", worked, {}) == 0.0
    with pytest.raises(ValueError, match="no number"):
        last_number_match("3", {"answer": "unknown"}, {})


def test_regex_reward() -> None:
    pattern = {"pattern": r"<think>.*</think>"}
    assert regex_match("<think>a\nb</think> 12", {}, pattern) == 1.0
    assert regex_match("<think>a\nb</think> 12", {}, {**pattern, "mode": "full"}) == 0.0
    with pytest.raises(ValueError, match="mode"):
        regex_match("x", {}, {**pattern, "mode": "begin"})
    with pytest.raises(ValueError, match="pattern"):
        regex_match("x", {}, {})


def test_length_reward_falls_from_the_limit_to_twice_it() -> None:
    assert [length_limit("a" * n, {}, {"max_chars": 10}) for n in (5, 10, 15, 20, 30)] == [1.0, 1.0, 0.5, 0.0, 0.0]
    with pytest.raises(ValueError, match="max_chars"):
        length_limit("a", {}, {})


def test_weighted_reward_sums_its_parts() -> None:
    params = {"parts": [{"name": "contains", "weight": 2}, {"name": "length", "max_chars": 4, "weight": 0.5}, {"name": "exact_match"}]}
    assert weighted("12", {"answer": "12"}, params) == 3.5
    assert reward_names("weighted", params) == ["weighted", "contains", "length", "exact_match"]
    assert reward_names("contains", {}) == ["contains"]


def test_weighted_reward_rejects_bad_parts() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        weighted("x", {}, {"parts": []})
    with pytest.raises(ValueError, match="another weighted"):
        reward_names("weighted", {"parts": [{"name": "weighted", "parts": [{"name": "contains"}]}]})
    with pytest.raises(ValueError, match="Unknown reward 'nope'"):
        reward_names("weighted", {"parts": [{"name": "nope"}]})
