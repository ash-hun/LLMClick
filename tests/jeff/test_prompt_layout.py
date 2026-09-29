import pytest

from jeff.model import PROMPT_LAYOUTS, decision_messages

CODES = ["A", "B", "C"]


def row(transcript: str, options: dict[str, str | None]) -> dict:
    return {"state": {"current_screen": "All deals", "previous_screen": None, "voice_transcript_may_contain_errors": transcript},
            "question": {"type": "choice", "instructions": "Which of these does the user want?", "criteria": options}}


def text(messages: list[dict]) -> str:
    return messages[1]["content"][-1]["text"]


def test_state_first_is_the_original_prompt() -> None:
    prompt = text(decision_messages(row("go home", {"1": "Home", "2": "Inbox"}), CODES))
    assert prompt == ('State:\n{"current_screen": "All deals", "previous_screen": null, "voice_transcript_may_contain_errors": "go home"}'
                      "\n\nQuestion:\nWhich of these does the user want?"
                      "\n\nOptions:\nA: 1: Home\nB: 2: Inbox"
                      "\n\nReturn only the letter code of the best option.")
    assert decision_messages(row("go home", {"1": "Home"}), CODES, "state-first") == decision_messages(row("go home", {"1": "Home"}), CODES)


def test_live_last_puts_the_question_first_and_the_last_state_field_after_the_options() -> None:
    prompt = text(decision_messages(row("go home", {"1": "Home", "2": "Inbox"}), CODES, "live-last"))
    assert prompt == ("Question:\nWhich of these does the user want?"
                      '\n\nState:\n{"current_screen": "All deals", "previous_screen": null}'
                      "\n\nOptions:\nA: 1: Home\nB: 2: Inbox"
                      '\n\nLatest:\n{"voice_transcript_may_contain_errors": "go home"}'
                      "\n\nReturn only the letter code of the best option.")


def test_live_last_requests_on_one_screen_share_everything_before_the_changing_options() -> None:
    first = text(decision_messages(row("go home", {"1": "Home", "2": "Inbox"}), CODES, "live-last"))
    second = text(decision_messages(row("open galley", {"1": "Home", "2": "Galley"}), CODES, "live-last"))
    shared = first[:next(i for i, (a, b) in enumerate(zip(first, second)) if a != b)]
    assert shared.endswith("A: 1: Home\nB: 2: ")


def test_live_last_needs_an_object_state() -> None:
    plain = {"state": "just text", "question": {"type": "choice", "criteria": {"1": "Home"}}}
    with pytest.raises(ValueError, match="live-last"):
        decision_messages(plain, CODES, "live-last")
    with pytest.raises(ValueError, match="live-last"):
        decision_messages({**plain, "state": {}}, CODES, "live-last")


def test_unknown_layout_is_an_error() -> None:
    assert PROMPT_LAYOUTS == ("state-first", "live-last")
    with pytest.raises(ValueError, match="layout"):
        decision_messages(row("go home", {"1": "Home"}), CODES, "question-first")
