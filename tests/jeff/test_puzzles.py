import re

import pytest

from jeff import puzzles
from jeff.families import make_slots


def slot(label: str, difficulty: str, index: int = 0) -> dict:
    base = [s for s in make_slots(400, seed=9) if s["family"] == "object_tracking"][index]
    return {**base, "label": label, "difficulty": difficulty, "people": "Ana Ruiz, Ben Okafor, Chen Wei, Dara Kaur, Eli Stone"}


def solve(text: str) -> str:
    """Replay the puzzle independently of the generator and return the correct option letter."""
    lines = text.split("\n")
    holding = dict(re.findall(r"(\w+ \w+) has the ([^.]+)\.", lines[1]))
    for line in lines[2:]:
        if line.startswith("Question:"):
            target = re.search(r"does (\w+ \w+) hold", line).group(1)
            break
        a, b = re.findall(r"(?:Ana Ruiz|Ben Okafor|Chen Wei|Dara Kaur|Eli Stone)", line)
        holding[a], holding[b] = holding[b], holding[a]
    options = {line[0]: line[3:].removeprefix("the ") for line in lines if re.match(r"^[A-D]: ", line)}
    return next(letter for letter, item in options.items() if item == holding[target])


@pytest.mark.parametrize("label", list("ABCD"))
@pytest.mark.parametrize("difficulty", ["easy", "hard"])
def test_correct_option_sits_at_the_required_letter(label: str, difficulty: str) -> None:
    for index in range(10):
        assert solve(puzzles.object_tracking(slot(label, difficulty, index))) == label


def test_never_states_holdings_after_a_swap() -> None:
    text = puzzles.object_tracking(slot("B", "hard"))
    assert text.count(" has the ") == len(re.findall(r" has the ", text.split("\n")[1]))


def test_needs_five_people() -> None:
    with pytest.raises(ValueError, match="five people"):
        puzzles.object_tracking({**slot("A", "easy"), "people": "Ana Ruiz, Ben Okafor"})


def date_slot(label: str, index: int) -> dict:
    base = [s for s in make_slots(4000, seed=9) if s["family"] == "date_arithmetic"][index]
    return {**base, "label": label, "people": "Ana Ruiz, Ben Okafor, Chen Wei, Dara Kaur, Eli Stone"}


def test_date_answers_are_right_and_sit_at_the_required_letter() -> None:
    import random
    from datetime import date, timedelta

    for index in range(200):
        rng = random.Random(index)
        text, question, answer, wrong = puzzles.date_problem(rng)
        assert answer not in wrong and len(set(wrong)) == 3
        when = text.split(" on ")[1].split(". ")[0] if " on " in text else None
        assert when is not None
        if "business days" in text:
            assert answer.weekday() < 5
        if "last day of the month" in text:
            assert (answer + timedelta(days=1)).day == 1
    for label in "ABCD":
        text = puzzles.date_arithmetic(date_slot(label, 3))
        options = {line[0]: line[3:] for line in text.split("\n") if re.match(r"^[A-D]: ", line)}
        assert len(options) == 4 and len(set(options.values())) == 4


def test_business_day_counting_skips_weekends_and_the_holiday() -> None:
    from datetime import date

    friday = date(2026, 9, 25)
    assert puzzles.add_business_days(friday, 1, None) == date(2026, 9, 28)  # Monday
    assert puzzles.add_business_days(friday, 1, date(2026, 9, 28)) == date(2026, 9, 29)  # Monday is a holiday
    assert puzzles.add_months_end(date(2026, 1, 31), 1) == date(2026, 2, 28)
    assert puzzles.add_months_end(date(2027, 12, 15), 2) == date(2028, 2, 29)  # leap year


def test_date_correct_option_is_at_the_required_letter() -> None:
    import random

    for index in range(20):
        for label in "ABCD":
            s = date_slot(label, index)
            _, _, answer, _ = puzzles.date_problem(random.Random(f"date-arithmetic-{s['id']}"))
            text = puzzles.date_arithmetic(s)
            line = next(l for l in text.split("\n") if l.startswith(f"{label}: "))
            assert line == f"{label}: {answer.strftime(puzzles.DATE_FORMAT)}"


def test_hard_puzzles_ask_about_someone_in_at_least_two_swaps() -> None:
    for index in range(10):
        text = puzzles.object_tracking(slot("A", "hard", index))
        target = re.search(r"does (\w+ \w+) hold", text).group(1)
        swap_lines = [l for l in text.split("\n")[2:] if not l.startswith(("Question", "A:", "B:", "C:", "D:"))]
        assert sum(target in line for line in swap_lines) >= 2


def family_slots(family: str, count: int = 40) -> list[dict]:
    from jeff.families import FAMILIES

    return [s for s in make_slots(len(FAMILIES) * count, seed=11) if s["family"] == family][:count]


def test_navigate_label_matches_replayed_position() -> None:
    for s in family_slots("navigate"):
        x = y = 0
        for n, direction in re.findall(r"Take (\d+) steps? (forward|backward|left|right)", puzzles.navigate(s)):
            dx, dy = puzzles.MOVES[direction]
            x, y = x + dx * int(n), y + dy * int(n)
        assert ((x, y) == (0, 0)) == (s["label"] == "true"), s["id"]


def test_boolean_expression_evaluates_to_label() -> None:
    for s in family_slots("boolean_expressions"):
        text = puzzles.boolean_expressions(s)
        assert text.endswith(" is") and set(re.findall(r"[A-Za-z]+", text)) <= {"True", "False", "not", "and", "or", "is"}
        assert eval(text[:-3], {"__builtins__": {}}) is (s["label"] == "true")


def test_adjective_order_correct_option_follows_the_standard_order() -> None:
    rank = {word: i for i, group in enumerate(puzzles.ADJECTIVES) for word in group}
    for s in family_slots("adjective_order"):
        lines = dict(line.split(": ", 1) for line in puzzles.adjective_order(s).split("\n")[1:])
        ordered = lambda phrase: [rank[w] for w in phrase.split()[:-1]] == sorted(rank[w] for w in phrase.split()[:-1])
        assert ordered(lines[s["label"]]) and not ordered(lines["B" if s["label"] == "A" else "A"]), s["id"]


def test_coloured_objects_answer_is_at_the_label() -> None:
    for s in family_slots("coloured_objects"):
        text = puzzles.coloured_objects(s)
        row = re.findall(r"a (\w+) ([a-z ]+?)(?:,| and|\.)", text.split("\n")[0].split("you see ", 1)[1])
        colours, things = [c for c, _ in row], [t.strip() for _, t in row]
        question = text.split("\n")[1]
        if m := re.search(r"colour of the (.+)\?$", question):
            target_text = m.group(1)
            if target_text.startswith("object directly to the left of the "):
                target = things.index(target_text.removeprefix("object directly to the left of the ")) - 1
            elif target_text.startswith("object directly to the right of the "):
                target = things.index(target_text.removeprefix("object directly to the right of the ")) + 1
            elif target_text.startswith("object furthest from the "):
                target = len(things) - 1 - things.index(target_text.removeprefix("object furthest from the "))
            else:
                target = things.index(target_text)
        options = {line[0]: line[3:] for line in text.split("\n") if re.match(r"^[A-D]: ", line)}
        assert len(set(options.values())) == 4
        assert options[s["label"]] == colours[target], (s["id"], text)
