import re

import pytest

from jeff import layout, puzzles
from jeff.data import validate
from jeff.families import BY_NAME

PEOPLE = "Ana, Ben, Cai, Dee, Eli"


def slot(family: str, label: str, index: int = 0, difficulty: str = "hard") -> dict:
    return {"id": f"syn-1-{family}-{index:06d}", "family": family, "label": label, "difficulty": difficulty, "people": PEOPLE}


def options(text: str) -> dict[str, str]:
    return dict(re.findall(r"(?m)^([A-Q]): (.+)$", text))


@pytest.mark.parametrize("index", range(40))
def test_tracking_five_answer_follows_the_swaps(index: int) -> None:
    label = "ABCDE"[index % 5]
    text = puzzles.tracking_five(slot("tracking_five", label, index))
    first, moves, question = text.split("\n")[:3]
    scene = next(s for s in puzzles.TRACKING_SCENES if s[0] in first)
    pattern = re.escape(scene[1]).replace(r"\{p\}", r"(\w+)").replace(r"\{x\}", r"(.+)")
    holding = dict(re.fullmatch(pattern, part).groups() for part in first.split(": ", 1)[1].rstrip(".").split(", "))
    swap = re.escape(scene[2]).replace(r"\{a\}", r"(\w+)").replace(r"\{b\}", r"(\w+)")
    for a, b in re.findall(swap, moves):
        holding[a], holding[b] = holding[b], holding[a]
    target = re.search(r"(\w+) (?:is|has)", question.removeprefix("Question: At the end of the ")).group(1)
    assert options(text)[label] == holding[target]


@pytest.mark.parametrize("index", range(34))
def test_colour_counting_count_matches_the_label(index: int) -> None:
    label = "ABCDEFGHIJKLMNOPQ"[index % 17]
    text = puzzles.colour_counting(slot("colour_counting", label, index))
    first, question = text.split("\n")[:2]
    colour = re.search(r"(?i)how many (\w+) objects", question).group(1)
    removed = re.search(r"remove all the (.+?) from", question)
    listing = re.split(r", and |, | and ", first.split(" there ", 1)[1].split(" ", 1)[1].rstrip("."))
    numbers = {name: i for i, name in enumerate(puzzles.NUMBER_NAMES)}
    total = 0
    for entry in listing:
        count, col, thing = entry.split(" ", 2)
        if col == colour and not (removed and thing == removed.group(1)) and not (removed and puzzles.plural(thing, 2) == removed.group(1)):
            total += numbers[count]
    assert total == index % 17, text
    assert list(options(text).values()) == list(puzzles.NUMBER_NAMES)


@pytest.mark.parametrize("index", range(30))
def test_navigate_turns_returns_home_only_when_labelled_true(index: int) -> None:
    label = ("true", "false")[index % 2]
    text = puzzles.navigate_turns(slot("navigate_turns", label, index))
    x = y = heading = 0
    for part in re.findall(r"Turn left\.|Turn right\.|Turn around\.|Take \d+ steps?\.", text):
        if part.startswith("Take"):
            steps = int(part.split()[1])
            dx, dy = ((0, 1), (-1, 0), (0, -1), (1, 0))[heading]
            x, y = x + dx * steps, y + dy * steps
        else:
            heading = (heading + puzzles.TURNS[part]) % 4
    assert ((x, y) == (0, 0)) == (label == "true")


def test_date_understanding_has_six_distinct_dates() -> None:
    for index in range(60):
        text = puzzles.date_understanding(slot("date_understanding", "ABCDEF"[index % 6], index))
        values = list(options(text).values())
        assert len(values) == 6 and len(set(values)) == 6 and all(re.fullmatch(r"\d\d/\d\d/\d{4}", v) for v in values)


def test_counting_options_keep_their_order_in_the_panel_layout() -> None:
    text = puzzles.colour_counting(slot("colour_counting", "D"))
    row = {"id": "x", "suite": "synthetic", "family": "f", "state": text, "question": BY_NAME["colour_counting"].question(),
           "label": "D", "target": "D", "source": {"family": "colour_counting"}}
    panel = layout.in_order(row, panel=True, target=0)
    assert list(panel["question"]["criteria"].values()) == list(puzzles.NUMBER_NAMES) and panel["label"] == "D"
    validate([panel])


@pytest.mark.parametrize("index", range(80))
def test_ordering_clues_are_true_and_pin_down_exactly_one_order(index: int) -> None:
    import itertools
    for n in (5, 7):
        text, order, tests, asked, _, _ = puzzles.ordering_puzzle(slot("ordering", "A", index), n)
        assert all(test(tuple(order)) for test in tests), text  # every clue holds for the true order
        solutions = [a for a in itertools.permutations(order) if all(test(a) for test in tests)]
        assert solutions == [tuple(order)], text  # and no other order satisfies them
        full = puzzles.ordering(slot("ordering", "ABCDEFG"[index % n], index), n)
        assert options(full)["ABCDEFG"[index % n]].split()[1] == order[asked].split()[0]
