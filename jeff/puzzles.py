"""Object-tracking puzzles built in code, so the swaps and the answer are always correct."""

import calendar
import random
import re
from datetime import date, timedelta

from jeff.families import Slot

ITEM_SETS = {
    "ball": ["red", "blue", "green", "yellow", "purple", "orange", "white", "black", "pink", "brown"],
    "book": ["atlas", "cookbook", "field guide", "poetry collection", "mystery novel", "biography", "dictionary", "comic"],
    "gift": ["scarf", "watch", "candle", "mug", "wallet", "notebook", "plant", "puzzle"],
    "tool": ["hammer", "wrench", "tape measure", "screwdriver", "level", "drill", "saw", "pliers"],
    "ticket": ["opera", "football", "museum", "concert", "theatre", "ferry", "zoo", "cinema"],
}
NOUN = {"ball": "{} ball", "book": "{}", "gift": "{}", "tool": "{}", "ticket": "{} ticket"}
SWAPS = ("Then {a} and {b} swap {kind}s.", "Next, {a} trades {kind}s with {b}.", "After that, {a} and {b} exchange {kind}s.",
         "{a} and {b} then swap.", "Later, {b} swaps {kind}s with {a}.")
OPENINGS = ("{people} each start with a different {kind}.", "At the start, {people} each hold one {kind}.",
            "{people} are each given a {kind}.")
LETTERS = "ABCD"


def object_tracking(slot: Slot) -> str:
    """Write the puzzle for this slot so that the option lettered slot["label"] is correct."""
    rng = random.Random(f"object-tracking-{slot['id']}")
    if "people" not in slot:
        raise ValueError(f"Slot {slot['id']} has no people; dress slots with materials before generating")
    people = slot["people"].split(", ")
    if len(people) < 5:
        raise ValueError(f"Slot {slot['id']} needs five people for object tracking, got {len(people)}")
    count, swaps = (4, rng.randint(2, 3)) if slot["difficulty"] == "easy" else (5, rng.randint(4, 6))
    people = people[:count]
    kind = rng.choice(sorted(ITEM_SETS))
    items = [NOUN[kind].format(item) for item in rng.sample(ITEM_SETS[kind], count)]
    holding = dict(zip(people, items))
    names = ", ".join(people[:-1]) + f" and {people[-1]}"
    lines = [rng.choice(OPENINGS).format(people=names, kind=kind),
             " ".join(f"{person} has the {item}." for person, item in holding.items())]
    involved = {person: 0 for person in people}
    for _ in range(swaps):
        a, b = rng.sample(people, 2)
        holding[a], holding[b] = holding[b], holding[a]
        involved[a] += 1
        involved[b] += 1
        lines.append(rng.choice(SWAPS).format(a=a, b=b, kind=kind))
    # Hard puzzles ask about someone in at least two swaps, so the answer cannot be read off a single sentence.
    candidates = [p for p in people if involved[p] >= 2] if slot["difficulty"] == "hard" else people
    target = rng.choice(candidates or sorted(people, key=lambda p: -involved[p])[:1])
    answer = holding[target]
    wrong = rng.sample([item for item in items if item != answer], 3)
    position = LETTERS.index(slot["label"])
    options = wrong[:position] + [answer] + wrong[position:]
    lines.append(f"Question: At the end, which {kind} does {target} hold?")
    lines.extend(f"{letter}: the {option}" for letter, option in zip(LETTERS, options))
    return "\n".join(lines)


DOCUMENTS = ("notice", "invoice", "application", "complaint", "contract amendment", "claim form", "work order", "refund request")
DATE_FORMAT = "%A, %B %-d, %Y"


def add_business_days(start: date, count: int, holiday: date | None) -> date:
    """Count forward from the day after start, skipping Saturdays, Sundays and the holiday."""
    current = start
    while count:
        current += timedelta(days=1)
        if current.weekday() < 5 and current != holiday:
            count -= 1
    return current


def add_months_end(start: date, months: int) -> date:
    """The last day of the month that is `months` after start's month."""
    year, month = divmod(start.month - 1 + months, 12)
    return date(start.year + year, month + 1, calendar.monthrange(start.year + year, month + 1)[1])


def date_problem(rng: random.Random) -> tuple[str, str, date, list[date]]:
    """Scenario text (with {person} and {document} to fill), the question, the correct date, and three dates made with
    the usual mistakes (off by one, ignoring weekends or holidays, assuming 30-day months)."""
    start = date(2021, 1, 1) + timedelta(days=rng.randrange(0, 365 * 6))
    when = start.strftime(DATE_FORMAT)
    kind = rng.choice(("calendar_after", "calendar_before", "business", "business_holiday", "month_end"))
    if kind == "calendar_after":
        days = rng.randint(10, 75)
        text = f"{{person}} received the {{document}} on {when}. A written reply is due {days} calendar days after that."
        question, answer = "When is the reply due?", start + timedelta(days=days)
        mistakes = [answer - timedelta(days=1), answer + timedelta(days=1), start + timedelta(days=days + 1 + days // 30)]
    elif kind == "calendar_before":
        days = rng.randint(10, 75)
        text = (f"{{person}} has a hearing about the {{document}} on {when}. Supporting documents must be filed "
                f"{days} calendar days before the hearing.")
        question, answer = "What is the filing deadline?", start - timedelta(days=days)
        mistakes = [answer + timedelta(days=1), answer - timedelta(days=1), start - timedelta(days=days + 2)]
    elif kind in ("business", "business_holiday"):
        days = rng.randint(3, 12)
        holiday = add_business_days(start, rng.randint(1, days), None) if kind == "business_holiday" else None
        text = (f"{{person}} received the {{document}} on {when}. It must be answered within {days} business days, "
                "not counting the day it arrived. Saturdays and Sundays are not business days.")
        if holiday is not None:
            text += f" {holiday.strftime(DATE_FORMAT)} is a public holiday and is not a business day either."
        question, answer = "What is the last day to answer?", add_business_days(start, days, holiday)
        mistakes = [add_business_days(start, days, None) if holiday else start + timedelta(days=days),
                    add_business_days(start, days + 1, holiday), add_business_days(start, days - 1, holiday)]
    else:
        months = rng.randint(1, 5)
        text = (f"{{person}} signed the {{document}} on {when}. The final payment is due on the last day of the month "
                f"that is {months} month{'s' if months > 1 else ''} after the signing month.")
        question, answer = "When is the final payment due?", add_months_end(start, months)
        mistakes = [answer - timedelta(days=1), answer.replace(day=30) if answer.day == 31 else answer + timedelta(days=1),
                    add_months_end(start, months + 1)]
    distinct = list(dict.fromkeys(d for d in mistakes if d != answer))
    while len(distinct) < 3:
        extra = answer + timedelta(days=rng.choice((-3, -2, 2, 3)))
        if extra not in distinct:
            distinct.append(extra)
    return text, question, answer, distinct[:3]


def date_arithmetic(slot: Slot) -> str:
    """Write the date puzzle for this slot so that the option lettered slot["label"] is correct."""
    rng = random.Random(f"date-arithmetic-{slot['id']}")
    if "people" not in slot:
        raise ValueError(f"Slot {slot['id']} has no people; dress slots with materials before generating")
    text, question, answer, wrong = date_problem(rng)
    position = LETTERS.index(slot["label"])
    options = wrong[:position] + [answer] + wrong[position:]
    lines = [text.format(person=rng.choice(slot["people"].split(", ")), document=rng.choice(DOCUMENTS)), f"Question: {question}"]
    lines.extend(f"{letter}: {option.strftime(DATE_FORMAT)}" for letter, option in zip(LETTERS, options))
    return "\n".join(lines)


# ---- BBH-style reasoning families, built entirely in code so every label is exact. ----

MOVES = {"forward": (0, 1), "backward": (0, -1), "left": (-1, 0), "right": (1, 0)}


def navigate(slot: Slot) -> str:
    """Movement instructions; the slot label says whether they end at the starting point ("true") or not ("false")."""
    rng = random.Random(f"navigate-{slot['id']}")
    count = rng.randint(3, 5) if slot["difficulty"] == "easy" else rng.randint(6, 9)
    moves = [(rng.choice(sorted(MOVES)), rng.randint(1, 10)) for _ in range(count)]
    x = sum(MOVES[d][0] * n for d, n in moves)
    y = sum(MOVES[d][1] * n for d, n in moves)
    if slot["label"] == "true":
        if x:
            moves.append(("left" if x > 0 else "right", abs(x)))
        if y:
            moves.append(("backward" if y > 0 else "forward", abs(y)))
        rng.shuffle(moves)
    elif x == 0 and y == 0:
        moves.append((rng.choice(sorted(MOVES)), rng.randint(1, 10)))
    steps = " ".join(f"Take {n} step{'s' if n > 1 else ''} {direction}." for direction, n in moves)
    return f"Always face forward. {steps}"


def boolean_expression(rng: random.Random, depth: int) -> str:
    if depth == 0:
        return rng.choice(("True", "False"))
    kind = rng.choice(("not", "and", "or", "or", "and"))
    if kind == "not":
        return f"not {boolean_expression(rng, depth - 1)}" if rng.random() < .5 else f"not ( {boolean_expression(rng, depth - 1)} )"
    left, right = boolean_expression(rng, depth - 1), boolean_expression(rng, rng.randint(0, depth - 1))
    return f"( {left} {kind} {right} )" if rng.random() < .5 else f"{left} {kind} {right}"


def boolean_expressions(slot: Slot) -> str:
    """A Python-style boolean expression whose value matches the slot label."""
    rng = random.Random(f"boolean-{slot['id']}")
    depth = 2 if slot["difficulty"] == "easy" else 3
    want = slot["label"] == "true"
    for _ in range(1000):
        expression = boolean_expression(rng, depth)
        if eval(expression, {"__builtins__": {}}) is want:  # only True/False/not/and/or/parentheses are generated
            return f"{expression} is"
    raise ValueError(f"Slot {slot['id']}: no expression with value {want} found")


# Standard English order: opinion, size, age, shape, colour, origin, material, purpose.
ADJECTIVES = (("lovely", "ugly", "wonderful", "awful", "nice", "terrible", "silly", "obnoxious"),
              ("big", "small", "tiny", "enormous", "large", "little", "massive", "huge"),
              ("old", "new", "ancient", "brand-new", "old-fashioned", "young"),
              ("square", "circular", "triangular", "rectangular", "spherical", "oval"),
              ("red", "blue", "green", "grey", "purple", "black", "white", "brown"),
              ("Japanese", "Brazilian", "Egyptian", "Mexican", "German", "Indian", "Turkish", "Russian"),
              ("wooden", "plastic", "steel", "paper", "glass", "leather", "wool", "cardboard"),
              ("drinking", "hiking", "walking", "smoking", "exercise", "eating", "snorkeling", "whittling"))
NOUNS = ("knife", "shoe", "table", "cat", "ship", "surfboard", "bag", "sweater", "car", "box", "chair", "lamp")


def adjective_order(slot: Slot) -> str:
    """Two noun phrases; the one at the slot's letter has correct adjective order, the other the same words shuffled."""
    rng = random.Random(f"adjectives-{slot['id']}")
    count = rng.randint(2, 3) if slot["difficulty"] == "easy" else rng.randint(4, 5)
    categories = sorted(rng.sample(range(len(ADJECTIVES)), count))
    words = [rng.choice(ADJECTIVES[c]) for c in categories]
    noun = rng.choice(NOUNS)
    wrong = words[:]
    while wrong == words:
        rng.shuffle(wrong)
    right_text, wrong_text = f"{' '.join(words)} {noun}", f"{' '.join(wrong)} {noun}"
    first, second = (right_text, wrong_text) if slot["label"] == "A" else (wrong_text, right_text)
    return f"Which sentence has the correct adjective order?\nA: {first}\nB: {second}"


COLOURS = ("red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "grey", "silver", "gold", "teal", "burgundy")
OBJECTS = ("pencil", "mug", "notebook", "stapler", "keychain", "sheet of paper", "fidget spinner", "paperclip", "envelope",
           "jug", "booklet", "cup", "plate", "necklace", "crayon", "scrunchie", "sunglasses", "puzzle")


def coloured_objects(slot: Slot) -> str:
    """Coloured objects in a row, then a question about colour or position; the right colour sits at the slot's letter."""
    rng = random.Random(f"colours-{slot['id']}")
    count = rng.randint(3, 4) if slot["difficulty"] == "easy" else rng.randint(5, 7)
    things = rng.sample(OBJECTS, count)
    colours = rng.sample(COLOURS, count)
    row = ", ".join(f"a {c} {t}" for c, t in zip(colours, things))
    place = rng.choice(("desk", "table", "floor", "nightstand"))
    kind = rng.choice(("colour", "left", "right", "furthest")) if slot["difficulty"] == "hard" else "colour"
    if kind == "colour":
        target = rng.randrange(count)
        question = f"What is the colour of the {things[target]}?"
    elif kind == "left":
        anchor = rng.randrange(1, count)
        target = anchor - 1
        question = f"What is the colour of the object directly to the left of the {things[anchor]}?"
    elif kind == "right":
        anchor = rng.randrange(0, count - 1)
        target = anchor + 1
        question = f"What is the colour of the object directly to the right of the {things[anchor]}?"
    else:
        anchor = rng.choice((0, count - 1))
        target = count - 1 - anchor
        question = f"What is the colour of the object furthest from the {things[anchor]}?"
    answer = colours[target]
    others = [c for c in colours if c != answer] + [c for c in COLOURS if c not in colours]
    wrong = rng.sample(others[:max(3, len(colours) - 1)], 3)
    position = LETTERS.index(slot["label"])
    options = wrong[:position] + [answer] + wrong[position:]
    lines = [f"On the {place}, arranged in a row from left to right, you see {row}.", f"Question: {question}"]
    lines.extend(f"{letter}: {option}" for letter, option in zip(LETTERS, options))
    return "\n".join(lines)


# ---- Table questions in the shape of BBH's penguin task, with our own invented animals and values. ----

PENGUIN_NAMES = ("Otto", "Pia", "Rudi", "Mika", "Nell", "Bruno", "Ines", "Tomas", "Wren", "Yara", "Kasper", "Lotte", "Emil",
                 "Frida", "Hugo", "Jonas", "Maja", "Nils", "Oskar", "Runa", "Sven", "Tilda", "Ulla", "Vera")
FIELDS = ("age", "height", "weight")
UNITS = {"age": "", "height": " cm", "weight": " kg"}
TABLE_INTRO = "Here is a table where the first line is a header and each subsequent line is a penguin:  name, age, height (cm), weight (kg)"
NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight")


def penguin_rows(rng: random.Random, count: int) -> list[dict[str, object]]:
    """Distinct values per column, so "oldest" or "heaviest" always has one answer."""
    names = rng.sample(PENGUIN_NAMES, count)
    ages, heights, weights = rng.sample(range(2, 16), count), rng.sample(range(40, 121, 5), count), rng.sample(range(6, 31), count)
    return [{"name": n, "age": a, "height": h, "weight": w} for n, a, h, w in zip(names, ages, heights, weights)]


def penguin_question(rng: random.Random, table: list[dict[str, object]], hard: bool) -> tuple[str, str, list[str]]:
    """A question, its answer and the pool of plausible wrong answers."""
    names = [str(row["name"]) for row in table]
    kind = rng.choice(("extreme", "count", "sorted", "lookup") if hard else ("extreme", "lookup"))
    field = rng.choice(FIELDS)
    if kind == "extreme":
        words = {"age": ("oldest", "youngest"), "height": ("tallest", "shortest"), "weight": ("heaviest", "lightest")}[field]
        highest = rng.random() < .5
        pick = (max if highest else min)(table, key=lambda row: row[field])
        return f"Which is the {words[0] if highest else words[1]} penguin", str(pick["name"]), [n for n in names if n != pick["name"]]
    if kind == "count":
        threshold = rng.choice(sorted({int(row[field]) for row in table}))  # type: ignore[call-overload]
        count = sum(int(row[field]) > threshold for row in table)  # type: ignore[call-overload]
        noun = {"age": "years old", "height": "cm tall", "weight": "kg"}[field]
        question = f"How many penguins are more than {threshold} {noun}"
        return question, NUMBER_WORDS[count], [w for w in NUMBER_WORDS[:len(table) + 2] if w != NUMBER_WORDS[count]]
    if kind == "sorted":
        position = rng.randrange(len(table))
        ordinal = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh")[position]
        answer = sorted(names)[position]
        return f"If we sort the penguins by name, which one is {ordinal}", answer, [n for n in names if n != answer]
    row = rng.choice(table)
    answer = f"{row[field]}{UNITS[field]}"
    others = sorted({f"{other[field]}{UNITS[field]}" for other in table if other is not row})
    return f"What is the {field} of {row['name']}", answer, others


def penguins_table(slot: Slot) -> str:
    """A penguin table, sometimes changed by adding or removing one penguin, then one question with five options."""
    rng = random.Random(f"penguins-{slot['id']}")
    table = penguin_rows(rng, rng.randint(4, 6))
    lines = [TABLE_INTRO + " " + " ".join(f"{r['name']}, {r['age']}, {r['height']}, {r['weight']}" for r in table)]
    example = rng.sample(table, 2)
    lines.append(f"For example: the age of {example[0]['name']} is {example[0]['age']}, "
                 f"the weight of {example[1]['name']} is {example[1]['weight']} kg.")
    if slot["difficulty"] == "hard" and rng.random() < .5:
        if rng.random() < .5:
            extra = penguin_rows(rng, len(table) + 1)[-1]
            while extra["name"] in {r["name"] for r in table} or any(extra[f] == r[f] for r in table for f in FIELDS):
                extra = penguin_rows(rng, 1)[0]
            table.append(extra)
            lines.append(f"We now add a penguin to the table: {extra['name']}, {extra['age']}, {extra['height']}, {extra['weight']}")
        else:
            removed = table.pop(rng.randrange(len(table)))
            lines.append(f"We now delete the penguin named {removed['name']} from the table.")
    question, answer, wrong = penguin_question(rng, table, slot["difficulty"] == "hard")
    distinct = list(dict.fromkeys(w for w in wrong if w != answer))
    # Small tables give too few wrong answers: add names not in the table, or nearby values in the same unit.
    if answer in PENGUIN_NAMES:
        spare = [n for n in PENGUIN_NAMES if n != answer and n not in distinct]
    else:
        value, unit = re.fullmatch(r"(\d+)(.*)", answer).groups() if re.fullmatch(r"\d+.*", answer) else (None, "")  # type: ignore[union-attr]
        spare = [f"{int(value) + d}{unit}" for d in (1, -1, 2, -2, 5, -5)] if value else []
    distinct += [w for w in rng.sample(spare, len(spare)) if w not in distinct and w != answer][:max(0, 4 - len(distinct))]
    options = rng.sample(distinct, 4)
    position = "ABCDE".index(slot["label"])
    options = options[:position] + [answer] + options[position:]
    lines.append(f"Question: {question}?")
    lines.extend(f"{letter}: {option}" for letter, option in zip("ABCDE", options))
    return "\n".join(lines)


# ---- Schedules in the shape of BBH's temporal-sequence task: exactly one free window fits. ----

PLACES = ("bakery", "library", "post office", "swimming pool", "art gallery", "bike shop", "pharmacy", "museum", "market",
          "gym", "bookstore", "dentist", "train station", "florist", "cinema", "park")
ACTIVITIES = ("buying groceries at the supermarket", "walking in the botanical garden", "working at the office",
              "reading at the café", "waiting at the airport", "sitting in a meeting at the town hall",
              "jogging along the river", "fixing their car at the garage", "playing tennis at the club",
              "taking photos at the harbour", "attending a lecture at the university", "eating lunch at the diner")


def clock(hour: int) -> str:
    return "12pm" if hour == 12 else f"{hour}am" if hour < 12 else f"{hour - 12}pm"


def temporal_sequences(slot: Slot) -> str:
    """A day cut into sightings; the one unseen window before closing time is the answer."""
    rng = random.Random(f"temporal-{slot['id']}")
    if "people" not in slot:
        raise ValueError(f"Slot {slot['id']}: no people; dress slots with materials before generating")
    person, *witnesses = slot["people"].split(", ")
    count = rng.randint(3, 4) if slot["difficulty"] == "easy" else rng.randint(5, 6)
    wake = rng.randint(5, 8)
    cuts = sorted(rng.sample(range(wake + 1, 21), count))  # boundaries of the windows between waking and closing
    windows = list(zip([wake] + cuts[:-1], cuts))
    free = rng.randrange(len(windows))
    close = cuts[-1]
    place = rng.choice(PLACES)
    activities = rng.sample(ACTIVITIES, len(windows))
    lines = [f"Today, {person} went to the {place}. Between what times could they have gone?", "We know that:",
             f"{person} woke up at {clock(wake)}."]
    for index, (start, end) in enumerate(windows):
        if index != free:
            lines.append(f"{rng.choice(witnesses)} saw {person} {activities[index]} from {clock(start)} to {clock(end)}.")
    lines.append(f"The {place} was closed after {clock(close)}.")
    lines.append(f"Question: Between what times could {person} have gone to the {place}?")
    answer = f"{clock(windows[free][0])} to {clock(windows[free][1])}"
    wrong = [f"{clock(a)} to {clock(b)}" for index, (a, b) in enumerate(windows) if index != free]
    wrong = rng.sample(wrong, min(3, len(wrong)))
    while len(wrong) < 3:  # easy days can have too few sightings; add an unseen window after closing time
        extra = f"{clock(close + len(wrong))} to {clock(close + len(wrong) + 1)}"
        wrong.append(extra)
    position = LETTERS.index(slot["label"])
    options = wrong[:position] + [answer] + wrong[position:]
    lines.extend(f"{letter}: {option}" for letter, option in zip(LETTERS, options))
    return "\n".join(lines)


# ---- BBH-shaped families, matching the sub-tasks' own structure (object counts, option counts, wording variants). ----

LETTERS_17 = "ABCDEFGHIJKLMNOPQ"
TRACKING_SCENES = (
    ("are dancers at a square dance. At the start of a song, they each have a partner:", "{p} is dancing with {x}",
     "{a} and {b} switch partners", "At the end of the dance, {p} is dancing with",
     ("Patrick", "Sam", "Jamie", "Lola", "Melissa", "Rodrigo", "Karl", "Ophelia", "Izzi", "Helga", "Lionel", "Frank")),
    ("are playing a game of soccer. At the start of the match, they are each assigned to a position:", "{p} is playing {x}",
     "{a} and {b} trade positions", "At the end of the match, {p} is playing",
     ("goalkeeper", "left winger", "right winger", "striker", "center midfielder", "benchwarmer", "cheerleader",
      "fullback", "left midfielder", "right midfielder")),
    ("are playing a game. At the start of the game, they are each holding a ball:", "{p} has a {x} ball",
     "{a} and {b} swap balls", "At the end of the game, {p} has the",
     ("red", "orange", "yellow", "green", "blue", "purple", "pink", "white", "black", "brown")),
    ("are friends and avid readers who occasionally trade books. At the start of the semester, they each buy one new book:",
     "{p} gets {x}", "{a} and {b} swap books", "At the end of the semester, {p} has",
     ("Frankenstein", "Ulysses", "Moby Dick", "Hound of the Baskervilles", "The Odyssey", "Lolita", "Catch-22",
      "The Great Gatsby", "The Pearl", "Hamlet")),
    ("are holding a white elephant gift exchange. At the start of the event, they are each holding a present of a different color:",
     "{p} has a {x} present", "{a} and {b} swap their gifts", "At the end of the event, {p} has the",
     ("red", "orange", "yellow", "green", "blue", "purple", "pink", "white", "black", "brown")),
)


def tracking_five(slot: Slot) -> str:
    """Five people, a starting assignment, five pairwise swaps, and the question about one person's final item."""
    rng = random.Random(f"tracking-five-{slot['id']}")
    if "people" not in slot:
        raise ValueError(f"Slot {slot['id']}: no people; dress slots with materials before generating")
    people = slot["people"].split(", ")[:5]
    if len(people) < 5:
        raise ValueError(f"Slot {slot['id']} needs five people, got {len(people)}")
    intro, holds, swap, ending, pool = rng.choice(TRACKING_SCENES)
    items = rng.sample(pool, 5)
    holding = dict(zip(people, items))
    names = ", ".join(people[:-1]) + f", and {people[-1]}"
    swaps = [rng.sample(people, 2) for _ in range(5 if slot["difficulty"] == "hard" else 3)]
    words = ["First"] + ["Then"] * (len(swaps) - 2) + ["Finally"]
    steps = []
    for word, (a, b) in zip(words, swaps):
        holding[a], holding[b] = holding[b], holding[a]
        steps.append(f"{word}, {swap.format(a=a, b=b)}.")
    target = rng.choice(people)
    answer = holding[target]
    others = [item for item in items if item != answer]
    rng.shuffle(others)
    position = "ABCDE".index(slot["label"])
    options = others[:position] + [answer] + others[position:]
    start = ", ".join(holds.format(p=p, x=x) for p, x in zip(people, items))
    lines = [f"{names} {intro} {start}.", "Throughout the event, pairs of them trade. " + " ".join(steps),
             f"Question: {ending.format(p=target)}"]
    lines.extend(f"{letter}: {option}" for letter, option in zip("ABCDE", options))
    return "\n".join(lines)


def mmddyyyy(day: date) -> str:
    return day.strftime("%m/%d/%Y")


def date_understanding(slot: Slot) -> str:
    """A "today is ..." fact (sometimes indirect), then a relative date asked for in MM/DD/YYYY, with six options."""
    rng = random.Random(f"date-understanding-{slot['id']}")
    today = date(1930, 1, 1) + timedelta(days=rng.randrange(0, 365 * 95))
    kind = rng.choice(("direct", "yesterday", "tomorrow_was", "delayed", "anniversary") if slot["difficulty"] == "hard" else ("direct", "yesterday"))
    if kind == "direct":
        fact = f"Today is {mmddyyyy(today)}."
    elif kind == "yesterday":
        fact = f"Yesterday was {mmddyyyy(today - timedelta(days=1))}."
    elif kind == "tomorrow_was":
        fact = f"The day before yesterday was {mmddyyyy(today - timedelta(days=2))}."
    elif kind == "delayed":
        delay = rng.randint(1, 5)
        fact = (f"The concert was scheduled to be on {mmddyyyy(today - timedelta(days=delay))}, but was delayed by "
                f"{delay} day{'s' if delay > 1 else ''} to today.")
    else:
        years = rng.randint(2, 10)
        try:
            past = today.replace(year=today.year - years)
        except ValueError:  # 29 February in a non-leap year
            past = today.replace(year=today.year - years, day=28)
            today = past.replace(year=today.year)
        fact = f"Jane and John married on {mmddyyyy(past)}. Today is their {years}-year anniversary."
    shift = rng.choice((("tomorrow", 1), ("yesterday", -1), ("one week ago", -7), ("one week from today", 7),
                        ("10 days ago", -10), ("24 hours later", 1), ("one year ago", "year")))
    label, amount = shift
    if amount == "year":
        try:
            answer = today.replace(year=today.year - 1)
        except ValueError:
            answer = today.replace(year=today.year - 1, day=28)
    else:
        answer = today + timedelta(days=amount)  # type: ignore[arg-type]
    wrong_pool = {answer + timedelta(days=d) for d in (-1, 1, -7, 7, 30, -30)}
    for years in (-1, 1):
        try:
            wrong_pool.add(answer.replace(year=answer.year + years))
        except ValueError:
            pass
    if answer.day <= 12 and answer.day != answer.month:  # the classic month/day swap
        wrong_pool.add(date(answer.year, answer.day, answer.month))
    wrong = rng.sample(sorted(w for w in wrong_pool if w != answer), 5)
    position = "ABCDEF".index(slot["label"])
    options = [mmddyyyy(w) for w in wrong[:position]] + [mmddyyyy(answer)] + [mmddyyyy(w) for w in wrong[position:]]
    lines = [fact, f"Question: What is the date {label} in MM/DD/YYYY?"]
    lines.extend(f"{letter}: {option}" for letter, option in zip("ABCDEF", options))
    return "\n".join(lines)


ORDER_SETS = (  # (setting, noun, objects, left word, right word)
    ("On a branch, there are {n} birds: {items}.", "birds", ("owl", "raven", "falcon", "robin", "hawk", "crow", "blue jay",
     "cardinal", "hummingbird", "quail"), "left", "right"),
    ("On a shelf, there are {n} books: {items}.", "books", ("red book", "blue book", "green book", "white book",
     "orange book", "purple book", "black book", "brown book", "gray book", "yellow book"), "left", "right"),
    ("In an antique car show, there are {n} vehicles: {items}.", "vehicles", ("sedan", "convertible", "tractor",
     "minivan", "bus", "motorcyle", "truck", "limousine", "station wagon", "hatchback"), "newest", "oldest"),
    ("A fruit stand sells {n} fruits: {items}.", "fruits", ("apples", "peaches", "mangoes", "kiwis", "plums",
     "watermelons", "loquats", "pears", "oranges", "cantaloupes"), "cheapest", "most expensive"),
)
ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh")


def order_clue(order: list[str], kind: str, rng: random.Random, left: str, right: str, verb: str) -> tuple[str, object]:
    """A clue that is true of `order` (index 0 = the left / newest / cheapest end), and a test of it on any arrangement."""
    n = len(order)
    spatial = left in ("left", "right")
    if kind == "end":
        end = rng.choice((0, n - 1))
        obj = order[end]
        word = (f"the {left}most" if end == 0 else f"the {right}most") if spatial else f"the {left if end == 0 else right}"
        return f"The {obj} {verb} {word}.", lambda a, o=obj, e=end: a[e] == o
    if kind == "nth":
        index = rng.randrange(n)
        obj = order[index]
        place = f"the {ORDINALS[index]} from the {left}" if spatial else f"the {ORDINALS[index]}-{left}"
        return f"The {obj} {verb} {place}.", lambda a, o=obj, i=index: a[i] == o
    first, second = sorted(rng.sample(order, 2), key=order.index)  # `first` really is nearer the start
    relation = f"to the {left} of" if spatial else ("newer than" if left == "newest" else "cheaper than")
    return f"The {first} {verb} {relation} the {second}.", lambda a, x=first, y=second: a.index(x) < a.index(y)


def ordering_puzzle(slot: Slot, n: int) -> tuple[str, list[str], list[object], int, str, str]:
    """The puzzle text, the true order, the clue tests, the asked position, the ordering word ("left", "newest",
    "cheapest") and the verb ("is"/"are"); ordering() adds the question and options."""
    import itertools
    rng = random.Random(f"ordering-{n}-{slot['id']}")
    setting, noun, pool, left, right = rng.choice(ORDER_SETS)
    verb = "are" if noun == "fruits" else "is"
    objects = rng.sample(pool, n)
    order = objects[:]
    rng.shuffle(order)
    clues: list[tuple[str, object]] = []
    arrangements = list(itertools.permutations(objects))
    while len(arrangements) > 1:
        text, test = order_clue(order, rng.choice(("end", "nth", "relative", "relative")), rng, left, right, verb)
        narrowed = [a for a in arrangements if test(a)]  # type: ignore[operator]
        if len(narrowed) < len(arrangements) and text not in [c for c, _ in clues]:
            clues.append((text, test))
            arrangements = narrowed
    rng.shuffle(clues)

    def named(thing: str) -> str:
        return thing if noun == "fruits" else f"{'an' if thing[0] in 'aeiou' else 'a'} {thing}"
    items = ", ".join(named(o) for o in objects[:-1]) + f", and {named(objects[-1])}"
    count = ("five", "seven")[n == 7]
    text = (f"The following paragraphs each describe a set of {count} objects arranged in a fixed order. The statements are "
            f"logically consistent within each paragraph. {setting.format(n=count, items=items)} " + " ".join(t for t, _ in clues))
    return text, order, [test for _, test in clues], rng.randrange(n), left, verb


def ordering(slot: Slot, n: int) -> str:
    """BBH's logical deduction: n objects in a fixed order, clues with exactly one consistent order, one option per object."""
    text, order, _, index, left, verb = ordering_puzzle(slot, n)
    rng = random.Random(f"ordering-options-{n}-{slot['id']}")
    ask = f"the {ORDINALS[index]} from the {left}" if left in ("left", "right") else f"the {ORDINALS[index]}-{left}"
    answer = order[index]
    others = [o for o in order if o != answer]
    rng.shuffle(others)
    letters = "ABCDEFG"[:n]
    position = letters.index(slot["label"])
    options = others[:position] + [answer] + others[position:]
    lines = [text, f"Question: Which of these is {ask}?"]
    lines.extend(f"{letter}: The {option} {verb} {ask}" for letter, option in zip(letters, options))
    return "\n".join(lines)


def ordering_five(slot: Slot) -> str:
    return ordering(slot, 5)


def ordering_seven(slot: Slot) -> str:
    return ordering(slot, 7)


TURNS = {"Turn left.": 1, "Turn right.": -1, "Turn around.": 2}


def navigate_turns(slot: Slot) -> str:
    """BBH's other navigate variant: steps in the direction you face, with turns in between. For a "true" slot, a
    random walk is followed by the turns and steps that lead straight back to the start."""
    rng = random.Random(f"navigate-turns-{slot['id']}")
    count = rng.randint(3, 5) if slot["difficulty"] == "easy" else rng.randint(6, 9)
    headings = ((0, 1), (-1, 0), (0, -1), (1, 0))  # north, west, south, east: each "Turn left." moves one place on
    x = y = heading = 0
    parts: list[str] = []

    def turn_to(target: int) -> None:
        nonlocal heading
        turn = {1: "Turn left.", 2: "Turn around.", 3: "Turn right."}.get((target - heading) % 4)
        if turn:
            parts.append(turn)
        heading = target

    def walk(steps: int) -> None:
        nonlocal x, y
        dx, dy = headings[heading]
        x, y = x + dx * steps, y + dy * steps
        parts.append(f"Take {steps} step{'s' if steps > 1 else ''}.")

    for _ in range(count):
        if parts and rng.random() < 0.4:
            turn_to((heading + rng.choice((1, 2, 3))) % 4)
        else:
            walk(rng.randint(1, 10))
    if slot["label"] == "true":
        if x:
            turn_to(3 if x < 0 else 1)
            walk(abs(x))
        if y:
            turn_to(0 if y < 0 else 2)
            walk(abs(y))
    elif x == 0 and y == 0:
        walk(rng.randint(1, 10))
    return " ".join(parts)


COUNT_COLOURS = ("red", "orange", "yellow", "green", "blue", "purple", "pink", "grey", "black", "gold", "silver",
                 "teal", "mauve", "magenta", "fuchsia", "burgundy", "turquoise")
COUNT_THINGS = ("cat toy", "notebook", "keychain", "pen", "sheet of paper", "stress ball", "puzzle", "dog leash",
                "booklet", "mug", "sunglasses", "fidget spinner", "bracelet", "envelope", "paperclip", "phone charger")
NUMBER_NAMES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                "twelve", "thirteen", "fourteen", "fifteen", "sixteen")


def plural(thing: str, count: int) -> str:
    if count == 1:
        return thing
    if thing.endswith(("ss", "sh")):
        return thing + "es"
    if thing == "sheet of paper":
        return "sheets of paper"
    return thing if thing.endswith("sunglasses") else thing + "s"


def colour_counting(slot: Slot) -> str:
    """BBH's counting variant of coloured objects: a collection, an optional removal, and 'how many ... remain?' with the
    seventeen options zero to sixteen. The collection is built so that the count equals the slot's letter."""
    rng = random.Random(f"colour-count-{slot['id']}")
    want = LETTERS_17.index(slot["label"])
    kept_thing, removed_thing = rng.sample(COUNT_THINGS, 2)
    colour, *other_colours = rng.sample(COUNT_COLOURS, 3)
    removal = rng.random() < 0.6
    merged: dict[tuple[str, str], int] = {}
    left = want
    while left:  # the counted objects: the asked colour, not removed, in groups of one to four
        part = min(left, rng.randint(1, 4))
        thing = kept_thing if removal else rng.choice((kept_thing, removed_thing))
        merged[(colour, thing)] = merged.get((colour, thing), 0) + part
        left -= part
    for _ in range(rng.randint(2, 4)):  # distractors: other colours, and the asked colour on things that get removed
        target = (rng.choice(other_colours), rng.choice((kept_thing, removed_thing)))
        if removal and rng.random() < 0.5:
            target = (colour, removed_thing)
        elif not removal and target[0] == colour:
            continue
        merged[target] = merged.get(target, 0) + rng.randint(1, 4)
    if sum(merged.values()) > 40 or not merged:
        raise ValueError(f"Slot {slot['id']}: collection too large")
    entries = list(merged.items())
    rng.shuffle(entries)
    listing = [f"{NUMBER_NAMES[c] if c < len(NUMBER_NAMES) else c} {col} {plural(t, c)}" for (col, t), c in entries]
    text = listing[0] if len(listing) == 1 else ", ".join(listing[:-1]) + f", and {listing[-1]}"
    place = rng.choice(("floor", "desk", "table", "nightstand"))
    question = (f"If I remove all the {plural(removed_thing, 2)} from the {place}, how many {colour} objects remain on it?"
                if removal else f"How many {colour} objects are on the {place}?")
    lines = [f"On the {place}, there {'is' if entries[0][1] == 1 else 'are'} {text}.", f"Question: {question}"]
    lines.extend(f"{letter}: {name}" for letter, name in zip(LETTERS_17, NUMBER_NAMES))
    return "\n".join(lines)


# ---- BBH's formal fallacies: syllogisms over properties, valid or invalid by exhaustive model checking. ----

PROPERTY_POOLS = (
    ("a cousin of {n}", "a schoolmate of {n}", "a stepsister of {n}", "a close friend of {n}", "an ancestor of {n}",
     "a workmate of {n}", "a half-brother of {n}", "a great-grandfather of {n}"),
    ("a fan of {c}", "an opponent to {c}", "a devotee of {c}", "an expert of {c}", "a backer of {c}", "a critic of {c}"),
    ("a regular consumer of {p} soap", "an occasional purchaser of {p} shampoo", "a loyal buyer of {p} products",
     "an owner of a {p} soap", "a frequent consumer of {p} cheese"),
)
FILL = {"n": ("Lisa", "Tom", "Carmen", "Jeffrey", "Priya", "Oscar", "Hannah", "Yusuf", "Ines", "Marco"),
        "c": ("FC Porto", "Real Betis", "AS Roma", "Celtic FC", "Ajax Amsterdam", "SK Rapid Wien", "FC Basel"),
        "p": ("Lush", "Dove", "Nivea", "Aveeno", "Olay", "Softsoap", "Pears")}
OPENERS = ('Here comes a perfectly valid argument: ', 'Is the following argument sound? ',
           'Some people claim the following, and we want to know whether it holds: ',
           'Consider this argument about a group of people: ')


def phrase(form: tuple, words: list[str]) -> str:
    """Natural wording for a statement form over property phrases (index into words) or a named individual."""
    kind = form[0]

    def bare(phrase_: str) -> str:  # "a cousin of Lisa" -> "cousin of Lisa", after "every" or "no"
        return phrase_.split(" ", 1)[1] if phrase_.startswith(("a ", "an ")) else phrase_
    if kind == "all":
        a, b = words[form[1]], words[form[2]]
        return random.Random(str(form)).choice((f"every {bare(a)} is {b}", f"being {a} is sufficient for being {b}",
                                                f"being {b} is necessary for being {a}"))
    if kind == "none":
        a, b = words[form[1]], words[form[2]]
        return random.Random(str(form)).choice((f"no {bare(a)} is {b}", f"whoever is {a} is not {b}"))
    if kind == "all_not":
        a, b = words[form[1]], words[form[2]]
        return f"whoever is not {a} is {b}"
    person, index, positive = form[1], form[2], form[3]
    return f"{person} is {'' if positive else 'not '}{words[index]}"


def holds(form: tuple, realised: frozenset, individual: dict[str, frozenset]) -> bool:
    """Whether a statement is true in a situation: `realised` is the set of property combinations that exist (each a
    frozenset of property indexes), `individual` gives each named person's combination."""
    kind = form[0]
    if kind == "all":
        return all(form[2] in t for t in realised if form[1] in t)
    if kind == "none":
        return all(form[2] not in t for t in realised if form[1] in t)
    if kind == "all_not":
        return all(form[2] in t for t in realised if form[1] not in t)
    return (form[2] in individual[form[1]]) == form[3]


def situations(properties: int, people: list[str]):
    """Every situation over `properties` properties: a non-empty set of existing combinations, with each named person
    given one of them."""
    import itertools
    types = [frozenset(i for i in range(properties) if mask >> i & 1) for mask in range(2 ** properties)]
    for size in range(1, len(types) + 1):
        for realised in itertools.combinations(types, size):
            for assignment in itertools.product(realised, repeat=len(people)):
                yield frozenset(realised), dict(zip(people, assignment))


def valid(premises: list[tuple], conclusion: tuple, properties: int, people: list[str]) -> bool | None:
    """True if the conclusion holds wherever all premises hold; False if not; None if the premises can never all hold."""
    consistent = False
    for realised, individual in situations(properties, people):
        if all(holds(p, realised, individual) for p in premises):
            consistent = True
            if not holds(conclusion, realised, individual):
                return False
    return True if consistent else None


def random_form(rng: random.Random, properties: int, people: list[str]) -> tuple:
    if people and rng.random() < 0.35:
        return ("person", rng.choice(people), rng.randrange(properties), rng.random() < 0.7)
    a, b = rng.sample(range(properties), 2)
    return (rng.choice(("all", "all", "none", "all_not")), a, b)


def formal_fallacies(slot: Slot) -> str:
    """Two or three premises and a conclusion; the slot label says whether the conclusion follows ("valid") or not."""
    rng = random.Random(f"formal-fallacies-{slot['id']}")
    want = slot["label"] == "valid"
    for _ in range(5000):
        properties = 3
        people = [rng.choice(FILL["n"])] if rng.random() < 0.4 else []
        premises = [random_form(rng, properties, people) for _ in range(rng.choice((2, 2, 3)))]
        conclusion = random_form(rng, properties, people)
        if conclusion in premises or len(set(premises)) < len(premises):
            continue
        # A conclusion true in every situation by itself would make "valid" trivial; require the premises to matter.
        if valid([], conclusion, properties, people) is not False:
            continue
        verdict = valid(premises, conclusion, properties, people)
        if verdict is None or verdict != want:
            continue
        pool = rng.choice(PROPERTY_POOLS)
        words = [w.format(**{k: rng.choice(v) for k, v in FILL.items()}) for w in rng.sample(pool, properties)]
        ordinals = ("First", "Second", "Third")
        sentences = [f"{ordinals[i]}, {phrase(p, words)}." for i, p in enumerate(premises)]
        claim = f"It follows that {phrase(conclusion, words)}."
        opener = rng.choice(OPENERS)
        return f'"{opener}{" ".join(sentences)} {claim}"'
    raise ValueError(f"Slot {slot['id']}: no argument with verdict {want} found")
