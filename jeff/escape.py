"""Escape options ("None of these", "Other", "Not applicable" ...) added to ordinary multiple-choice rows, so the model
learns that "nothing listed fits" is a real answer. Without this it almost never picks such an option (the navigation
test: 0% on "none of these" cases before fine-tuning).

A small share of eligible rows (picked by a hash of the id) gets one escape option, worded in one of several ways and
placed at a random position. In half of them the correct option is removed, so the escape option becomes the answer;
in the other half it is just one more wrong option. A row is eligible only when this is sound: its options carry their
own content (not "Option A" or "Response A is more correct"), the state does not list the options itself, at least
two other options remain, and it has no escape-like option already."""

import hashlib
import random
import re
from copy import deepcopy

from jeff.types import Example

SHARE = 0.05
# Sixteen generic, domain-neutral wordings, so the model learns the idea rather than one phrase; each row also gets a
# random capitalisation (see CASES).
WORDINGS = (("none_of_these", "None of these"), ("other", "Other: something not listed"), ("not_applicable", "Not applicable"),
            ("none_of_the_above", "None of the above"), ("no_match", "No option matches"), ("something_else", "Something else"),
            ("neither", "Neither of these"), ("n_a", "N/A"), ("none_apply", "None of these apply"),
            ("not_listed", "Not listed"), ("no_suitable_option", "No suitable option"), ("none_of_the_options", "None of the options"),
            ("does_not_apply", "Does not apply"), ("no_answer_fits", "No answer fits"), ("unlisted", "Something unlisted"),
            ("none", "None"))
CASES = (str, str.lower, str.title, str.upper)  # as written (sentence case), lower, Title, UPPER
GENERIC = re.compile(r"^(option|response|statement|item|date|source|choice)\s+[a-z0-9]+\b", re.IGNORECASE)
ESCAPE_LIKE = re.compile(r"\b(none|other|neither|unknown|insufficient|not applicable|n/a|cannot|can't|undetermined|no match)\b",
                         re.IGNORECASE)


def fraction(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16) / 16 ** 12


def eligible(row: Example) -> bool:
    question = row["question"]
    if question["type"] != "choice" or row["target"] != row["label"]:  # soft targets cannot simply be relabelled
        return False
    criteria = question["criteria"]
    if len(criteria) < 3 or any(not isinstance(text, str) or not text.strip() or GENERIC.match(text) for text in criteria.values()):
        return False
    if any(ESCAPE_LIKE.search(f"{key} {text}") for key, text in criteria.items()):
        return False
    state = row["state"] if isinstance(row["state"], str) else " ".join(str(v) for v in row["state"].values())  # type: ignore[union-attr]
    return not any(re.search(rf"(?m)^\s*{re.escape(key)}\s*[:.)]", state) for key in criteria)


def add_escape(row: Example, seed: int) -> Example:
    """The row with one escape option; in half the cases the correct option is removed and the escape is the answer."""
    rng = random.Random(f"{seed}-escape-{row['id']}")
    key, text = rng.choice(WORDINGS)
    text = rng.choice(CASES)(text)
    result = deepcopy(row)
    criteria = dict(result["question"]["criteria"])  # type: ignore[typeddict-item]
    remove = rng.random() < 0.5
    if remove:
        del criteria[str(row["label"])]
    items = list(criteria.items())
    items.insert(rng.randrange(len(items) + 1), (key, text))
    result["question"] = {**result["question"], "criteria": dict(items)}  # type: ignore[typeddict-item]
    result["label"] = result["target"] = key if remove else row["label"]  # type: ignore[typeddict-item]
    result["id"] = f"{row['id']}-escape"
    result["source"] = {**result["source"], "escape_option": key, "escape_is_answer": remove}
    return result


def add(rows: list[Example], seed: int, share: float = SHARE) -> tuple[list[Example], int]:
    """Replace `share` of the eligible rows with their escape version (the original is dropped: nothing is duplicated)."""
    result, changed = [], 0
    for row in rows:
        if fraction(f"{seed}-escape-pick-{row['id']}") < share and eligible(row):
            result.append(add_escape(row, seed))
            changed += 1
        else:
            result.append(row)
    return result, changed
