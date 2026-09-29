"""Voice-navigation decisions with speech-recognition errors, built in code from two public sources.

Real voice-assistant commands (Amazon MASSIVE, English, with their labelled intents and marked-up phrases) are
"misheard" by swapping words for others that sound the same, using the CMU Pronouncing Dictionary (for example
"buy" -> "by", "sell" -> "cell", "buyout" -> "by out"). Two kinds of decision follow, each with an exact answer:

- intent: which of several actions the (possibly garbled) command asks for; questions such as "what does X mean" are
  their own intent, so "asking a question" is one of the options when it applies;
- phrase: which of several real names (places, businesses, songs, devices...) a garbled phrase in the command meant.

No model is involved: the commands are human-written, the sound-alikes come from the dictionary, and the labels come
from MASSIVE's annotations. Training rows use MASSIVE's train split; the evaluation rows use its test split."""

import argparse
import json
import random
import re
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

from jeff.data import validate, write_rows
from jeff.types import Example

CMUDICT = ("https://raw.githubusercontent.com/cmusphinx/cmudict/{commit}/cmudict.dict",
           "74790861f652b15e4ac49015a90074ad62a27690")  # BSD-2-Clause licence
MASSIVE_ARCHIVE = Path("data/public/raw/massive-1.1.tar.gz")  # pinned by checksum in jeff.data
TRANSCRIPT = "Voice transcript (may contain speech-recognition errors): "
INTENT_OPTIONS = 6
# Words in MASSIVE intent names that are abbreviations or run together, spelled out for the option descriptions.
INTENT_WORDS = {"iot": "smart home", "hue": "lights", "wemo": "smart plug", "qa": "question", "lightoff": "light off",
                "lighton": "light on", "lightup": "light brighter", "lightdim": "light dimmer", "lightchange": "light colour",
                "addcontact": "add contact", "sendemail": "send email", "querycontact": "contact details",
                "createoradd": "create or add to", "dislikeness": "dislike", "likeness": "like", "datetime": "date and time",
                "query": "ask", "quirky": "chat", "factoid": "fact", "maths": "maths", "remove": "remove", "set": "set"}
QUESTION_INTENTS = {"qa_definition", "qa_factoid", "qa_maths", "qa_currency", "qa_stock"}
PHRASE_SLOTS = {"place_name": "place", "business_name": "business", "event_name": "event", "artist_name": "artist",
                "song_name": "song", "device_type": "device", "food_type": "food", "radio_name": "radio station",
                "playlist_name": "playlist", "app_name": "app", "game_name": "game", "podcast_name": "podcast",
                "audiobook_name": "audiobook", "news_topic": "news topic", "list_name": "list", "business_type": "kind of business"}


def load_pronunciations(path: Path) -> dict[str, list[str]]:
    """Word -> its pronunciations, from the CMU dictionary file (variants such as "read(2)" folded into "read")."""
    pronunciations: dict[str, list[str]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line or line.startswith(";;;"):
            continue
        word, phones = line.split(" ", 1)
        phones = phones.split("#")[0].strip()
        pronunciations[re.sub(r"\(\d+\)$", "", word)].append(re.sub(r"\d", "", phones))  # stress marks do not matter
    return dict(pronunciations)


class SoundAlikes:
    """Other spellings of the same sounds: whole-word homophones, and splits of a word into two sound-alike words."""

    def __init__(self, pronunciations: dict[str, list[str]], common: set[str]) -> None:
        """Only words in `common` are used as replacements: speech recognisers output everyday words, not the rare
        entries and abbreviations the dictionary also holds."""
        self.pronunciations = pronunciations
        self.common = common
        self.cache: dict[str, list[str]] = {}
        self.by_sound: dict[str, list[str]] = defaultdict(list)
        for word, sounds in pronunciations.items():
            if re.fullmatch(r"[a-z]+", word):
                for sound in sounds:
                    self.by_sound[sound].append(word)

    def homophones(self, word: str) -> list[str]:
        return sorted({other for sound in self.pronunciations.get(word, []) for other in self.by_sound[sound]
                       if other != word and other in self.common})

    def splits(self, word: str) -> list[str]:
        """Two words that together sound like `word`, e.g. "buyout" -> "by out"."""
        found = set()
        for cut in range(2, len(word) - 1):
            left, right = word[:cut], word[cut:]
            for whole in self.pronunciations.get(word, []):
                for a in self.pronunciations.get(left, []):
                    if whole.startswith(a + " ") and whole[len(a) + 1:] in self.pronunciations.get(right, []):
                        for x in [left] + self.homophones(left):
                            for y in [right] + self.homophones(right):
                                if x in self.common and y in self.common:
                                    found.add(f"{x} {y}")
        return sorted(found)

    def garble(self, text: str, rng: random.Random, changes: int) -> str | None:
        """Replace up to `changes` words with sound-alikes; None when no word in the text has one."""
        words = text.split()
        for word in words:
            if word not in self.cache:
                self.cache[word] = self.homophones(word) + self.splits(word)
        options = {i: self.cache[w] for i, w in enumerate(words)}
        candidates = [i for i, found in options.items() if found]
        if not candidates:
            return None
        for i in rng.sample(candidates, min(changes, len(candidates))):
            words[i] = rng.choice(options[i])
        garbled = " ".join(words)
        return garbled if garbled != text else None


def common_words(path: Path, minimum: int) -> set[str]:
    """Words used at least `minimum` times across the states of a JSONL file of public training rows."""
    counts: dict[str, int] = defaultdict(int)
    for line in path.read_text().split("\n"):
        if line:
            for word in re.findall(r"[a-z]+", json.dumps(json.loads(line)["state"]).lower()):
                counts[word] += 1
    return {word for word, count in counts.items() if count >= minimum and (len(word) > 1 or word in ("a", "i"))}


def intent_description(intent: str) -> str:
    words = [INTENT_WORDS.get(part, part) for part in intent.split("_")]
    if intent in QUESTION_INTENTS:
        return f"Not a command: the user is asking a question ({' '.join(words[1:])})"
    return " ".join(words).capitalize()


def phrases(annotated: str) -> list[tuple[str, str]]:
    return re.findall(r"\[(\w+) : ([^\]]+)\]", annotated)


def intent_row(row: dict, intents: list[str], by_scenario: dict[str, list[str]], sounds: SoundAlikes,
               rng: random.Random, split: str) -> Example | None:
    changes = rng.choice((0, 1, 1, 2))  # a quarter of commands are heard correctly
    heard = row["utt"] if changes == 0 else sounds.garble(row["utt"], rng, changes)
    if heard is None:
        return None
    related = [i for i in by_scenario[row["scenario"]] if i != row["intent"]]
    others = rng.sample([i for i in intents if i != row["intent"] and i not in related], INTENT_OPTIONS)
    chosen = ([row["intent"]] + rng.sample(related, min(2, len(related))) + others)[:INTENT_OPTIONS]
    rng.shuffle(chosen)
    return example(f"voice-intent-{split}-{row['id']}", TRANSCRIPT + f'"{heard}"',
                   "Which action does the user want?", {intent: intent_description(intent) for intent in chosen},
                   row["intent"], row, split, "intent", changes)


def phrase_row(row: dict, values: dict[str, list[str]], sounds: SoundAlikes, rng: random.Random, split: str) -> Example | None:
    marked = [(slot, value) for slot, value in phrases(row["annot_utt"]) if slot in PHRASE_SLOTS and len(values[slot]) >= 5]
    if not marked:
        return None
    slot, value = rng.choice(marked)
    garbled = sounds.garble(value, rng, rng.choice((1, 1, 2)))
    if garbled is None:
        return None
    heard = row["utt"].replace(value, garbled, 1) if rng.random() < .7 else garbled  # the command, or the phrase alone
    if heard == row["utt"] or value in heard:
        return None
    # Distractors: other names of the same kind, preferring ones that share a word with the true name.
    others = [v for v in values[slot] if v != value]
    sharing = [v for v in others if set(v.split()) & set(value.split())]
    pool = rng.sample(sharing, min(2, len(sharing))) + rng.sample(others, min(6, len(others)))
    distractors = list(dict.fromkeys(v for v in pool if v != value))[:rng.randint(3, 5)]
    options = distractors + [value]
    rng.shuffle(options)
    keys = [f"option_{index + 1}" for index in range(len(options))]
    return example(f"voice-phrase-{split}-{row['id']}", TRANSCRIPT + f'"{heard}"',
                   f"Which {PHRASE_SLOTS[slot]} did the user mean?", dict(zip(keys, options)),
                   keys[options.index(value)], row, split, "phrase", 1)


def example(identifier: str, state: str, instructions: str, criteria: dict[str, str], label: str, row: dict,
            split: str, kind: str, changes: int) -> Example:
    return {"id": identifier, "suite": "voice_navigation", "family": f"voice-{row['id']}", "state": state,
            "question": {"type": "choice", "instructions": instructions, "criteria": criteria},
            "label": label, "target": label,
            "source": {"dataset": "voice_navigation", "built_from": ["amazon-massive-1.1 en-US", f"cmudict@{CMUDICT[1]}"],
                       "massive_split": split, "massive_id": row["id"], "kind": kind, "words_changed": changes}}


def build(rows: list[dict], sounds: SoundAlikes, split: str, seed: int) -> list[Example]:
    intents = sorted({row["intent"] for row in rows})
    by_scenario: dict[str, list[str]] = defaultdict(list)
    for intent in intents:
        by_scenario[intent.split("_")[0]].append(intent)
    values: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        for slot, value in phrases(row["annot_utt"]):
            values[slot].append(value)
    values = {slot: sorted(set(found)) for slot, found in values.items()}
    result: list[Example] = []
    for row in rows:
        rng = random.Random(f"{seed}-{split}-{row['id']}")
        for made in (intent_row(row, intents, by_scenario, sounds, rng, split), phrase_row(row, values, sounds, rng, split)):
            if made is not None:
                result.append(made)
    return result


def scoped(rows: list[Example], intent_rows: int, clean_share: float, seed: int) -> list[Example]:
    """All phrase rows (the rarer sound-matching skill) plus `intent_rows` intent rows, of which about `clean_share`
    were heard correctly: MASSIVE already feeds the mix plain intent-like classification."""
    rng = random.Random(f"{seed}-scope")
    intents = [r for r in rows if r["source"]["kind"] == "intent"]
    clean = [r for r in intents if r["source"]["words_changed"] == 0]
    garbled = [r for r in intents if r["source"]["words_changed"] > 0]
    keep_clean = min(len(clean), round(intent_rows * clean_share))
    chosen = rng.sample(clean, keep_clean) + rng.sample(garbled, min(len(garbled), intent_rows - keep_clean))
    return sorted([r for r in rows if r["source"]["kind"] == "phrase"] + chosen, key=lambda r: r["id"])


def massive(split: str) -> list[dict]:
    with tarfile.open(MASSIVE_ARCHIVE) as archive:
        member = next(m for m in archive.getmembers() if m.name.endswith("/data/en-US.jsonl"))
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError("MASSIVE archive has no en-US file")
        return [row for row in map(json.loads, stream) if row["partition"] == split]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/voice"))
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--vocabulary", type=Path, default=Path("data/extra/train.jsonl"),
                        help="Public training rows whose words count as everyday words")
    parser.add_argument("--minimum-count", type=int, default=5)
    parser.add_argument("--train-intent-rows", type=int, default=4000,
                        help="Intent rows kept for training (all phrase rows are kept); the test split is kept whole")
    args = parser.parse_args()
    dictionary = args.out / f"cmudict-{CMUDICT[1][:12]}.dict"
    if not dictionary.exists():
        args.out.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CMUDICT[0].format(commit=CMUDICT[1]), dictionary)
    sounds = SoundAlikes(load_pronunciations(dictionary), common_words(args.vocabulary, args.minimum_count))
    report = {}
    for split, name in (("train", "train"), ("test", "test")):
        rows = build(massive(split), sounds, split, args.seed)
        if split == "train":
            rows = scoped(rows, args.train_intent_rows, clean_share=0.15, seed=args.seed)
        validate(rows)
        report[name] = {"rows": len(rows), "sha256": write_rows(args.out / f"{name}.jsonl", rows),
                        "kinds": {kind: sum(r["source"]["kind"] == kind for r in rows) for kind in ("intent", "phrase")}}
    (args.out / "manifest.json").write_text(json.dumps({"cmudict": CMUDICT[1], "seed": args.seed, "vocabulary": str(args.vocabulary),
                                                        "minimum_count": args.minimum_count, **report}, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
