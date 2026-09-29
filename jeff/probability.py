"""Probability decisions built in code, with the exact probability as the training target.

Each row is a short realistic situation (an inspection sample, dice, cards, a raffle, rain over several days, coin
flips) and a yes/no question about whether an event happens. Code computes the exact probability p; the row's label
is p > 0.5 and its target is p itself, so the model learns to give calibrated probabilities, as JevBench's probability
items ask ("give probabilities that reflect the evidence"). Some situations include a superseded plan or a detail that
changes the arithmetic (drawing without replacement), as JevBench's do. Items within 0.03 of one half are left out."""

import argparse
import json
import random
from fractions import Fraction
from math import comb
from pathlib import Path

from jeff.data import validate, write_rows
from jeff.types import Example

INSTRUCTION = " Give probabilities that reflect the evidence in the state."
NAMES = ("Mia", "Tomás", "Aisha", "Jonas", "Priya", "Kenji", "Lucia", "Omar", "Freya", "Dmitri", "Zanele", "Hugo")
GAMES = ("a board game", "a dice game at a school fair", "a tabletop role-playing game", "a family game night",
         "a charity casino evening", "a pub quiz tiebreak")
EVENTS = ("outdoor market", "open-air concert series", "harvest festival", "garden party", "football tournament",
          "school sports week", "wedding weekend")


def at_least_one_defective(rng: random.Random) -> tuple[str, str, Fraction]:
    lot = rng.randint(8, 30)
    defective = rng.randint(1, max(1, lot // 3))
    sample = rng.randint(1, min(6, lot - defective))
    product = rng.choice(("power supply units", "valves", "circuit boards", "helmets", "pumps", "batteries"))
    text = (f"Incoming inspection, lot of {lot} {product}. The supplier's own test found that exactly {defective} of the "
            f"{lot} units are defective; the defective units are not marked. The inspection plan draws {sample} units at random "
            f"from the lot, without replacement, and tests each one.")
    if rng.random() < 0.4:
        old = max(1, sample - 1) if sample > 1 else sample + 1
        text += f" (An earlier version of the plan drew {old} units; it was replaced by the current plan before this lot.)"
    p = 1 - Fraction(comb(lot - defective, sample), comb(lot, sample))
    return text, "Will the inspection sample contain at least one defective unit?", p


def dice_sum(rng: random.Random) -> tuple[str, str, Fraction]:
    dice = rng.randint(2, 3)
    target = rng.randint(dice + 2, 6 * dice - 2)
    import itertools
    hits = sum(sum(roll) >= target for roll in itertools.product(range(1, 7), repeat=dice))
    who = rng.choice(NAMES)
    text = (f"In {rng.choice(GAMES)}, {who} rolls {dice} fair six-sided dice and adds them. {who} wins the round only if "
            f"the total is at least {target}.")
    return text, f"Will {who}'s total be at least {target} on this roll?", Fraction(hits, 6 ** dice)


def cards(rng: random.Random) -> tuple[str, str, Fraction]:
    drawn = rng.randint(2, 8)
    rank = rng.choice(("ace", "king", "queen", "seven"))
    who = rng.choice(NAMES)
    text = (f"A standard 52-card deck is shuffled and {drawn} cards are dealt face down to {who}, without "
            f"replacement. The deck has four cards of each rank.")
    p = 1 - Fraction(comb(48, drawn), comb(52, drawn))
    return text, f"Does {who}'s hand contain at least one {rank}?", p


def raffle(rng: random.Random) -> tuple[str, str, Fraction]:
    tickets = rng.choice((50, 80, 100, 120, 200))
    mine = rng.randint(3, 40)
    prizes = rng.randint(1, 5)
    who = rng.choice(NAMES)
    text = (f"A club raffle sold {tickets} tickets. {who} bought {mine} of them. {prizes} winning ticket"
            f"{'s are' if prizes > 1 else ' is'} drawn at random, one after another, and a drawn ticket is not put back.")
    p = 1 - Fraction(comb(tickets - mine, prizes), comb(tickets, prizes))
    return text, f"Will {who} win at least one prize?", p


def rainy_days(rng: random.Random) -> tuple[str, str, Fraction]:
    days = rng.randint(2, 5)
    chance = rng.choice((10, 15, 20, 25, 30, 40))
    event = rng.choice(EVENTS)
    text = (f"The forecast gives a {chance}% chance of rain on each of the next {days} days, and forecasters say the "
            f"days are independent of one another. The {event} runs on all {days} days.")
    p = 1 - Fraction(100 - chance, 100) ** days
    return text, "Will it rain on at least one of the event days?", p


def coins(rng: random.Random) -> tuple[str, str, Fraction]:
    flips = rng.randint(3, 10)
    need = rng.randint(1, flips)
    who, other = rng.sample(NAMES, 2)
    text = (f"{who} and {other} bet on a fair coin flipped {flips} times. {who} wins the bet if at least {need} of the "
            f"flips land heads.")
    p = Fraction(sum(comb(flips, k) for k in range(need, flips + 1)), 2 ** flips)
    return text, f"Does {who} win the bet?", p


SITUATIONS = (at_least_one_defective, dice_sum, cards, raffle, rainy_days, coins)


def build(count: int, seed: int) -> list[Example]:
    rows: list[Example] = []
    seen: set[tuple[str, str]] = set()
    rng = random.Random(seed)
    for _ in range(200 * count):
        if len(rows) == count:
            break
        situation = rng.choice(SITUATIONS)
        state, question, p = situation(rng)
        if abs(p - Fraction(1, 2)) < Fraction(3, 100) or (state, question) in seen:
            continue
        # Keep the two answers about equally common, so "yes" is not the easy guess.
        wanted = len(rows) % 2 == 0
        if (p > Fraction(1, 2)) != wanted:
            continue
        seen.add((state, question))
        identifier = f"probability-{seed}-{len(rows):06d}"
        rows.append({"id": identifier, "suite": "probability", "family": identifier, "state": state,
                     "question": {"type": "noul", "instructions": question + INSTRUCTION,
                                  "criteria": {"true": "Yes, it happens.", "false": "No, it does not happen."}},
                     "label": p > Fraction(1, 2), "target": round(float(p), 4),
                     "source": {"dataset": "probability", "situation": situation.__name__, "probability": float(p)}})
    if len(rows) < count:
        raise ValueError(f"Only {len(rows)} distinct probability rows found; ask for fewer")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/probability"))
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    rows = build(args.count, args.seed)
    validate(rows)
    print(json.dumps({"rows": len(rows), "sha256": write_rows(args.out / "train.jsonl", rows)}))


if __name__ == "__main__":
    main()
