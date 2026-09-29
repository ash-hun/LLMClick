"""Build pools of people, organisations and topics once with the teacher, so each generation slot draws fresh material."""

import argparse
import asyncio
import json
import random
from pathlib import Path
from typing import Protocol, TypedDict

import httpx

from jeff.families import DOMAINS, Slot
from jeff.teacher import TEACHER_MODEL, teacher_url, Teacher
from jeff.types import JSONValue

REGIONS = ("the United Kingdom and Ireland", "the United States", "Latin America", "Spain and Portugal", "France and Belgium",
           "Germany, Austria and Switzerland", "Italy and Greece", "Eastern Europe", "India and Pakistan", "China, Japan and Korea",
           "Southeast Asia", "West Africa", "East Africa", "the Middle East", "Scandinavia and the Netherlands")
PER_CALL = 50
PEOPLE_PER_SLOT = 5  # Object-tracking puzzles need up to five people.
MINIMUM = {"people": 500, "organisations": 30, "topics": 40}

NAMES = """List {n} realistic full names (first name and family name) of ordinary people from {region}.
Mix women and men, and older and younger names. No famous people. Every name must be different."""
ORGANISATIONS = """List {n} invented but realistic names of organisations in the area of {domain}: small companies, shops,
clinics, clubs, agencies, schools or charities. Do not use real brands. Every name must be different."""
TOPICS = """List {n} different, concrete situations in the area of {domain} that a message, review, report, argument, rule
or dispute could be about. Each is 5 to 12 words and names who is involved and what happened, for example
"a tenant disputing a late fee after a bank holiday". Make them varied; no two should be about the same thing."""


class Materials(TypedDict):
    people: list[str]
    organisations: dict[str, list[str]]
    topics: dict[str, list[str]]


class Completer(Protocol):
    async def complete(self, *, system: str, user: str, schema: dict[str, object], temperature: float,
                       max_tokens: int) -> dict[str, JSONValue]: ...


def list_schema(count: int) -> dict[str, object]:
    return {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "string"},
                                                      "minItems": count, "maxItems": count}},
            "required": ["items"], "additionalProperties": False}


async def ask(teacher: Completer, prompt: str) -> list[str]:
    reply = await teacher.complete(system="You produce lists of varied, realistic material. Return JSON only.",
                                   user=prompt, schema=list_schema(PER_CALL), temperature=1.0, max_tokens=4000)
    items = reply["items"]
    if not isinstance(items, list):
        raise ValueError(f"Teacher reply has no list of items for prompt: {prompt[:80]}")
    return [str(item).strip() for item in items if str(item).strip()]


def unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values:
        key = " ".join(value.casefold().split())
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def require(name: str, values: list[str], minimum: int) -> list[str]:
    if len(values) < minimum:
        raise ValueError(f"Only {len(values)} unique {name} after deduplication; at least {minimum} are needed")
    return values


async def build(teacher: Completer) -> Materials:
    people = await asyncio.gather(*(ask(teacher, NAMES.format(n=PER_CALL, region=region)) for region in REGIONS))
    organisations = await asyncio.gather(*(ask(teacher, ORGANISATIONS.format(n=PER_CALL, domain=d)) for d in DOMAINS))
    topics = await asyncio.gather(*(ask(teacher, TOPICS.format(n=PER_CALL, domain=d)) for d in DOMAINS))
    return {"people": require("people", unique([n for group in people for n in group]), MINIMUM["people"]),
            "organisations": {d: require(f"organisations for {d}", unique(o), MINIMUM["organisations"])
                              for d, o in zip(DOMAINS, organisations, strict=True)},
            "topics": {d: require(f"topics for {d}", unique(t), MINIMUM["topics"]) for d, t in zip(DOMAINS, topics, strict=True)}}


def dress(slots: list[Slot], materials: Materials, seed: int) -> list[Slot]:
    """Give every slot its own topic, five people and one organisation, drawn deterministically from the pools."""
    dressed: list[Slot] = []
    for slot in slots:
        rng = random.Random(f"{seed}-materials-{slot['id']}")
        dressed.append({**slot, "topic": rng.choice(materials["topics"][slot["domain"]]),
                        "people": ", ".join(rng.sample(materials["people"], PEOPLE_PER_SLOT)),
                        "organisation": rng.choice(materials["organisations"][slot["domain"]])})
    return dressed


def load(path: Path) -> Materials:
    materials: Materials = json.loads(path.read_text())
    missing = [domain for domain in DOMAINS if domain not in materials["topics"] or domain not in materials["organisations"]]
    if missing:
        raise ValueError(f"{path} has no topics or organisations for domains: {missing}")
    return materials


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/materials.json"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to replace existing materials: {args.out}")

    async def go() -> Materials:
        async with httpx.AsyncClient() as client:
            teacher = Teacher(client, url=teacher_url(), model=TEACHER_MODEL, cache=args.out.parent / "materials-cache",
                              log=args.out.parent / "materials-teacher.log")
            return await build(teacher)

    materials = asyncio.run(go())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(materials, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"people": len(materials["people"]),
                      "organisations": {d: len(v) for d, v in materials["organisations"].items()},
                      "topics": {d: len(v) for d, v in materials["topics"].items()}}, indent=1))


if __name__ == "__main__":
    main()
