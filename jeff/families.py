"""The decision families synthetic examples are made for, and a deterministic plan of generation slots."""

import random
from dataclasses import dataclass
from itertools import groupby
from typing import Literal, NotRequired, TypedDict

from jeff.types import Question

CONDITIONS = ("clean", "noisy", "negated", "long", "skewed")
DOMAINS = ("home repair", "healthcare", "law", "finance", "education", "travel", "retail", "software", "science",
           "sports", "government services", "food and restaurants")
LENGTHS = ("one or two sentences", "a short paragraph", "two or three paragraphs")
DIFFICULTIES = ("easy", "hard")


@dataclass(frozen=True)
class Family:
    name: str
    kind: Literal["choice", "noul"]
    instructions: str
    criteria: dict[str, str]

    @property
    def labels(self) -> list[str]:
        return ["true", "false"] if self.kind == "noul" else list(self.criteria)

    def question(self) -> Question:
        if self.kind == "noul":
            return {"type": "noul", "instructions": self.instructions,
                    "criteria": {"true": self.criteria["true"], "false": self.criteria["false"]}}
        return {"type": "choice", "instructions": self.instructions, "criteria": dict(self.criteria)}


def noul(name: str, instructions: str, true: str, false: str) -> Family:
    return Family(name, "noul", instructions, {"true": true, "false": false})


FAMILIES: tuple[Family, ...] = (
    Family("target_sentiment", "choice", "Classify the writer's overall attitude toward the target named in the text.",
           {"positive": "Overall favourable toward the target.", "negative": "Overall unfavourable toward the target.",
            "mixed": "Clearly both favourable and unfavourable.", "neutral": "No favourable or unfavourable judgement."}),
    Family("entailment", "choice", "Using only the premise, classify the hypothesis. Do not assume unstated facts.",
           {"entailment": "The premise shows the hypothesis is true.", "contradiction": "The premise shows the hypothesis is false.",
            "neutral": "The premise shows neither."}),
    noul("answerability", "Can the question be answered from the passage alone?",
         "The passage gives enough information for a definite answer.", "The passage lacks information needed for a definite answer."),
    noul("response_faithfulness", "Does the response contain any information that conflicts with the source or is not supported by it?",
         "The response contains conflicting or unsupported information.", "Every claim in the response is supported by the source."),
    Family("pairwise_answer_quality", "choice", "Which response answers the question more correctly?",
           {"A": "Response A is more correct.", "B": "Response B is more correct."}),
    Family("pronoun_resolution", "choice", "Which option correctly fills the blank (_) in the sentence?",
           {"1": "The first person or thing named in the options.", "2": "The second person or thing named in the options."}),
    Family("logical_deduction", "choice", "Using the clues, which statement is true?",
           {"A": "Statement A is true.", "B": "Statement B is true.", "C": "Statement C is true.", "D": "Statement D is true."}),
    Family("object_tracking", "choice", "After all the swaps, which item does the named person hold?",
           {"A": "Item A.", "B": "Item B.", "C": "Item C.", "D": "Item D."}),
    Family("date_arithmetic", "choice", "Which date is correct?",
           {"A": "Date A.", "B": "Date B.", "C": "Date C.", "D": "Date D."}),
    noul("rule_evaluation", "Is the final statement true under the stated rules?",
         "The statement is true.", "The statement is false."),
    Family("argument_validity", "choice", "Is the argument deductively valid given its premises?",
           {"valid": "The conclusion follows necessarily from the premises.", "invalid": "The conclusion does not follow necessarily."}),
    Family("policy_application", "choice", "Apply only the fictional policy in the text, including its exceptions, to the request.",
           {"allowed": "The facts show the policy permits the request.", "denied": "The facts show the policy denies the request.",
            "insufficient": "A fact needed to decide is missing."}),
    Family("support_routing", "choice", "Route the customer's main request to one department.",
           {"billing": "Charges, invoices, payments or refunds.", "technical": "Something is broken or hard to use.",
            "account": "Login, identity or profile changes.", "delivery": "Shipping, tracking or missing packages.",
            "other": "None of the other departments fits."}),
    Family("numeric_comparison", "choice", "Compare the first named quantity with the second, using only the facts and any unit conversions.",
           {"less": "The first is smaller.", "equal": "They are equal.", "greater": "The first is larger.",
            "unknown": "The facts do not determine the relationship."}),
    Family("event_order", "choice", "Using only the timeline, when did the first named event happen relative to the second?",
           {"before": "The first event happened earlier.", "after": "The first event happened later.",
            "same_time": "They happened at the same time.", "unknown": "The order cannot be determined."}),
    Family("evidence_attribution", "choice", "Which source on its own establishes every part of the claim, including any stated cause? Judge each source separately.",
           {"source_a": "Only source A.", "source_b": "Only source B.", "both": "Each source on its own.", "neither": "Neither source on its own."}),
    Family("financial_news_tone", "choice", "From an investor's point of view, classify the sentiment of this financial news text.",
           {"positive": "Likely to improve the company's financial position or share price.",
            "negative": "Likely to harm the company's financial position or share price.",
            "neutral": "No clear positive or negative effect."}),
    Family("paraphrase", "choice", "Do the two statements make the same factual claim, including scope, numbers and qualifications?",
           {"equivalent": "Same meaning.", "different": "At least one substantive difference."}),
    noul("sarcasm", "Is the statement sarcastic?", "The statement is sarcastic.", "The statement is sincere."),
    Family("grounded_pairwise", "choice", "Which response answers the question more correctly?",
           {"A": "Response A is more correct.", "B": "Response B is more correct."}),
    noul("navigate", "If you follow these instructions, do you end up back at the starting point?",
         "You end up back at the starting point.", "You end up somewhere else."),
    noul("boolean_expressions", "Evaluate the boolean expression. Is its value True?",
         "The expression evaluates to True.", "The expression evaluates to False."),
    Family("adjective_order", "choice", "Which option uses the correct English adjective order?",
           {"A": "Option A has the correct adjective order.", "B": "Option B has the correct adjective order."}),
    Family("coloured_objects", "choice", "Answer the question about the objects.",
           {"A": "Option A.", "B": "Option B.", "C": "Option C.", "D": "Option D."}),
    noul("summary_consistency", "Is every statement in the summary consistent with the document?",
         "The summary is fully consistent with the document.", "The summary contains at least one inconsistent statement."),
    # Families shaped like the BBH sub-tasks and JudgeBench items the earlier families did not cover.
    Family("penguins_table", "choice", "Answer the question about the table.",
           {"A": "Option A.", "B": "Option B.", "C": "Option C.", "D": "Option D.", "E": "Option E."}),
    Family("temporal_sequences", "choice", "Using the day's schedule, between what times could the person have gone to the place?",
           {"A": "Option A.", "B": "Option B.", "C": "Option C.", "D": "Option D."}),
    noul("causal_judgement", "How would a typical person answer the question about causation at the end of the story?",
         "A typical person would answer yes.", "A typical person would answer no."),
    Family("translation_error", "choice", "The English translation of the German source contains one error. Which kind of error is it?",
           {"named_entities": "An entity (a name, place or organisation) is changed to a different one.",
            "numerical_values": "A number, ordinal, date or unit is changed.",
            "modifiers_or_adjectives": "A modifier or adjective describing a noun is changed.",
            "negation_or_antonyms": "A negation is added or removed, or a comparison is turned into its opposite.",
            "facts": "A small factual error not covered by the other kinds is introduced.",
            "dropped_content": "A significant clause is left out of the translation."}),
    Family("disambiguation", "choice", "Which option states what the pronoun in the sentence refers to, or is the reference ambiguous?",
           {"A": "Option A is the correct reading.", "B": "Option B is the correct reading.", "C": "The pronoun is ambiguous."}),
    Family("long_pairwise", "choice", "Which response answers the question more correctly?",
           {"A": "Response A is more correct.", "B": "Response B is more correct."}),
    # Built in code in the exact shapes of BBH sub-tasks the model was weakest on.
    Family("tracking_five", "choice", "After all the swaps, what does the named person end up with?",
           {letter: f"Option {letter}." for letter in "ABCDE"}),
    Family("date_understanding", "choice", "Which date answers the question?", {letter: f"Option {letter}." for letter in "ABCDEF"}),
    Family("ordering_five", "choice", "Using the clues, which statement about the order is true?",
           {letter: f"Option {letter}." for letter in "ABCDE"}),
    Family("ordering_seven", "choice", "Using the clues, which statement about the order is true?",
           {letter: f"Option {letter}." for letter in "ABCDEFG"}),
    noul("navigate_turns", "If you follow these instructions, do you end up back at the starting point?",
         "You end up back at the starting point.", "You end up somewhere else."),
    Family("colour_counting", "choice", "Answer the question about the objects.",
           {letter: f"Option {letter}." for letter in "ABCDEFGHIJKLMNOPQ"}),
    Family("formal_fallacies", "choice", "Is the argument, given the explicitly stated premises, deductively valid or invalid?",
           {"valid": "The conclusion follows from the premises.", "invalid": "The conclusion does not follow from the premises."}),
)
BY_NAME = {family.name: family for family in FAMILIES}
# Families whose labels are only positions (A-D, 1/2, A/B). Skewing them would teach a positional bias, so they stay balanced.
POSITIONAL = frozenset({"pairwise_answer_quality", "pronoun_resolution", "logical_deduction", "object_tracking", "date_arithmetic",
                        "adjective_order", "coloured_objects", "grounded_pairwise", "penguins_table", "temporal_sequences",
                        "long_pairwise", "disambiguation", "tracking_five", "date_understanding", "ordering_five",
                        "ordering_seven", "colour_counting"})
# Generation focused on the benchmarks the public-only model is weakest on (JudgeBench, WinoGrande, BBH).
# Families behind benchmarks it already does well on get weight 0.
FOCUS = {"grounded_pairwise": 4, "pronoun_resolution": 3, "pairwise_answer_quality": 2, "logical_deduction": 2,
         "rule_evaluation": 2, "argument_validity": 2, "event_order": 2, "sarcasm": 1, "numeric_comparison": 1,
         "object_tracking": 1, "date_arithmetic": 1, "navigate": 1, "boolean_expressions": 1, "adjective_order": 1,
         "coloured_objects": 1}
# FOCUS plus a small share of faithfulness data: the first focused run lifted BBH, JudgeBench and WinoGrande but
# RAGTruth slipped 3 points when these families were dropped entirely.
FOCUS_V2 = {**FOCUS, "response_faithfulness": 1, "summary_consistency": 1}
# After the panel-shape review: long JudgeBench-style comparisons and the BBH sub-tasks nothing covered yet. RAGTruth and
# WinoGrande now come from their own training splits, so their synthetic families are left out.
FOCUS_V3 = {"long_pairwise": 4, "causal_judgement": 2, "translation_error": 2, "disambiguation": 1, "temporal_sequences": 1,
            "penguins_table": 1, "grounded_pairwise": 1, "logical_deduction": 1, "argument_validity": 1, "sarcasm": 1}
# FOCUS_V3 without its two low-yield families: long pairwise moved to jeff.sampled, disambiguation kept only 22%.
FOCUS_V4 = {"causal_judgement": 2, "translation_error": 2, "grounded_pairwise": 1, "logical_deduction": 1,
            "argument_validity": 1, "sarcasm": 1}
# JudgeBench-shaped data (it has no training split of its own): pairwise "which response is better" comparisons, plus a
# little of the two weakest BBH families. WinoGrande-style data now comes from its full 40k-row training split instead.
FOCUS_V5 = {"grounded_pairwise": 3, "pairwise_answer_quality": 3, "causal_judgement": 1, "translation_error": 1}
# Only the code-built BBH-shaped families: instant (no teacher calls), exact labels, as many as wanted.
BBH_CODE = {"tracking_five": 2, "date_understanding": 2, "ordering_five": 2, "ordering_seven": 2, "navigate_turns": 1,
            "colour_counting": 1, "penguins_table": 2, "formal_fallacies": 2}
# Texts per teacher call; long families get fewer so a reply fits the output limit.
BATCH_SIZE = {"long_pairwise": 3}


class Slot(TypedDict):
    id: str
    family: str
    label: str
    domain: str
    length: str
    difficulty: str
    condition: str
    topic: NotRequired[str]  # Material fields, added by materials.dress for real runs.
    people: NotRequired[str]
    organisation: NotRequired[str]
    question: NotRequired[str]  # Grounded pairwise: a question with known right and wrong answers.
    correct_answer: NotRequired[str]
    wrong_answer: NotRequired[str]


def make_slots(count: int, seed: int, weights: dict[str, int] | None = None) -> list[Slot]:
    """Slots cycle through the families; with weights, a family with weight w appears w times per cycle and weight 0
    leaves it out. Labels cycle evenly within each family, except that 'skewed' slots of non-positional families always
    take the family's first label. Without weights every family has weight 1 (the original even plan)."""
    weights = weights if weights is not None else {family.name: 1 for family in FAMILIES}
    unknown = set(weights) - set(BY_NAME)
    if unknown:
        raise ValueError(f"Unknown families in weights: {sorted(unknown)}")
    cycle = [family for round_ in range(max(weights.values())) for family in FAMILIES if weights.get(family.name, 0) > round_]
    if not cycle:
        raise ValueError("Every family has weight 0")
    counters: dict[str, int] = {}
    slots: list[Slot] = []
    for index in range(count):
        family = cycle[index % len(cycle)]
        n = counters.get(family.name, 0)
        counters[family.name] = n + 1
        condition = CONDITIONS[n % len(CONDITIONS)]
        if family.name in POSITIONAL:  # every slot cycles, so each position is equally common
            label = family.labels[n % len(family.labels)]
        elif condition == "skewed":
            label = family.labels[0]
        else:  # cycle over the non-skewed slots only (four in every five)
            balanced_index = (n // len(CONDITIONS)) * (len(CONDITIONS) - 1) + n % len(CONDITIONS)
            label = family.labels[balanced_index % len(family.labels)]
        rng = random.Random(f"{seed}-{index}")
        slots.append({"id": f"syn-{seed}-{family.name}-{n:06d}", "family": family.name, "label": label,
                      "domain": rng.choice(DOMAINS), "length": rng.choice(LENGTHS),
                      "difficulty": rng.choice(DIFFICULTIES), "condition": condition})
    return slots


def batches(slots: list[Slot], size: int = 8) -> list[list[Slot]]:
    ordered = sorted(slots, key=lambda slot: (slot["family"], slot["id"]))
    result: list[list[Slot]] = []
    for family, group in groupby(ordered, key=lambda slot: slot["family"]):
        members = list(group)
        step = BATCH_SIZE.get(family, size)
        result.extend(members[start:start + step] for start in range(0, len(members), step))
    return result
