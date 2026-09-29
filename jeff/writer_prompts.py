"""One direct writing prompt per task family, sent to the teacher model as its instructions."""

# Every prompt says what to write, the exact layout, what the required label means, and what never to include.
# The slot settings (domain, length, difficulty, condition) are explained once in SETTINGS.

SETTINGS = """Each slot also gives you settings. Follow them:
- domain: the subject area of the text.
- length: how long the text should be.
- difficulty: "easy" means one clear fact decides the answer. "hard" means the reader has to combine several facts,
  notice a small detail, or avoid a tempting wrong answer. The correct answer must still be certain.
- condition: "clean" means plain, well-written text. "noisy" means realistic typos, casual wording or irrelevant details.
  "negated" means a key fact is expressed through a negation (for example "not unhappy", "never arrived").
  "long" means extra realistic but irrelevant detail. "skewed" means write normally.
- topic: build the text around this topic.
- people: a pool of names you may use. Use one or two of them only where people naturally appear; never force a name in.
  Invent any organisation names, places and other details that fit the topic.

For every text: invent new names, places and companies (never reuse a name within the batch), make each text
clearly different from the others, and never mention labels, answers, options by letter, or that this is an example.
Return JSON only."""

WRITER: dict[str, str] = {
    "target_sentiment": """Write a review, message or post about one specific target (a product, place, person or service) and
name that target explicitly in the text.
The required label is the writer's overall attitude toward that target:
- positive: overall favourable. negative: overall unfavourable. mixed: clearly both favourable and unfavourable points.
- neutral: facts only, with no favourable or unfavourable judgement.
Show the attitude through what the writer says. Never name the attitude itself (do not write "overall I'm mixed").""",

    "entailment": """Write two parts:
Premise: <a paragraph of facts>
Hypothesis: <one sentence>
The required label says how the premise relates to the hypothesis:
- entailment: the premise alone makes the hypothesis certainly true.
- contradiction: the premise alone makes the hypothesis certainly false.
- neutral: the premise makes it neither certainly true nor certainly false.
For "hard" slots, the hypothesis should reword or combine premise facts rather than repeat them word for word.""",

    "answerability": """Write two parts:
Passage: <a factual passage>
Question: <one specific question about the passage's subject>
The required label says whether the passage alone answers the question:
- true: the passage contains everything needed for one definite answer.
- false: the passage is on the same subject but lacks a fact needed to answer.
Do not answer the question, and do not say whether the passage answers it.""",

    "response_faithfulness": """Write three parts:
Source: <a document of a few sentences or paragraphs>
Instruction: <a request such as "Summarise the source" or a question about it>
Response: <an answer to the instruction, written as if from the source>
The required label says whether the response has any problem:
- true: the response contains at least one claim that contradicts the source or is not stated in it
  (a changed number, a wrong name, an added detail).
- false: every claim in the response is supported by the source.
Do not point out the problem anywhere in the text.""",

    "pairwise_answer_quality": """Write three parts:
Question: <a question with one correct answer, from maths, science, law, logic or everyday reasoning>
Response A: <an answer with short reasoning>
Response B: <an answer with short reasoning>
The required label names the response that is correct: A or B. The other response must contain a real error
(a wrong step, a wrong fact, a misread condition) that leads to a wrong answer. Both responses should sound confident.
Do not say which response is correct.""",

    "grounded_pairwise": """Each slot gives you a question, its correct_answer and a wrong_answer. Write the text in this layout:
Question: <the question exactly as given, including any lettered options>
Response A: <an answer with its reasoning>
Response B: <an answer with its reasoning>
The response named by required_label must reach correct_answer through sound, correct reasoning. The other response
must reach wrong_answer through reasoning that looks plausible but contains one real mistake (an arithmetic slip, a
misread condition, a wrong fact or a skipped step). Each response ends with its own line "Final answer: <answer>",
using the answers exactly as given. Both responses should sound confident. Do not say which response is correct, and
do not point out the mistake.
Write only the finished text. Never think aloud inside it: no "wait", "let me check", "let's re-read", no talk of
labels, and no sentence about mistakes, common errors or misinterpretations (such as "if one mistakenly..." or "a
common mistake is..."). Each response simply gives its reasoning and its answer as if it were right.""",

    "pronoun_resolution": """Write one or two sentences that name two people or things and contain exactly one blank written as _ that
refers to one of them. Then two lines:
Option 1: <first person or thing>
Option 2: <second person or thing>
The required label (1 or 2) is the option that correctly fills the blank. Common sense about the situation must
decide it; grammar alone must not. Do not fill in the blank.""",

    "logical_deduction": """Write a small ordering puzzle: three to five people or objects and clues about their positions, ranks or times.
The clues must never state the full order directly. The correct statement must only follow by combining at least two
clues; no option may repeat or reword a single clue. Then four lines:
A: <statement about the order>
B: <statement about the order>
C: <statement about the order>
D: <statement about the order>
The required label is the letter of the only statement that follows from the clues. Before answering, check each
statement against the clues: exactly one must follow; the other three must be false or not determined. Do not solve the
puzzle in the text.""",

    "object_tracking": """Write a short scene in which three to five named people each start with a different item; say who starts with what.
Then describe a sequence of two to five swaps between pairs of people, one swap per sentence
(for example "Then Mara and Joel trade items."). Never say who holds what after any swap. Then:
Question: At the end, which item does <one person's name> hold?
A: <item>
B: <item>
C: <item>
D: <item>
The required label is the letter of the correct item.""",

    "date_arithmetic": """Write a short scenario that gives the dates and rules needed to work out one date
(days before or after, business days, weekdays, month lengths or leap years). Do not do the calculation and do not
state the resulting date anywhere in the scenario. Then:
Question: <the date to work out>
A: <date>
B: <date>
C: <date>
D: <date>
The required label is the letter of the correct date. The wrong options should be plausible mistakes
(off by one day, wrong month length).""",

    "rule_evaluation": """Write two parts:
Rules: <a few explicit rules or facts, for example about eligibility, logic gates, or who always tells the truth>
Statement: <one plain claim about a specific case>
The required label says whether the statement is true under the rules: true or false.
Write nothing after the statement: no reasoning, no "therefore", no verdict.""",

    "argument_validity": """Write an argument in everyday language with two or three premises and a conclusion.
The required label says whether the conclusion must follow from the premises:
- valid: the conclusion follows necessarily, even if the premises sound doubtful.
- invalid: the conclusion does not follow necessarily, even if it sounds plausible (for example affirming the consequent).
State the premises and conclusion; do not comment on whether the reasoning is sound.""",

    "policy_application": """Write two parts:
Policy: <a made-up organisation's policy with at least one condition or exception>
Request: <who asks for what, and the relevant facts>
The required label is what the policy says about this request:
- allowed: the facts state everything the policy needs, and they show it permits the request.
- denied: the facts state everything the policy needs, and they show it refuses the request.
- insufficient: exactly one fact the policy needs is not stated, so a careful reader cannot decide.
For allowed and denied, state every fact the policy's conditions depend on. Do not state the decision.""",

    "support_routing": """Write a customer's message to a company. It may mention several things, but it has one main request.
The required label is the department for that main request:
- billing: charges, invoices, payments or refunds. technical: something is broken or hard to use.
- account: login, identity or profile changes. delivery: shipping, tracking or missing packages.
- other: none of these departments.
Do not name the department.""",

    "numeric_comparison": """Write a short text with facts about quantities, possibly in different units that need converting.
End with one line:
Compare: <first quantity> vs <second quantity>
The required label says how the first quantity relates to the second: less, equal, greater, or unknown
(unknown: the facts do not determine it). Do not do the calculation or the unit conversion in the text.""",

    "event_order": """Write a narrative or timeline with several events. Give events full calendar dates, or clock times on one
stated day; never weekdays alone.
End with one line:
Events: <first event> / <second event>
The required label says when the first event happened relative to the second: before, after, same_time, or unknown
(unknown: the text does not determine it). Never state the order of these two events directly;
the reader must work it out from the times or the sequence.""",

    "evidence_attribution": """Write three parts:
Claim: <one specific claim>
Source A: <a paragraph>
Source B: <a paragraph>
The required label says which source, taken on its own, establishes the claim:
source_a (only A), source_b (only B), both (each one alone), or neither (neither alone).
A source establishes the claim only if it states every part of it on its own, including any cause or reason the claim
gives. A source that only hints at the claim, or supports part of it, does not establish it. Do not comment on the sources.""",

    "financial_news_tone": """Write one or two sentences of company or market news in the style of a newswire, about an invented company.
The required label is the likely effect on the company from an investor's point of view:
- positive: likely to improve its finances or share price. negative: likely to harm them.
- neutral: no clear effect (routine announcements, scheduling, personnel with no stated impact).
Report the facts only; do not add an opinion about the effect.""",

    "paraphrase": """Write two lines:
Statement 1: <a factual statement>
Statement 2: <a statement that shares most of its words or ideas>
The required label says whether they make the same factual claim:
- equivalent: same meaning, including scope, numbers and qualifications.
- different: at least one real difference (a number, a scope word like "some" versus "all", a condition, a swapped role).
Do not comment on the statements.""",

    "sarcasm": """Write a short remark, message or social media post, with enough context to judge its tone.
The required label says whether it is sarcastic:
- true: the writer means the opposite of what the words literally say.
- false: the writer means what they say, even if it is enthusiastic, blunt or negative.
Do not mark the tone (no "/s", no "just kidding").""",

    "summary_consistency": """Write two parts:
Document: <a few paragraphs>
Summary: <a few sentences summarising the document>
The required label says whether the summary is fully consistent with the document:
- true: every statement in the summary agrees with the document.
- false: at least one statement in the summary contradicts the document or adds something it does not say.
Do not point out any inconsistency.""",

    "causal_judgement": """Write a short story (one or two paragraphs) in which an outcome happens after several people's actions or
events, then end with one question of the form "Did <a specific person or event> cause <the outcome>?".
The required label is how a typical person would answer that question:
- true: most people would say yes, for example because the person acted deliberately, broke a rule or norm, or their
  action was the unusual one that made the difference.
- false: most people would say no, for example because the action was normal and expected, the outcome would have
  happened anyway, or someone else's deliberate or unusual act was the real cause.
Make the story concrete: who did what, when, and what the rules or usual practice were. Do not answer the question
and do not say what people would think.""",

    "translation_error": """Write two lines, and nothing else:
Source: <one or two sentences in German, in the style of an encyclopedia article about a person, place, organisation or event>
Translation: <an English translation of the source with exactly one error>
First work out a faithful English translation, then change exactly one thing in it. The required label is the kind of
change:
- named_entities: replace a name, place or organisation with a different one (for example "Munich" becomes "Hamburg").
- numerical_values: change a number, ordinal, date or unit (for example "1901" becomes "1910", "third" becomes "fourth").
- modifiers_or_adjectives: change an adjective or modifier describing a noun (for example "imperial" becomes "judicial").
- negation_or_antonyms: add or remove a negation, or turn a comparison into its opposite ("larger" becomes "smaller").
- facts: change a small factual detail that is none of the above (for example "a river" becomes "a lake").
- dropped_content: leave out a significant clause of the source.
The change must be clear to a careful bilingual reader who compares the two lines word by word; everything else must
be a faithful, natural translation. Keep both lines short (one or two sentences each), whatever the length setting says.
Do not mark or explain the error.""",

    "disambiguation": """Write one sentence, in the style of a grammar exercise, that mentions two people by their role (for example
"the patient" and "the specialist", "the chef" and "the customer", "the developer" and "the designer") and then uses one
pronoun (he, she, they, his, her, their) that could grammatically refer to either of them. Then three lines:
A: <what the sentence says if the pronoun refers to the first person, as a short statement, e.g. "The patient had a rash">
B: <the same statement about the second person, e.g. "The specialist had a rash">
C: Ambiguous
The required label is the correct reading:
- A or B: common sense about the roles makes only that reading reasonable (for example, a patient is referred to a
  specialist because the patient is ill).
- C: both readings remain reasonable, so the reference is genuinely ambiguous.
Start the text with "Sentence: ". Use roles, not personal names, keep the sentence under 30 words whatever the length
setting says, and do not explain the reference.""",

    "long_pairwise": """Each slot gives you a question, its correct_answer and a wrong_answer. Write the text in this layout:
Question: <the question exactly as given, including any lettered options>
Response A: <a long, careful answer>
Response B: <a long, careful answer>
Each response must read like a strong assistant's full answer: roughly 250 to 450 words, restating what is asked,
working through the problem step by step with explanations, checking intermediate results, and ending with its own line
"Final answer: <answer>", using the answers exactly as given.
The response named by required_label reaches correct_answer through correct reasoning. The other response reaches
wrong_answer through reasoning that is just as long, confident and well organised, but contains one real mistake
somewhere in the middle that a careful reader can find. Before writing, decide what that mistake is: for example a
misread quantity or condition, an operation applied to the wrong number, a skipped step, a wrong unit conversion, or a
wrong fact. Both responses must read as sincere attempts by someone who believes their answer: neither may mention
mistakes, alternative answers, re-checking or second thoughts, and the wrong one must never mention the correct answer.
Do not say which response is correct and do not point out the mistake.
Write only the finished text. Never think aloud inside it: no "wait", "let me check", "let's re-read", no talk of
labels, and no sentence about mistakes, common errors or misinterpretations (such as "if one mistakenly..." or "a
common mistake is..."). Each response simply gives its reasoning and its answer as if it were right.""",
}
