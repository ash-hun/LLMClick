---
title: Jarvis inner-thought router
description: A Jev decision model that decides, for every request the agent gets, whether it also runs inner thought; data from gpt-6.1-sol through data_synthetic, trained with llm_decision_sft.
---

# Experiment: an inner-thought router for the Jarvis agent

The agent always replies. For each request it also has to decide whether to run inner thought on it, which costs
time and tokens, or to answer from what it already has. This experiment builds that decision as a small Jev
decision model: two options, `direct` and `think`, read by a pointer head on Qwen3.5-0.8B, with up to two earlier
turns of the conversation in view. The data comes from a teacher model through `data_synthetic`; the router is
trained with `llm_decision_sft` and measured with `evaluation_decision`.

```
user request (+ 2 earlier turns) -> [router: direct | think] -> reply, and inner thought when `think`
```

## Results

| Router | Held-out (167) | Boundary set (19) | Boundary, think at P(think) >= 0.3 | ECE held-out | Seconds a request (p50 / p95) | Training |
|---|---|---|---|---|---|---|
| **Qwen3-0.6B** | **166 (99.4%)** | 17 (89.5%) | **19 (100%)** | 0.016 | **0.08 / 0.12** | 10 min |
| Qwen3.5-0.8B | 166 (99.4%) | 18 (94.7%) | 18 (94.7%) | 0.006 | 0.30 / 0.41 | 25 min |

Qwen3-0.6B is the router: as accurate on the held-out rows, a quarter of the latency, and its mistakes are unsure
ones (P(think) about 0.35 where it said `direct` to a `think`), which a threshold of 0.3 on P(think) turns right
without sending more than one held-out `direct` request to thought. Qwen3.5-0.8B's one boundary mistake is sure
(0.997), so no threshold reaches it. The threshold was read off the same 19 requests it is scored on: take 100% as
the best case, not the expectation.

Latency is one request at a time on an Apple M4 Pro (MPS), the router's checkpoint loaded once.

## The decision and its criteria

The options and what they mean were settled with the user in two rounds before any data was made.

| Question | Decision | Why |
|---|---|---|
| What separates the options | whether the agent runs inner thought on the request; a reply is given either way | the router gates thought, not the reply |
| A lookup the assistant can fetch as it is (the weather, a score) | `direct` | slow, but nothing is worked out; latency alone is not the criterion |
| A question that weighs several sources | `think` | the sources have to be reconciled |
| A request where it is unclear what the user means | `direct` | the right reply is to ask back at once |
| Long or creative writing | `think` | it has to be composed |
| Conversation in view | the two turns before the request (user and assistant), possibly none | a short follow-up ("go ahead") is `think` or `direct` depending on what came before |
| Languages | English and Korean | the agent is used in both |

The meanings the router reads with every request (`criteria` of the question, also the teacher's routing prompt):

- `direct`: No inner thought: the reply needs nothing worked out first. Small talk, a known fact, a single command or
  lookup, a one-step answer, or asking back when it is unclear what the user means.
- `think`: Inner thought: a good reply needs working out first, in several steps. Calculation, planning,
  comparison, analysis, weighing sources, a judgement call, or long and creative writing.

The state the router reads, and the agent must build the same way: the earlier turns, a blank line, the request.

```
User: <earlier request>
Assistant: <its reply>

<the request to route>
```

## Step by step

### 1. The data channel learns to make decision rows

`data_synthetic` assembled rows for `sft`, `dpo` and `grpo` only. Making this data needed three additions and
uncovered one bug:

- **Bug**: `verify.schema_recipe: llm_decision_sft` assembled every candidate as a chat (`messages`), which a
  decision method rejects (`KeyError: 'state'`): every row would have been dropped. It now assembles the decision
  row it will become.
- `labelled_requests` generator: a seed's `kind` is the option and its text a situation; the teacher sees every
  option's meaning and writes requests that belong to this one, as a JSON array with the conversation before each.
  `languages` and `history` (turns before the request) turn by the prompt's index, so `per_seed: 6` covers English
  and Korean with 0, 1 and 2 earlier turns once each.
- `select.assemble: decision` with `select.decision` (the question's name, instructions and criteria): one Jev row
  per request, the label being the situation's option.
- The OpenAI teacher learned reasoning models (`reasoning_effort`): they take `max_completion_tokens`, which also
  covers their reasoning, and no temperature.

Labels are not trusted from the generator alone: the teacher routes every request again in `respond`, three times,
without seeing the situation, and `verify` keeps a request only when its routes agree with its situation's option
(`reward: exact_match`) and with each other (`majority: exact`). Disagreements are the boundary cases; they are kept
in `rejected.jsonl` for review rather than trained on.

### 2. Choosing the teacher

The key could reach 80 chat models. Each candidate routed the same 20 requests, labelled by hand from the criteria
above, with the routing prompt of the config (reasoning models at `reasoning_effort: low`):

| Model | Agreed | Disagreed on | Price per million tokens (in / out) |
|---|---|---|---|
| **gpt-6.1-sol** | **19/20** | summarize a short abstract | 2 / 10 |
| gpt-6-astra | 19/20 | summarize a short abstract | 10 / 50 |
| gpt-4.1 | 18/20 | is 1001 prime; summarize a short abstract | |
| gpt-5.5 | 17/20 | autumn poem; "go ahead" after a planning offer; abstract | |
| gpt-6-sol | 17/20 | is 1001 prime; autumn poem; abstract | |
| gpt-6-luna | 16/20 | 1001; poem; flat sourdough; abstract | |
| gpt-4.1-mini | 15/20 | 1001; poem; rambling message with a simple ask; abstract; "then plan day 3" | 0.4 / 1.6 |
| gpt-5.4 | 15/20 | 1001; poem; slow code; sourdough; abstract | |
| gpt-5.4-mini | 13/20 | seven, one of them an empty reply | |

gpt-6.1-sol was chosen: as accurate as the flagship at a fifth of its price. Every model routed "summarize the
contribution of this abstract" (two sentences) as `direct`; the hand label `think` is probably what was wrong: a
summary of a short text needs nothing worked out. Prices are from OpenAI's announcement of gpt-6.1-sol and
third-party price tables (October 2026).

### 3. Generating the requests

19 situations, 9 for `direct` and 10 for `think`; the two marked as traps look like the other option:

| Option | Situations |
|---|---|
| `direct` | greetings and small talk; a well-known fact; controlling a device or the app; looking up one thing; a quick conversion; translating one sentence; an unclear request (ask back); a short follow-up needing a short reply; **a long, rambling message whose request is simple** |
| `think` | a multi-step calculation; planning under constraints; comparing options; code and design; summarizing or analyzing a long text; weighing sources; advice on a personal decision; long or creative writing; a short follow-up that starts a task needing thought; **a short question that needs real reasoning** |

Per situation 6 prompts of 10 requests: 114 teacher calls, all answered with 10 parseable examples, so 1140
requests: 570 English and 570 Korean, 380 each with 0, 1 and 2 earlier turns, 540 `direct` and 600 `think`. The
median request has 15 words; the median earlier conversation 136 characters. 26,932 input and 126,465 output tokens
(reasoning included): 1.32 USD, 56 minutes.

An example with two earlier turns, made for "a short follow-up that starts a task needing thought":

```
User: 월급이 세후 320만 원이고 월세 75만 원, 대출 상환 40만 원, 생활비 90만 원 정도 써. 1년 안에 비상금 1천만 원을 모으고 싶어.
Assistant: 현재 지출만 보면 매달 115만 원이 남습니다. 기존 저축액과 앞으로 예정된 큰 지출도 알려주시면 월별 계획을 세울 수 있어요.
User: 모아둔 건 100만 원이고, 9월에 이사비로 150만 원 나가. 지금은 1월 초야.
Assistant: 1월부터 12월까지의 저축 가능액과 9월 이사비를 반영해 목표 달성 여부를 계산하고, 월별 적립 계획을 만들 수 있습니다.

응, 월별로 짜줘.
```

### 4. Routing and filtering

The teacher routed each of the 1140 requests three times, not seeing its situation: 3420 calls, 748,038 input and
13,970 output tokens, 1.64 USD, 80 minutes (about 1.4 s a call, one at a time). The whole data run cost 2.95 USD.

| Step | Requests | Note |
|---|---|---|
| generated | 1140 | |
| all three routes agree with the situation | 1112 | 97.5% |
| most routes agree (kept by `verify`) | 1121 | 66 of 3420 routes rejected: 61 against the situation, 5 against the majority |
| after near-duplicates (4-grams, 0.75) | **1116** | 5 near-duplicates; no exact duplicates across requests |

The kept rows: 532 `direct`, 584 `think`; English 555 and Korean 561; 364, 376 and 376 with 0, 1 and 2 earlier
turns. Per situation, all 60 survived in 12 of 19; the losses:

| Situation | Kept | What the teacher did |
|---|---|---|
| summarizing or analyzing a document (`think`) | 50/60 | routed 10 `direct`: in each, the document is referred to but not there ("analyze the attached paper", "이 보고서를 요약하고") |
| a short question that needs real reasoning (`think`) | 56/60 | `direct` for "why did all three rows change" (a known Python pitfall) and for wording an apology email |
| a short follow-up that starts a task needing thought (`think`) | 58/60 | `direct` where, with no earlier turns, "now challenge that argument" refers to nothing |
| looking up one thing (`direct`) | 58/60 | `think` for two lookups that follow a conversation (sunset in Denver, a library's hours) |
| a short follow-up needing a short reply (`direct`) | 58/60 | twice it answered "A little louder, please" by repeating the earlier reply instead of routing |
| a well-known fact, controlling a device (`direct`) | 58/60 each | duplicates of other requests |

The largest loss is a boundary the criteria did not settle: **a request about material that is not in the
conversation**. The teacher treats it as asking back (`direct`), which matches the criteria as written ("asking back
when it is unclear what the user means"), while the situation meant `think`. The rows are dropped, not mislabelled,
so training is unaffected; the router will meet such requests in use, and its answer on them is checked below.

Two more decisions came out of the review, both to be settled with the user:

- A summary of a short text in view (the abstract in the teacher comparison) reads as `direct` to every model.
- In the agent, attachments may reach the router only as text in the turns; if they do not, every "analyze the
  attached ..." looks like this boundary.

### 5. Training

`llm_decision_sft` with a pointer head and LoRA (r 16), no reasoning in training (`think_fraction: 0`), 15% of the
1116 rows held out for `validate` (949 left, of which 15% calibrate the temperature), 2 epochs, lr 5e-5, head lr
1e-4. The requests are short: with the question and both options, the median input is 164 tokens and the longest 379.

**The first run stalled.** At `batch_size: 16` on Qwen3.5-0.8B a step took 7 minutes (the first three steps:
02:31 to 02:48), which would have been 14 hours. The process waited on the GPU (`MTLCommandBuffer
waitUntilCompleted`) while the Mac swapped: training held 32 GB and a virtual machine 24 GB of the 48 GB. Neither the
input length nor Qwen3.5's slow fallback kernels (no flash-linear-attention on a Mac) was the cause. Timed again, 8
steps each, at `batch_size: 4`:

| Backbone | Seconds a step | Peak memory | Validation (167 rows) |
|---|---|---|---|
| Qwen3-0.6B (`transformer`) | 2 to 4 | 12 GB | 20 s |
| Qwen3.5-0.8B (`hybrid`) | about 10 | 19 GB | 64 s |

The config now uses `batch_size: 4`. The router runs on every request, so its latency matters as much as its
accuracy: both backbones are trained on the same rows and compared.

| Backbone | Steps | Mean loss, first 20 steps | Mean loss, last 20 | `validate` (167 held out) | Temperature | Wall time |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 404 | 0.664 | 0.002 | accuracy 0.994, ECE 0.016, NLL 0.020 | 3.46 | 9.6 min |
| Qwen3.5-0.8B | 404 | 0.617 | 0.000 | accuracy 0.994, ECE 0.006, NLL 0.148 | 0.22 | 25 min |

Both fit the training rows (accuracy 1.000 over the last 50 steps) and miss one held-out request each. Calibration
moved them in opposite directions: Qwen3-0.6B came out overconfident and was softened (temperature 3.46), Qwen3.5-0.8B
underconfident and was sharpened (0.22); that is why its NLL is higher while its ECE is lower: the one mistake it
makes it makes with certainty.

### 6. Evaluation

`evaluation_decision` (`configs/evaluation/route.yaml`), `think: "off"`, one request at a time. Two sets: the 167
held-out rows (generated, so the same distribution as training) and `samples/evaluation/route_boundary.jsonl`, the 19 requests
of the teacher comparison routed by hand (the disputed abstract summary left out): traps on both sides, follow-ups
whose route depends on the turns before them, English and Korean.

Mistakes at the default threshold (P(think) >= 0.5):

| Router | Set | Request | Label | Said | Confidence |
|---|---|---|---|---|---|
| Qwen3-0.6B | boundary | "응 진행해" after "I'll draft a two-week plan. Want me to start?" | think | direct | 0.65 |
| Qwen3-0.6B | boundary | "Why does my sourdough come out flat?" | think | direct | 0.66 |
| Qwen3-0.6B | held-out | "30도쯤 되는 주방에서 두 시간 뒀는데, 이스트가 부족했던 걸까 너무 오래 둔 걸까?" (after two turns about bread) | think | direct | 0.69 |
| Qwen3.5-0.8B | boundary | "응 진행해" after the plan offer | think | direct | 0.997 |
| Qwen3.5-0.8B | held-out | "Please revise the itinerary so we spend less without adding more travel time." (no earlier turns) | think | direct | 1.000 |

Both stumble on a short Korean "go ahead" after an English offer, the case gpt-5.5 also got wrong in the teacher
comparison. Moving the threshold on P(think):

| Router | Threshold | Boundary right | Held-out right | Held-out `direct` sent to thought |
|---|---|---|---|---|
| Qwen3-0.6B | 0.5 | 17 | 166 | 0 |
| Qwen3-0.6B | **0.3** | **19** | **166** | 1 |
| Qwen3-0.6B | 0.1 | 19 | 162 | 5 |
| Qwen3.5-0.8B | 0.5 to 0.1 | 18 | 166 | 0 |

An extra `think` costs time; a missed one costs the answer. The agent should route to thought at P(think) >= 0.3.

## What the runs taught

1. A teacher's routes agree with the situation it wrote for 97.5% of the time; the requests it disagreed on mark the
   boundaries the criteria leave open (here, material that is not in the conversation).
2. On a Mac shared with a virtual machine, batch size decides whether training runs at all: 16 swapped (7 minutes a
   step), 4 took 1 to 10 seconds. Look at memory before blaming the kernels.
3. A held-out split of generated data says the router learned the data (99.4% for both); a small set written by hand
   separates the models (17, 18, 19 with a threshold).
4. Calibration matters more than a point of accuracy for a gate: the smaller model's unsure mistakes can be moved
   by a threshold, the larger one's sure mistake cannot.


## Limits

- Requests and labels come from one teacher; the router has learned that teacher's boundary. The 19 hand-routed
  requests are the only check from outside it, and they are few.
- The `think` threshold was chosen on the same 19 requests it is reported on.
- Short follow-ups sit near the threshold: "응 진행해" after the same offered plan scored P(think) 0.353 when the
  earlier request was "Can you plan my study schedule for the exam?" and 0.188 when it was "Can you plan my study
  schedule?". More follow-ups of this kind, in mixed languages, are the first thing the data needs.
- Requests about material that is not in the conversation were dropped, not settled: how the agent passes
  attachments to the router decides how they should be routed.
- The state holds the conversation as text; tool results, attachments and the agent's own inner thought are not in
  it.

## Reproducing

`configs/data/route.yaml` (needs `OPENAI_API_KEY` in `environment/.env`; 2.95 USD), then `configs/llm/route.yaml` and
`configs/evaluation/route.yaml`, each with the previous experiment's key in place of `<hash>`. Gpt-6.1-sol's answers
vary between runs (a reasoning model, no temperature), so a new data run will differ a little.

## Appendix: the 19 requests routed by hand

`samples/evaluation/route_boundary.jsonl` (local), one Jev row each with the question and criteria above. Earlier turns
are shown before the request, separated by a slash.

| # | State | Route |
|---|---|---|
| 1 | Is 1001 a prime number? | `think` |
| 2 | What's the weather in Seoul tomorrow? | `direct` |
| 3 | 그거 해줘 | `direct` |
| 4 | Set a timer for 10 minutes | `direct` |
| 5 | Write a short poem about autumn for my mom's birthday card | `think` |
| 6 | Thanks, that's perfect! | `direct` |
| 7 | I've been really busy this week with work and the kids and honestly I'm exhausted, but anyway, what time is it in Tokyo right now? | `direct` |
| 8 | Compare the iPhone 17 and Pixel 10 cameras and tell me which to buy for low-light photos | `think` |
| 9 | 이 코드 왜 느린지 봐줘: for x in a: if x in b_list: out.append(x) | `think` |
| 10 | User: Can you plan my study schedule for the exam? / Assistant: Sure, I'll draft a two-week plan. Want me to start? / 응 진행해 | `think` |
| 11 | User: What's 15% of 80? / Assistant: 12. / say that again | `direct` |
| 12 | 서울 인구가 몇 명이야? | `direct` |
| 13 | Translate 'see you tomorrow' into Japanese | `direct` |
| 14 | 내 월급 350만원인데 적금, 투자, 생활비 어떻게 나누는 게 좋을까? | `think` |
| 15 | Turn on focus mode | `direct` |
| 16 | Why does my sourdough come out flat? | `think` |
| 17 | User: 부산 2박 3일 일정 짜줘 / Assistant: 1일차는 해운대, 2일차는 감천마을과 남포동으로 잡았어요. / User: 좋아, 맛집도 넣어줘 / Assistant: 1일차 저녁은 해운대 암소갈비, 2일차 점심은 남포동 밀면이에요. / 그럼 3일차도 짜줘 | `think` |
| 18 | 오늘 기분 어때? | `direct` |
| 19 | Should I quit my job to start a company? I have 6 months of savings. | `think` |
