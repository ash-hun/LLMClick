# LLMClick API 명세

| 항목 | 값 |
|---|---|
| Last Updated | 2026-10-04 |
| Base URL | `http://localhost:8000` |
| 데이터 포맷 | JSON (UTF-8) |
| 인증 | 없음 |

> `Last Updated`는 **문서 내용이 실제로 바뀐 날**만 적는다. 코드가 그대로인 채로 문서를
> 다시 생성했다면 이 값도 그대로다. 시각은 적지 않는다.

---

## 목차

1. [개요](#개요)
2. [공통 규약](#공통-규약)
3. [세부 API 명세](#세부-api-명세)
   - [POST /api/config/load](#post-apiconfigload)
   - [POST /api/config/validate](#post-apiconfigvalidate)
   - [GET /api/jobs](#get-apijobs)
   - [POST /api/jobs](#post-apijobs)
   - [GET /api/jobs/{job_id}](#get-apijobsjob_id)
   - [GET /api/system/recipes](#get-apisystemrecipes)
   - [GET /health](#get-health)
4. [데이터 모델](#데이터-모델)

---

## 개요

YAML 설정 하나를 커스텀 모델 하나로 실행하는 서버. config 의 `pipeline.recipe` 가 레시피(서브모델)를 고르고,
`pipeline.stages` 가 수행 단계를 고른다. 정본은 `output/_stages/<stage>-<fingerprint>/` 의 스테이지 산출물과
`output/<name>-<hash>/manifest.json` 이며, API 는 파이프라인(`core/pipeline.py`)을 백그라운드 워커 하나로 실행하고
상태와 진행률을 돌려준다. 라우터: `core/api/routers/`.

---

## 공통 규약

**에러 응답**

```json
{ "detail": "사람이 읽을 수 있는 메시지" }
```

| 상태 코드 | 언제 |
|---|---|
| 200 | 정상 |
| 202 | Job 접수 (이미 있는 Job 이면 그 Job 을 그대로 반환) |
| 404 | config 파일 또는 job_id 없음 |
| 422 | 요청 본문 또는 config 가 스키마에 안 맞음 (Pydantic 메시지가 `detail`) |
| 500 | 서버 처리 실패 |

**페이지네이션** — 없음. Job 은 프로세스 메모리에 있고 개수가 작다.

**공통 헤더** — 없음.

---

## 세부 API 명세

경로 알파벳순.

### POST /api/config/load

서버 안의 YAML 파일을 읽어 검증하고 실험 키를 돌려준다.

#### Endpoint

```
POST /api/config/load?config_path=configs/llm/sft.yaml
```

#### 호출 규약

| 항목 | 값 |
|---|---|
| Method | `POST` |
| 인증 | 없음 |
| 요청 Content-Type | 없음 (본문 없음) |
| 응답 Content-Type | `application/json` |
| 필수 헤더 | 없음 |

#### Request Body

없음.

#### Query / Path 파라미터

| 파라미터 | 위치 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---|---|---|---|
| `config_path` | query | string | 예 | — | 서버 작업 디렉토리 기준 YAML 경로 |

#### Response Body

[ConfigSummary](#configsummary).

#### 응답 코드

| 코드 | 의미 | 본문 |
|---|---|---|
| `200` | 정상 | ConfigSummary |
| `404` | 파일 없음 | `{"detail": "No such config: ..."}` |
| `422` | 스키마 위반 | `{"detail": "..."}` |

#### 호출 예시

```bash
$ curl -s -X POST 'http://localhost:8000/api/config/load?config_path=configs/llm/sft.yaml'
{"valid":true,"recipe":"llm_sft","experiment":"sft-qwen3.5-0.8b-<hash>","directory":"output/sft-qwen3.5-0.8b-<hash>","stages":["data","train","validate"],"config":{...}}
$ curl -s -X POST 'http://localhost:8000/api/config/load?config_path=configs/none.yaml'
{"detail":"No such config: configs/none.yaml"}
```

#### 특이사항

**재호출** — 상태를 바꾸지 않음. 파일만 읽는다.

---

### POST /api/config/validate

본문으로 받은 config 를 검증한다. YAML 과 같은 구조(`pipeline` 섹션 포함)를 JSON 으로 보낸다.

#### Endpoint

```
POST /api/config/validate
```

#### 호출 규약

| 항목 | 값 |
|---|---|
| Method | `POST` |
| 인증 | 없음 |
| 요청 Content-Type | `application/json` |
| 응답 Content-Type | `application/json` |
| 필수 헤더 | 없음 |

#### Request Body

[Config](#config) 와 같은 구조의 객체. `pipeline` 섹션은 최상위로 병합된다.

#### Query / Path 파라미터

| 파라미터 | 위치 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---|---|---|---|
| — | — | — | — | — | 없음 |

#### Response Body

[ConfigSummary](#configsummary).

#### 응답 코드

| 코드 | 의미 | 본문 |
|---|---|---|
| `200` | 정상 | ConfigSummary |
| `422` | 스키마 위반 | `{"detail": "..."}` |

#### 호출 예시

```bash
$ curl -s -X POST http://localhost:8000/api/config/validate -H 'content-type: application/json' \
  -d '{"pipeline":{"recipe":"jev","name":"x"},"model":{"backbone":"nope","name":"m","revision":"0000000000000000000000000000000000000000"}}'
{"detail":"1 validation error for JevConfig\n  Value error, Unknown backbone 'nope'; registered: ['gemma4', 'modernbert', 'qwen3_5'] ..."}
$ curl -s -X POST http://localhost:8000/api/config/validate -H 'content-type: application/json' -d '{"pipeline":{"name":"x"}}'
{"detail":"pipeline.recipe is None; registered: ['embedding_contrastive', 'jev', 'llm_dpo', 'llm_grpo', 'llm_instruction', 'llm_sft']"}
```

#### 특이사항

**재호출** — 상태를 바꾸지 않음.

---

### GET /api/jobs

접수된 모든 Job.

#### Endpoint

```
GET /api/jobs
```

#### 호출 규약

| 항목 | 값 |
|---|---|
| Method | `GET` |
| 인증 | 없음 |
| 요청 Content-Type | 없음 (본문 없음) |
| 응답 Content-Type | `application/json` |
| 필수 헤더 | 없음 |

#### Request Body

없음.

#### Query / Path 파라미터

| 파라미터 | 위치 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---|---|---|---|
| — | — | — | — | — | 없음 |

#### Response Body

[JobResponse](#jobresponse) 배열.

#### 응답 코드

| 코드 | 의미 | 본문 |
|---|---|---|
| `200` | 정상 | 배열 (비어 있을 수 있음) |

#### 호출 예시

```bash
$ curl -s http://localhost:8000/api/jobs
[]
```

#### 특이사항

**재호출** — 상태를 바꾸지 않음. 서버 재시작 후에는 비어 있다(메모리 저장).

---

### POST /api/jobs

config 로 파이프라인을 백그라운드에서 실행한다. 수행 단계는 config 의 `pipeline.stages` 로 정한다(없으면 전부).

#### Endpoint

```
POST /api/jobs
```

#### 호출 규약

| 항목 | 값 |
|---|---|
| Method | `POST` |
| 인증 | 없음 |
| 요청 Content-Type | `application/json` |
| 응답 Content-Type | `application/json` |
| 필수 헤더 | 없음 |

#### Request Body

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---|---|---|
| `config_path` | string | `config` 와 택일 | null | 서버 안 YAML 경로 |
| `config` | object | `config_path` 와 택일 | null | 인라인 config (YAML 과 같은 구조) |

#### Query / Path 파라미터

| 파라미터 | 위치 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---|---|---|---|
| — | — | — | — | — | 없음 |

#### Response Body

[JobResponse](#jobresponse).

#### 응답 코드

| 코드 | 의미 | 본문 |
|---|---|---|
| `202` | 접수 (또는 같은 Job 이미 존재) | JobResponse |
| `404` | config 파일 없음 | `{"detail": "..."}` |
| `422` | 본문·config 스키마 위반 | `{"detail": "..."}` |

#### 호출 예시

```bash
$ curl -s -X POST http://localhost:8000/api/jobs -H 'content-type: application/json' \
  -d '{"config_path":"configs/llm/sft.yaml"}'
{"job_id":"sft-qwen3.5-0.8b-<hash>-<stages>","status":"pending","recipe":"llm_sft","experiment":"sft-qwen3.5-0.8b-<hash>","directory":"output/sft-qwen3.5-0.8b-<hash>","stages":["data","train","validate"],"progress":{"experiment":null,"stages":{},"current":null,"done":0,"total":null,"note":""},"result":null,"error":null}
```

#### 특이사항

**재호출** — 같은 결과, 상태 변화 없음. `job_id` 는 `실험 키 + 스테이지 계획 해시` 이므로 같은 config 를 다시 보내면
새 작업을 만들지 않고 기존 Job(대기, 실행, 완료)을 돌려준다. 실패한 Job 은 재호출 시 다시 실행되며, 파이프라인은
완료된 스테이지를 지문으로 건너뛰므로 실패 지점부터 이어진다. validate 스테이지가 기준 미달로 실패한 Job 은
재호출해도 같은 이유로 실패한다(config 의 `validation` 기준이나 학습 설정을 바꿔야 한다).

동시성: Job 은 워커 하나가 접수 순서대로 실행한다(학습 두 개가 가속기를 나눠 쓰지 않도록). 스테이지 디렉토리는
파일 잠금으로 보호되므로 CLI 와 서버가 같은 스테이지를 동시에 요청하면 한쪽이 기다렸다가 결과를 재사용한다.

---

### GET /api/jobs/{job_id}

Job 하나의 상태·결과.

#### Endpoint

```
GET /api/jobs/{job_id}
```

#### 호출 규약

| 항목 | 값 |
|---|---|
| Method | `GET` |
| 인증 | 없음 |
| 요청 Content-Type | 없음 (본문 없음) |
| 응답 Content-Type | `application/json` |
| 필수 헤더 | 없음 |

#### Request Body

없음.

#### Query / Path 파라미터

| 파라미터 | 위치 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---|---|---|---|
| `job_id` | path | string | 예 | — | `POST /api/jobs` 가 돌려준 값 |

#### Response Body

[JobResponse](#jobresponse). `status` 가 `done` 이면 `result` 에 스테이지별 출력, `failed` 면 `error` 에 traceback.

#### 응답 코드

| 코드 | 의미 | 본문 |
|---|---|---|
| `200` | 정상 | JobResponse |
| `404` | 없음 | `{"detail": "No such job: ..."}` |

#### 호출 예시

```bash
$ curl -s http://localhost:8000/api/jobs/nope
{"detail":"No such job: nope"}
```

#### 특이사항

**재호출** — 상태를 바꾸지 않음.

---

### GET /api/system/recipes

이 서버가 만들 수 있는 레시피와, 레시피별 스테이지 순서 및 YAML 의 `name:` 키로 쓸 수 있는 값 목록.

#### Endpoint

```
GET /api/system/recipes
```

#### 호출 규약

| 항목 | 값 |
|---|---|
| Method | `GET` |
| 인증 | 없음 |
| 요청 Content-Type | 없음 (본문 없음) |
| 응답 Content-Type | `application/json` |
| 필수 헤더 | 없음 |

#### Request Body

없음.

#### Query / Path 파라미터

| 파라미터 | 위치 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---|---|---|---|
| — | — | — | — | — | 없음 |

#### Response Body

레시피 키(`pipeline.recipe` 값)마다 객체 하나. 모든 레시피에 `stages` 가 있고, 나머지 필드는 레시피가 정한다.

LLM, Embedding 계열 레시피(`llm_sft`, `llm_instruction`, `llm_dpo`, `llm_grpo`, `embedding_contrastive`)의 필드:

| 필드 | 타입 | 널 허용 | 설명 |
|---|---|---|---|
| `stages` | string[] | 아니오 | `["data","train","validate"]` |
| `architecture` | string[] | 아니오 | `model.architecture` 에 쓸 수 있는 모델 카탈로그 키 |
| `source` | string[] | 아니오 | `data.sources[*].name` 에 쓸 수 있는 키 |
| `method_keys` | string[] | 아니오 | `method` 섹션에 쓸 수 있는 키 |
| `reward` | string[] | 아니오 | `llm_grpo` 에만 있음. `method.reward.name` 에 쓸 수 있는 키 |

`jev` 의 필드:

| 필드 | 타입 | 널 허용 | 설명 |
|---|---|---|---|
| `stages` | string[] | 아니오 | 스테이지 이름, 실행 순서 |
| `builder` | string[] | 아니오 | 데이터 빌더 키 |
| `converter` | string[] | 아니오 | 원시 행 → Example 변환기 키 |
| `backbone` | string[] | 아니오 | 백본 패밀리 키 |
| `teacher` | string[] | 아니오 | 합성 데이터 교사 키 |
| `benchmark` | string[] | 아니오 | 평가셋 키 |

#### 응답 코드

| 코드 | 의미 | 본문 |
|---|---|---|
| `200` | 정상 | 위 스키마 |

#### 호출 예시

```bash
$ curl -s http://localhost:8000/api/system/recipes
{"embedding_contrastive":{"stages":["data","train","validate"],"architecture":["bi_encoder"],"source":["huggingface","local_jsonl"],"method_keys":["query_instruction","temperature"]},"jev":{"stages":["data","benchmarks","synthetic","mix","train","validate","evaluate"],"builder":[...],"converter":[...],"backbone":[...],"teacher":[...],"benchmark":[...]},"llm_dpo":{"stages":["data","train","validate"],"architecture":["hybrid","transformer"],"source":["huggingface","local_jsonl"],"method_keys":["beta"]},"llm_grpo":{...,"method_keys":["group_size","max_new_tokens","reward","temperature"],"reward":["contains","exact_match","last_number"]},"llm_instruction":{...,"method_keys":["system"]},"llm_sft":{...,"method_keys":[]}}
```

#### 특이사항

**재호출** — 상태를 바꾸지 않음.

---

### GET /health

서버 생존 확인.

#### Endpoint

```
GET /health
```

#### 호출 규약

| 항목 | 값 |
|---|---|
| Method | `GET` |
| 인증 | 없음 |
| 요청 Content-Type | 없음 (본문 없음) |
| 응답 Content-Type | `application/json` |
| 필수 헤더 | 없음 |

#### Request Body

없음.

#### Query / Path 파라미터

| 파라미터 | 위치 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---|---|---|---|
| — | — | — | — | — | 없음 |

#### Response Body

| 필드 | 타입 | 널 허용 | 설명 |
|---|---|---|---|
| `status` | string | 아니오 | 항상 `"ok"` |

#### 응답 코드

| 코드 | 의미 | 본문 |
|---|---|---|
| `200` | 정상 | `{"status":"ok"}` |

#### 호출 예시

```bash
$ curl -s http://localhost:8000/health
{"status":"ok"}
```

#### 특이사항

**재호출** — 상태를 바꾸지 않음.

---

## 데이터 모델

### ConfigSummary

| 필드 | 타입 | 널 허용 | 설명 |
|---|---|---|---|
| `valid` | bool | 아니오 | 항상 true (실패는 422) |
| `recipe` | string | 아니오 | config 의 `pipeline.recipe` |
| `experiment` | string | 아니오 | `<name>-<config sha256[:8]>`. `stages`, `output_dir`, `tracker` 는 해시에 넣지 않는다 |
| `directory` | string | 아니오 | `output_dir/experiment` |
| `stages` | string[] | 아니오 | 실행하거나 재사용할 스테이지, 순서대로. 요청한 스테이지가 읽는 상위 스테이지와 `train` 에 따라붙는 `validate` 포함 |
| `config` | object | 아니오 | 기본값이 채워진 정규화 config |

### JobResponse

| 필드 | 타입 | 널 허용 | 설명 |
|---|---|---|---|
| `job_id` | string | 아니오 | `<experiment>-<stages sha256[:6]>` |
| `status` | string | 아니오 | `pending` / `running` / `done` / `failed` |
| `recipe` | string | 아니오 | 레시피 키 |
| `experiment` | string | 아니오 | 실험 키 |
| `directory` | string | 아니오 | 실험 디렉토리 |
| `stages` | string[] | 아니오 | 스테이지 계획 (ConfigSummary 의 `stages` 와 같음) |
| `progress` | object | 아니오 | [Progress](#progress) |
| `result` | object | 예 | `done` 일 때 `{"experiment","directory","stages":{stage: outputs}}` |
| `error` | string | 예 | `failed` 일 때 메시지 + traceback |

### Progress

| 필드 | 타입 | 널 허용 | 설명 |
|---|---|---|---|
| `experiment` | string | 예 | 실행이 시작되기 전에는 null |
| `stages` | object | 아니오 | 스테이지 이름 → `pending` / `running` / `done` / `cached` / `failed` |
| `current` | string | 예 | 실행 중인 스테이지. 대기 중이거나 끝났으면 null |
| `done` | int | 아니오 | 현재 스테이지 안에서 끝난 단위 (학습이면 업데이트 스텝) |
| `total` | int | 예 | 현재 스테이지의 전체 단위. 스테이지가 알리지 않으면 null |
| `note` | string | 아니오 | 현재 스테이지의 부가 정보 (예: `loss 0.4312`, 처리 중인 벤치마크 이름) |

### Config

공통 키는 `core/config/schema.py` 의 `BaseConfig`(recipe, name, seed, output_dir, device, stages), 모델링 공통 키는
`modeling/config.py` 의 `ModelingConfig`(tracker, validation), 레시피 고유 섹션은 레시피의 스키마가 정본이다.
`jev` 는 `modeling/jev/config.py`: `checkpoint`, `data`(builders, folds, synthetic, mix), `model`(backbone, name,
revision, prompt_layout), `training`(jeff.train 인자와 동명), `validation`(min, max, batch_size),
`evaluation`(benchmarks, batch_size). `builders[*]`, `benchmarks[*]`, `teacher` 는 `name` + 임의 파라미터.
스키마에 없는 최상위 키는 422.

LLM, Embedding 계열 레시피는 `modeling/tuning/config.py` 의 `TuningConfig` 를 공유한다: `model`(architecture, name,
revision, template), `data`(sources, validation), `training`(epochs, lr, weight_decay, batch_size, accumulation,
warmup_ratio, max_grad_norm, max_length, max_steps, resume_every, adapter), `validation`(min, max, batch_size), `method`.
`training.adapter`(name, r, alpha, dropout, targets)가 있으면 LoRA 로 학습하고, 없으면 전체 가중치를 학습한다.
`method` 섹션의 키는 레시피마다 다르며 `modeling/llm/config.py`, `modeling/embedding/config.py` 가 정본이다.
`validation.min/max` 에 쓸 수 있는 지표도 레시피마다 다르다: `llm_sft` 와 `llm_instruction` 은 `loss`,
`perplexity`, `llm_dpo` 는 `accuracy`, `margin`, `llm_grpo` 는 `reward`, `embedding_contrastive` 는 `accuracy`, `mrr`.
