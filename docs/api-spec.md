# LLMClick API 명세

| 항목 | 값 |
|---|---|
| Last Updated | 2026-10-10 |
| Base URL | `http://localhost:8000` |
| 데이터 포맷 | JSON (UTF-8) |
| 인증 | `API_TOKEN` 이 설정되면 `/api/*` 전부 `Authorization: Bearer <token>`, 비어 있으면 없음 |

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
   - [DELETE /api/jobs/{job_id}](#delete-apijobsjob_id)
   - [GET /api/system/recipes](#get-apisystemrecipes)
   - [GET /health](#get-health)
4. [데이터 모델](#데이터-모델)

---

## 개요

YAML 설정 하나를 커스텀 모델 하나로 실행하는 서버. config 의 `pipeline.recipe` 가 레시피(서브모델)를 고르고,
`pipeline.stages` 가 수행 단계를 고른다. 정본은 `output/_stages/<stage>-<fingerprint>/` 의 스테이지 산출물과
`output/<name>-<hash>/manifest.json` 이며, API 는 파이프라인(`core/pipeline.py`)을 백그라운드 워커(`JOB_WORKERS`, 기본 1)로 실행하고
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
| 202 | Job 접수 (대기 중이거나 실행 중인 같은 Job 이 있으면 그 Job 을 반환), 또는 취소 접수 |
| 401 | `API_TOKEN` 이 설정된 서버에 토큰 없이, 또는 틀린 토큰으로 호출 |
| 403 | `config_path`, 또는 config 가 적은 경로(`output_dir`, `model.name`, `model.init`, `data.sources[*].path`)가 `API_PATHS` 밖 |
| 404 | config 파일 또는 job_id 없음 |
| 409 | 끝난 Job 을 취소하려 함 |
| 422 | 요청 본문 또는 config 가 스키마에 안 맞음 (Pydantic 메시지가 `detail`) |
| 500 | 서버 처리 실패 |

**페이지네이션** — `GET /api/jobs` 의 `limit` 가 최근 N 건으로 줄인다. Job 은 SQLite 테이블(`JOBS_DB`, 기본
`./output/_jobs.sqlite`)에 있고 접수 순으로 돌려준다.

**공통 헤더** — `API_TOKEN` 이 설정된 서버에서는 `/api/*` 전부 `Authorization: Bearer <API_TOKEN>`. `/health` 는 예외.

**경로 제한** — API 가 읽거나 쓰는 모든 경로는 `API_PATHS`(콜론으로 구분한 디렉토리 목록, 기본 `.` 즉 서버의 작업
디렉토리) 안에 있어야 한다. `..` 과 심볼릭 링크를 푼 실제 위치로 판정하며, 밖이면 403. CLI 에는 적용되지 않는다.

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
| 인증 | `API_TOKEN` 설정 시 Bearer |
| 요청 Content-Type | 없음 (본문 없음) |
| 응답 Content-Type | `application/json` |
| 필수 헤더 | `API_TOKEN` 설정 시 `Authorization` |

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
| 인증 | `API_TOKEN` 설정 시 Bearer |
| 요청 Content-Type | `application/json` |
| 응답 Content-Type | `application/json` |
| 필수 헤더 | `API_TOKEN` 설정 시 `Authorization` |

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
  -d '{"pipeline":{"recipe":"llm_sft","name":"x"},"model":{"architecture":"nope","name":"Qwen/Qwen3.5-0.8B","revision":"2fc06364715b967f1860aea9cf38778875588b17"},"data":{"sources":[{"name":"local_jsonl","path":"samples/llm_sft.jsonl"}]}}'
{"detail":"1 validation error for SFTConfig\n  Value error, Unknown architecture 'nope'; registered: ['hybrid', 'transformer'] [type=value_error, input_value={'recipe': 'llm_sft', 'na...mples/llm_sft.jsonl'}]}}, input_type=dict] ..."}
$ curl -s -X POST http://localhost:8000/api/config/validate -H 'content-type: application/json' -d '{"pipeline":{"name":"x"}}'
{"detail":"pipeline.recipe is None; registered: ['embedding_contrastive', 'llm_decision_cispo', 'llm_decision_sft', 'llm_dpo', 'llm_grpo', 'llm_instruction', 'llm_sft']"}
```

#### 특이사항

**재호출** — 상태를 바꾸지 않음.

---

### GET /api/jobs

접수된 Job 목록. 상태로 거르거나 최근 N 건으로 줄일 수 있다.

#### Endpoint

```
GET /api/jobs
```

#### 호출 규약

| 항목 | 값 |
|---|---|
| Method | `GET` |
| 인증 | `API_TOKEN` 설정 시 Bearer |
| 요청 Content-Type | 없음 (본문 없음) |
| 응답 Content-Type | `application/json` |
| 필수 헤더 | `API_TOKEN` 설정 시 `Authorization` |

#### Request Body

없음.

#### Query / Path 파라미터

| 파라미터 | 위치 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---|---|---|---|
| `status` | query | string | 아니오 | 없음 | 이 상태의 Job 만. `pending` / `running` / `done` / `failed` / `cancelled` / `interrupted` 중 하나, 그 외는 422 |
| `limit` | query | int (1 이상) | 아니오 | 없음 | 접수 순으로 마지막 N 건만 |

#### Response Body

[JobResponse](#jobresponse) 배열.

#### 응답 코드

| 코드 | 의미 | 본문 |
|---|---|---|
| `200` | 정상 | 배열 (비어 있을 수 있음) |
| `422` | `status` 가 목록에 없는 값이거나 `limit` 가 1 미만 | `{"detail": "..."}` |

#### 호출 예시

```bash
$ curl -s 'http://localhost:8000/api/jobs?status=running&limit=5'
[]
```

#### 특이사항

**재호출** — 상태를 바꾸지 않음. Job 은 서버를 재시작해도 남는다. 서버가 멈출 때 대기 중이거나 실행 중이던 Job 은
`interrupted` 로 표시되며, 다시 제출하면 이미 만들어진 스테이지를 건너뛰고 이어서 실행된다.

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
| 인증 | `API_TOKEN` 설정 시 Bearer |
| 요청 Content-Type | `application/json` |
| 응답 Content-Type | `application/json` |
| 필수 헤더 | `API_TOKEN` 설정 시 `Authorization` |

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
| `202` | 접수 (대기 중이거나 실행 중인 같은 Job 이 있으면 그 Job) | JobResponse |
| `404` | config 파일 없음 | `{"detail": "..."}` |
| `422` | 본문·config 스키마 위반 | `{"detail": "..."}` |

#### 호출 예시

```bash
$ curl -s -X POST http://localhost:8000/api/jobs -H 'content-type: application/json' \
  -d '{"config_path":"configs/llm/sft.yaml"}'
{"job_id":"sft-qwen3.5-0.8b-<hash>-<stages>","status":"pending","recipe":"llm_sft","experiment":"sft-qwen3.5-0.8b-<hash>","directory":"output/sft-qwen3.5-0.8b-<hash>","stages":["data","train","validate"],"progress":{"experiment":null,"stages":{},"current":null,"done":0,"total":null,"note":""},"result":null,"error":null}
```

#### 특이사항

**재호출** — `job_id` 는 `실험 키 + 스테이지 계획 해시` 이므로 같은 config 를 다시 보내면 같은 `job_id` 를 받는다.
대기 중이거나 실행 중인 Job 이 있으면 그 Job 을 그대로 돌려준다. 완료되었거나 실패한 Job 은 다시 실행된다.
파이프라인이 유효한 스테이지를 지문으로 건너뛰므로, 바뀐 것이 없으면 즉시 같은 결과로 끝나고, 입력 파일이
바뀌었거나 산출물이 지워졌으면 그 스테이지부터 다시 만든다(CLI 를 다시 실행한 것과 같다). validate 스테이지가 기준 미달로 실패한 Job 은
재호출해도 같은 이유로 실패한다(config 의 `validation` 기준이나 학습 설정을 바꿔야 한다).

동시성: Job 은 `JOB_WORKERS` 개의 워커가 접수 순서대로 실행한다(기본 1, 학습 두 개가 가속기 하나를 나눠 쓰지 않도록).
워커는 Job 하나를 실행하는 동안 가속기 하나를 맡는다. GPU 가 여러 개인 서버에서는 `JOB_WORKERS` 를 GPU 수로 두면
워커 N 이 `cuda:N` 에서 학습한다. GPU 하나나 Mac 에서 값을 올리면 여러 Job 이 같은 가속기를 나눠 쓴다. 스테이지
디렉토리는 파일 잠금으로 보호되므로 CLI 와 서버가 같은 스테이지를 동시에 요청하면 한쪽이 기다렸다가 결과를 재사용한다.

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
| 인증 | `API_TOKEN` 설정 시 Bearer |
| 요청 Content-Type | 없음 (본문 없음) |
| 응답 Content-Type | `application/json` |
| 필수 헤더 | `API_TOKEN` 설정 시 `Authorization` |

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

### DELETE /api/jobs/{job_id}

대기 중이거나 실행 중인 Job 을 취소한다.

#### Endpoint

```
DELETE /api/jobs/{job_id}
```

#### 호출 규약

| 항목 | 값 |
|---|---|
| Method | `DELETE` |
| 인증 | `API_TOKEN` 설정 시 Bearer |
| 요청 Content-Type | 없음 (본문 없음) |
| 응답 Content-Type | `application/json` |
| 필수 헤더 | `API_TOKEN` 설정 시 `Authorization` |

#### Request Body

없음.

#### Query / Path 파라미터

| 파라미터 | 위치 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---|---|---|---|
| `job_id` | path | string | 예 | — | `POST /api/jobs` 가 돌려준 값 |

#### Response Body

[JobResponse](#jobresponse). 취소 요청 직후의 상태이므로 아직 `pending` 이나 `running` 일 수 있다.

#### 응답 코드

| 코드 | 의미 | 본문 |
|---|---|---|
| `202` | 취소 접수 (이미 `cancelled` 인 Job 도 202) | JobResponse |
| `404` | 없음 | `{"detail": "No such job: ..."}` |
| `409` | `done` / `failed` / `interrupted` 인 Job 은 취소할 수 없음 | `{"detail": "..."}` |

#### 호출 예시

```bash
$ curl -s -X DELETE http://localhost:8000/api/jobs/sft-qwen3.5-0.8b-<hash>-<stages>
{"job_id":"sft-qwen3.5-0.8b-<hash>-<stages>","status":"running",...}
```

#### 특이사항

**취소 시점** — 스레드는 밖에서 멈출 수 없으므로 파이프라인이 다음 진행 보고(학습이면 다음 스텝, 검사면 다음 묶음)에서
스스로 멈춘다. 모델 다운로드처럼 진행을 보고하지 않는 구간은 그 구간이 끝난 뒤 멈춘다. 멈추면 `status` 가
`cancelled` 가 되고 `error` 는 null 이다. `GET /api/jobs/{job_id}` 로 확인한다.

**산출물** — 끝난 스테이지와 학습 중 `resume.pt` 는 남는다. 같은 config 를 다시 제출하면 거기서 이어서 실행된다.

**재호출** — 취소 중이거나 이미 취소된 Job 에 다시 보내도 202, 상태는 그대로.

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
| 인증 | `API_TOKEN` 설정 시 Bearer |
| 요청 Content-Type | 없음 (본문 없음) |
| 응답 Content-Type | `application/json` |
| 필수 헤더 | `API_TOKEN` 설정 시 `Authorization` |

#### Request Body

없음.

#### Query / Path 파라미터

| 파라미터 | 위치 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---|---|---|---|
| — | — | — | — | — | 없음 |

#### Response Body

레시피 키(`pipeline.recipe` 값)마다 객체 하나. 모든 레시피에 `stages` 가 있고, 나머지 필드는 레시피가 정한다.

LLM, Embedding 계열 레시피(`llm_sft`, `llm_instruction`, `llm_dpo`, `llm_grpo`, `llm_decision_sft`, `llm_decision_cispo`,
`embedding_contrastive`)의 필드:

| 필드 | 타입 | 널 허용 | 설명 |
|---|---|---|---|
| `stages` | string[] | 아니오 | `["data","train","validate"]` |
| `architecture` | string[] | 아니오 | `model.architecture` 에 쓸 수 있는 모델 카탈로그 키 |
| `source` | string[] | 아니오 | `data.sources[*].name` 에 쓸 수 있는 키 |
| `method_keys` | string[] | 아니오 | `method` 섹션에 쓸 수 있는 키 |
| `reward` | string[] | 아니오 | `llm_grpo` 에만 있음. `method.reward.name` 에 쓸 수 있는 키 |
| `head` | string[] | 아니오 | `llm_decision_sft`, `llm_decision_cispo` 에만 있음. `model.head` 에 쓸 수 있는 키 |

Evaluation 채널 레시피(`evaluation_custom`)의 필드: `stages`(`["rows","score","report"]`), `source`, `checkpoint`
(`model.checkpoint` 에 쓸 수 있는 값: `validate`, `train`, `base`, 또는 체크포인트 디렉토리).

#### 응답 코드

| 코드 | 의미 | 본문 |
|---|---|---|
| `200` | 정상 | 위 스키마 |

#### 호출 예시

```bash
$ curl -s http://localhost:8000/api/system/recipes
{"embedding_contrastive":{"stages":["data","train","validate"],"architecture":["bi_encoder"],"source":["huggingface","local_jsonl"],"method_keys":["query_instruction","temperature"]},"llm_dpo":{"stages":["data","train","validate"],"architecture":["hybrid","transformer"],"source":["huggingface","local_jsonl"],"method_keys":["beta"]},"llm_grpo":{...,"method_keys":["group_size","max_new_tokens","reward","temperature"],"reward":["contains","exact_match","last_number"]},"llm_decision_sft":{...,"method_keys":["calibration","eval_think","head_lr","max_think","think_fraction"],"head":["pointer","readout"]},"llm_decision_cispo":{...,"head":["pointer","readout"]},"llm_instruction":{...,"method_keys":["system"]},"llm_sft":{...,"method_keys":[]}}
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
| 인증 | 없음 (컨테이너 healthcheck 가 부름) |
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
| `status` | string | 아니오 | `pending` / `running` / `done` / `failed` / `cancelled`(`DELETE /api/jobs/{job_id}` 로 멈춤) / `interrupted`(서버가 멈춰 끝나지 못함) |
| `recipe` | string | 아니오 | 레시피 키 |
| `experiment` | string | 아니오 | 실험 키 |
| `directory` | string | 아니오 | 실험 디렉토리 |
| `stages` | string[] | 아니오 | 스테이지 계획 (ConfigSummary 의 `stages` 와 같음) |
| `progress` | object | 아니오 | [Progress](#progress). 실행 중에는 실시간 값, 끝난 뒤에는 마지막 값 |
| `device_slot` | int | 예 | Job 을 실행한 워커 번호. GPU 가 여러 개면 `cuda:<번호>` 를 씀. 대기 중에는 null |
| `result` | object | 예 | `done` 일 때 `{"experiment","directory","stages":{stage: outputs}}` |
| `error` | string | 예 | `failed` 일 때 메시지 + traceback, `interrupted` 일 때 안내문 |

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
스키마에 없는 키는 최상위든 섹션 안이든 422 (`data.sources[*]` 와 `method.reward` 처럼 `name` + 임의 파라미터인 항목 제외).

LLM, Embedding 계열 레시피는 `modeling/tuning/config.py` 의 `TuningConfig` 를 공유한다: `model`(architecture, name,
revision, init. LLM 계열은 여기에 template, head 가 더 있고 임베딩 계열에서 쓰면 422), `data`(sources, validation), `training`(epochs, lr, weight_decay, batch_size, accumulation,
warmup_ratio, max_grad_norm, max_length, overflow, max_steps, resume_every, adapter), `validation`(min, max, batch_size), `method`.
`training.adapter`(name, r, alpha, dropout, targets)가 있으면 LoRA 로 학습하고, 없으면 전체 가중치를 학습한다.
`method` 섹션의 키는 레시피마다 다르며 `modeling/llm/config.py`, `modeling/embedding/config.py` 가 정본이다.
`validation.min/max` 에 쓸 수 있는 지표도 레시피마다 다르다: `llm_sft` 와 `llm_instruction` 은 `loss`,
`perplexity`, `llm_dpo` 는 `accuracy`, `margin`, `llm_grpo` 는 `reward`, `llm_decision_sft` 는 `accuracy`, `nll`, `ece`,
`llm_decision_cispo` 는 여기에 `think_accuracy`, `embedding_contrastive` 는 `accuracy`, `mrr`.
`model.head` 는 결정 레시피에서 필수이고 다른 레시피에서 쓰면 422. `model.init` 은 이전 실험의 체크포인트 폴더이며,
폴더가 없어도 config 검증은 통과하고 train 스테이지에서 실패한다.
