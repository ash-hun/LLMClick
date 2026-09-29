# LLMClick API 명세

| 항목 | 값 |
|---|---|
| Last Updated | 2026-09-29 |
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
   - [GET /api/system/registries](#get-apisystemregistries)
   - [GET /api/system/stages](#get-apisystemstages)
   - [GET /health](#get-health)
4. [데이터 모델](#데이터-모델)

---

## 개요

YAML 설정 하나를 실험 하나로 실행하는 서버. 정본은 `output/<name>-<hash>/manifest.json` 과 그 디렉토리의 파일이며,
API 는 그 파이프라인(`core/pipeline.py`)을 백그라운드 스레드로 띄우고 상태를 돌려준다. 라우터: `core/api/routers/`.

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
POST /api/config/load?config_path=configs/jeff_public_only.yaml
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
$ curl -s -X POST 'http://localhost:8000/api/config/load?config_path=configs/jeff_public_only.yaml'
{"valid":true,"experiment":"jeff-0.8b-public-<hash>","directory":"output/jeff-0.8b-public-<hash>","config":{...}}
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

[PipelineConfig](#pipelineconfig) 와 같은 구조의 객체. `pipeline` 섹션은 최상위로 병합된다.

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
  -d '{"pipeline":{"name":"x"},"model":{"backbone":"nope","name":"m","revision":"0000000000000000000000000000000000000000"}}'
{"detail":"1 validation error for PipelineConfig\n  Value error, Unknown backbone 'nope'; registered: ['gemma4', 'modernbert', 'qwen3_5'] ..."}
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

config 로 파이프라인을 백그라운드에서 실행한다.

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
| `stages` | string[] | 아니오 | null (전부) | `data, benchmarks, synthetic, mix, train, evaluate` 의 부분집합 |

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
  -d '{"config_path":"configs/jeff_public_only.yaml","stages":["data","benchmarks","mix"]}'
{"job_id":"jeff-0.8b-public-<hash>-<stages>","status":"pending","experiment":"jeff-0.8b-public-<hash>","directory":"output/jeff-0.8b-public-<hash>","stages":["data","benchmarks","mix"],"result":null,"error":null}
```

#### 특이사항

**재호출** — 같은 결과, 상태 변화 없음. `job_id` 는 `실험 키 + stages 해시` 이므로 같은 config·stages 를 다시 보내면
새 스레드를 만들지 않고 기존 Job(대기·실행·완료)을 돌려준다. 실패한 Job 은 재호출 시 다시 실행되며, 파이프라인은
완료된 스테이지를 manifest 로 건너뛰므로 실패 지점부터 이어진다.

동시성: 같은 실험 디렉토리를 두 프로세스가 동시에 쓰는 것은 막지 않는다. 서버 하나에 워커 하나로 운용한다.

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

### GET /api/system/registries

YAML 의 `name:` 키로 쓸 수 있는 값 목록.

#### Endpoint

```
GET /api/system/registries
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
$ curl -s http://localhost:8000/api/system/registries
{"builder":["huggingface","jeff_extra","jeff_probability","local_jsonl"],"converter":["boolean","classification","example"],"backbone":["gemma4","modernbert","qwen3_5"],"teacher":["openai_compatible"],"benchmark":["huggingface","jeff_jevbench_hard","jeff_panel","local_jsonl"]}
```

#### 특이사항

**재호출** — 상태를 바꾸지 않음.

---

### GET /api/system/stages

파이프라인 스테이지 이름을 실행 순서대로.

#### Endpoint

```
GET /api/system/stages
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

string 배열.

#### 응답 코드

| 코드 | 의미 | 본문 |
|---|---|---|
| `200` | 정상 | `["data","benchmarks","synthetic","mix","train","evaluate"]` |

#### 호출 예시

```bash
$ curl -s http://localhost:8000/api/system/stages
["data","benchmarks","synthetic","mix","train","evaluate"]
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
| `experiment` | string | 아니오 | `<name>-<config sha256[:8]>` |
| `directory` | string | 아니오 | `output_dir/experiment` |
| `config` | object | 아니오 | 기본값이 채워진 정규화 config |

### JobResponse

| 필드 | 타입 | 널 허용 | 설명 |
|---|---|---|---|
| `job_id` | string | 아니오 | `<experiment>-<stages sha256[:6]>` |
| `status` | string | 아니오 | `pending` / `running` / `done` / `failed` |
| `experiment` | string | 아니오 | 실험 키 |
| `directory` | string | 아니오 | 실험 디렉토리 |
| `stages` | string[] | 아니오 | 요청된 스테이지 |
| `result` | object | 예 | `done` 일 때 `{"experiment","directory","stages":{stage: outputs}}` |
| `error` | string | 예 | `failed` 일 때 메시지 + traceback |

### PipelineConfig

`core/config/schema.py` 가 정본. 섹션: `pipeline`(name, seed, output_dir, device, checkpoint), `data`(builders,
folds, synthetic, mix), `model`(backbone, name, revision, prompt_layout), `training`(jeff.train 인자와 동명),
`tracker`, `evaluation`(benchmarks, batch_size). `builders[*]`·`benchmarks[*]`·`teacher` 는 `name` + 임의 파라미터.
