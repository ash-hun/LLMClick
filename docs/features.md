# LLMClick 기능 정의서

| 항목 | 값 |
|---|---|
| Last Updated | 2026-10-31 |

> `Last Updated` 는 문서 내용이 실제로 바뀐 날만 적는다. 시각은 적지 않는다.

---

## 서비스 개요

YAML 설정 하나를 모델 하나로 만드는 파이프라인 서비스. config 의 `pipeline.recipe` 가 레시피를 고르고, 레시피는 스테이지를 순서대로 실행하며 입력이 같은 스테이지는 다시 만들지 않는다(`core/pipeline_runner.py`). 레시피는 세 채널로 나뉜다: 학습 데이터를 만드는 `data`, 모델을 학습하고 검증하는 `modeling`, 학습된 모델을 채점하는 `evaluation`. 같은 실행을 CLI(`llmclick run`)와 HTTP Job API(`POST /api/jobs`) 두 입구로 쓴다.

범위 밖: 학습된 모델을 서빙하는 추론 엔드포인트는 없다. 화면(프론트엔드)도 이 저장소의 코드에 없다.

엔드포인트의 요청, 응답, 재호출 동작은 [api_spec.md](api_spec.md) 에 있다.

---

## 기능 목록

기능 ID 알파벳순. 레시피의 기능 ID 는 `core.recipe_registry.RECIPES` 에 등록된 이름이다. 모든 레시피는 `POST /api/jobs` 로도 실행된다.

| 기능 ID | 기능 | 진입점 | 상태 |
|---|---|---|---|
| `data_speech` | 시드 텍스트를 Piper 음성으로 읽혀 Whisper 로 검수한 음성 학습셋 생성 | `llmclick run configs/data/speech.yaml`, `POST /api/jobs` | 구현됨 |
| `data_synthetic` | 교사 모델로 시드에서 학습 행(SFT, DPO, GRPO, decision 형식) 생성 | `llmclick run configs/data/synthetic.yaml`, `POST /api/jobs` | 구현됨 |
| `embedding_contrastive` | 임베딩 모델 대조 학습(InfoNCE) | `llmclick run configs/embedding/contrastive.yaml`, `POST /api/jobs` | 구현됨 |
| `evaluation_benchmark` | lm-evaluation-harness 공개 벤치마크 채점 | `llmclick run configs/evaluation/benchmark.yaml`, `POST /api/jobs` | 구현됨 |
| `evaluation_compare` | 기준 모델을 포함한 여러 체크포인트 비교 | `llmclick run configs/evaluation/compare.yaml`, `POST /api/jobs` | 구현됨 |
| `evaluation_custom` | 학습 레시피 자신의 지표로 임의 행 채점 | `llmclick run configs/evaluation/custom.yaml`, `POST /api/jobs` | 구현됨 |
| `evaluation_decision` | decision 모델을 Jev 행 또는 JevBench 티어에서 채점 | `llmclick run configs/evaluation/decision.yaml`, `POST /api/jobs` | 구현됨 |
| `evaluation_embedding` | 임베딩 모델을 MTEB 태스크로 채점 | `llmclick run configs/evaluation/embedding.yaml`, `POST /api/jobs` | 구현됨 |
| `evaluation_wakeword` | 웨이크워드 검출기의 미검출률과 시간당 오수락 측정 | `llmclick run configs/evaluation/wakeword.yaml`, `POST /api/jobs` | 구현됨 |
| `job_api` | config 검증, 백그라운드 Job 실행과 조회, 평가 리포트 조회, 레시피 목록 | `POST /api/jobs` 외 [job_api](#job_api-http-job-api) 참조 | 구현됨 |
| `llm_decision_cispo` | decision 헤드 LLM 의 추론 강화(CISPO) | `llmclick run configs/llm/decision_cispo.yaml`, `POST /api/jobs` | 구현됨 |
| `llm_decision_sft` | decision 헤드 LLM 지도 학습 | `llmclick run configs/llm/decision_pointer.yaml`, `POST /api/jobs` | 구현됨 |
| `llm_dpo` | LLM 선호 최적화(DPO) | `llmclick run configs/llm/dpo.yaml`, `POST /api/jobs` | 구현됨 |
| `llm_grpo` | LLM 보상 기반 강화 학습(GRPO) | `llmclick run configs/llm/grpo.yaml`, `POST /api/jobs` | 구현됨 |
| `llm_instruction` | instruction/input/output 레코드로 LLM 지시 튜닝 | `llmclick run configs/llm/instruction.yaml`, `POST /api/jobs` | 구현됨 |
| `llm_sft` | 대화 데이터로 LLM 지도 미세조정 | `llmclick run configs/llm/sft.yaml`, `POST /api/jobs` | 구현됨 |
| `tts_piper` | Piper(VITS) 음성 미세조정과 청취 검증 | `llmclick run configs/tts/piper_jarvis.yaml`, `POST /api/jobs` | 구현됨 |
| `tts_qwen` | Qwen3-TTS 로 녹음 하나에서 음성 복제와 청취 검증 | `llmclick run configs/tts/qwen_jarvis.yaml`, `POST /api/jobs` | 구현됨 |
| `wakeword` | openWakeWord 검출기 학습과 스트리밍 테스트 | `llmclick run configs/wakeword/jarvis.yaml`, `POST /api/jobs` | 구현됨 |

---

## 기능별 상세

### data_speech: 음성 학습셋 생성

**설명**: 시드에서 만든 텍스트를 Piper 음성이 읽고, 길이, 클리핑, 무음 검사와 Whisper 받아쓰기로 걸러, Piper 학습셋(`wavs/`, `metadata.csv`, `rows.jsonl`)으로 쓴다. 녹음과 받아쓰기는 자연 키로 캐시되어 중단 후 재실행하면 없는 것만 만든다.

**진입점**: `llmclick run configs/data/speech.yaml`, `llmclick run configs/data/wake_phrase.yaml`, `llmclick run configs/data/wake_negatives.yaml`, `llmclick run configs/data/wake_nearmiss.yaml`, `llmclick run configs/data/wake_eval_phrase.yaml`, `llmclick run configs/data/wake_eval_negatives.yaml`, `POST /api/jobs`.

**입력과 결과**

| 입력 | 결과 |
|---|---|
| config 의 `seeds`, `prompts`(생성기), `voice`(Piper 음성, `speakers`), 음성 검사 설정 | `select` 스테이지의 `rows.jsonl`, `metadata.csv`, `wavs/`, manifest |

**처리 흐름**: `SpeechData` 에서 `seeds`, `prompts`, `synthesize`, `verify`, `select` 순.

**구현 파일**

| 파일 | 층 | 역할 |
|---|---|---|
| `data/speech/__init__.py` | package | 음성 데이터셋 패키지 |
| `data/speech/speech_config.py` | config | `SpeechConfig`: 텍스트 레시피의 seeds, prompts 에 음성과 음성 검사 설정을 더함 |
| `data/speech/synthesize_stage.py` | stage | `synthesize`: 프롬프트마다 Piper 로 읽어 자연 키로 보관 |
| `data/speech/speech_verify_stage.py` | stage | `verify`: 길이, 클리핑, 무음 검사 후 Whisper 받아쓰기와 텍스트 비교 |
| `data/speech/speech_select_stage.py` | stage | `select`: 텍스트당 녹음 하나를 Piper 학습셋으로 기록 |

**테스트**

| 종류 | 파일 |
|---|---|
| 단위 | 없음: 이 기능의 모듈 검사는 통합 테스트 파일이 함께 한다 |
| 통합 | `tests/integ/test_data_speech.py` |

**제약과 미구현**:
- `voice.speakers` 가 화자를 하나도 지정하지 않으면 config 검증에서 거부한다(`data/speech/speech_config.py`).
- `# idempotent:` Piper 는 노이즈를 시드 없이 뽑아 같은 텍스트도 매번 다른 파일이 된다. 첫 파일을 자연 키로 보관하고 다시 읽히지 않는다(`modeling/tts/piper_process.py`).
- `# ponytail:` WER 계산에 숫자, 약어 정규화가 없다(`modeling/tts/asr_listener.py`).
- `# ponytail:` Whisper 는 앞 30초만 듣는다(`modeling/tts/asr_listener.py`).

---

### data_synthetic: 합성 학습 데이터 생성

**설명**: 시드에서 교사 모델(anthropic, local, ollama, openai)로 프롬프트를 만들고, 선택적으로 Evol-Instruct 로 어렵게 바꾼 뒤, 답변을 받아 검증하고, 중복과 평가셋 유출을 걸러 목표 레시피 형식의 학습 행으로 쓴다. 교사 호출은 자연 키로 캐시되고 예산을 매 호출 전에 확인한다.

**진입점**: `llmclick run configs/data/synthetic.yaml`, `llmclick run configs/data/route.yaml`, `POST /api/jobs`.

**입력과 결과**

| 입력 | 결과 |
|---|---|
| config 의 `teacher`, `seeds`, `prompts`(생성기), `evolve`(선택), `respond`, `verify`(스키마, 보상, 다수결, 판정자), `select.assemble`(`sft`, `dpo`, `grpo`, `decision`) | `select` 스테이지의 `rows.jsonl`(목표 레시피 형식), manifest, 행 수와 제외 사유별 카운트 |

**처리 흐름**: `SyntheticData` 에서 `seeds`, `prompts`, `evolve`(config 에 `evolve` 가 있을 때만), `respond`, `verify`, `select` 순.

**구현 파일**

| 파일 | 층 | 역할 |
|---|---|---|
| `data/evolve_stage.py` | stage | `evolve`: Evol-Instruct 연산자(`EVOLVERS`)로 프롬프트를 어렵게 바꾸고 실패를 제거 |
| `data/respond_stage.py` | stage | `respond`: 교사가 프롬프트마다 N번 답변, 예산 초과 시 `BudgetExceeded` |
| `data/verify_stage.py` | stage | `verify`: 스키마, sorry 규칙, 보상, 다수결 검사 후 판정자, 거절 사유 기록 |
| `data/select_stage.py` | stage | `select`: 중복과 유출 제거, 목표 레시피 형식으로 조립, manifest 기록 |

**테스트**

| 종류 | 파일 |
|---|---|
| 단위 | `tests/unit/test_data_synthetic_teacher_client.py` |
| 통합 | `tests/integ/test_data_synthetic.py` |

**제약과 미구현**:
- 시드는 `values` 와 `source` 중 하나만, `source` 에는 `field` 가 필요하다. 등록되지 않은 evolve 연산자, 보상, 판정 교사는 config 검증에서 거부한다. `assemble: decision` 이나 `verify.schema_recipe: llm_decision_sft` 에는 `select.decision` 이 필요하다(`data/data_config.py`).
- `# ponytail:` 근사 중복 검사가 남긴 행 전부와 비교해 O(n^2) 이다. 약 5만 행을 넘으면 MinHash 와 LSH 로 바꾼다(`data/select_stage.py`).

---

### embedding_contrastive: 임베딩 대조 학습

**설명**: 쿼리가 배치 안의 다른 문서보다 자기 positive 를 높게 점수 매기도록 bi-encoder(Qwen3-Embedding-0.6B)를 InfoNCE 로 학습하고, 검증 행에서 accuracy, mrr 로 검증한다.

**진입점**: `llmclick run configs/embedding/contrastive.yaml`, `POST /api/jobs`.

**입력과 결과**

| 입력 | 결과 |
|---|---|
| 행 `{"query", "positive", "negative"?}`(예: `samples/embedding_contrastive.jsonl`), `model.architecture: bi_encoder`, `method`(`query_instruction`, `temperature`) | `train` 체크포인트, `validate` 의 `validation.json`(accuracy, mrr, documents)과 검증 통과 시 `checkpoint` 링크 |

**처리 흐름**: `ContrastivePipeline` 에서 `data`, `train`, `validate` 순.

**구현 파일**

| 파일 | 층 | 역할 |
|---|---|---|
| `modeling/embedding/__init__.py` | package | 임베딩 패밀리 |
| `modeling/embedding/embedding_pipeline.py` | pipeline | `ContrastivePipeline` 등록 |
| `modeling/embedding/embedding_config.py` | config | `ContrastiveConfig`: 임베딩 카탈로그와 방법 |
| `modeling/embedding/methods/__init__.py` | package | 임베딩 학습 방법 |
| `modeling/embedding/methods/contrastive_method.py` | method | InfoNCE 손실과 accuracy, mrr 평가 |
| `modeling/embedding/models/__init__.py` | model | `EMBEDDING_BACKBONES` 카탈로그 |
| `modeling/embedding/models/embedding_backbone.py` | model | 텍스트를 단위 벡터로 바꾸는 임베딩 백본 기반(`evaluation_embedding` 도 읽음) |
| `modeling/embedding/models/bi_encoder_backbone.py` | model | 마지막 토큰 풀링 bi-encoder |

**테스트**

| 종류 | 파일 |
|---|---|
| 단위 | `tests/unit/test_embedding_contrastive_bi_encoder_backbone.py` |
| 통합 | `tests/integ/test_tuning.py` |

**제약과 미구현**: 행의 `query`, `positive`, `negative` 는 빈 문자열이 아니어야 한다(`modeling/embedding/methods/contrastive_method.py`). `model.revision` 은 40자 커밋 해시여야 한다(로컬 디렉터리 제외, `modeling/tuning/tuning_config.py`).

---

### evaluation_benchmark: 공개 벤치마크 채점

**설명**: 실험의 체크포인트(또는 Hub 모델)를 lm-evaluation-harness 공개 벤치마크로 채점한다. 벤치마크마다 스테이지가 하나라 나중에 추가한 벤치마크만 새로 채점하고, `evaluation_compare` 와 결과를 공유한다.

**진입점**: `llmclick run configs/evaluation/benchmark.yaml`, `POST /api/jobs`.

**입력과 결과**

| 입력 | 결과 |
|---|---|
| `model`(`experiment` 와 `checkpoint`, 또는 Hub `name` 과 커밋 `revision`), `benchmarks`(`preset`: `korean`, `small_general` 또는 태스크 목록), harness 설정 | 벤치마크별 `results`, `samples.jsonl`, `report` 스테이지의 `report.json` |

**처리 흐름**: `BenchmarkEvaluation` 에서 `score:<task>`(벤치마크마다 하나, `BenchmarkStage`), `report` 순.

**구현 파일**

| 파일 | 층 | 역할 |
|---|---|---|
| `evaluation/benchmark_evaluation.py` | stage | `BenchmarkConfig`, 프리셋, 벤치마크별 `BenchmarkStage`, `BenchmarkReport`(`evaluation_compare` 도 재사용) |

**테스트**

| 종류 | 파일 |
|---|---|
| 단위 | `tests/unit/test_evaluation_benchmark_benchmark_evaluation.py` |
| 통합 | `tests/integ/test_evaluation_benchmark.py` |

**제약과 미구현**: `benchmarks.preset` 은 등록된 프리셋이어야 하고 프리셋이나 태스크가 하나 이상 있어야 한다. `thinking` 은 `chat_template: true` 가 필요하다. 언어 모델이 아닌 실험과 lm-eval 에 없는 태스크는 거부한다(`evaluation/benchmark_evaluation.py`).

---

### evaluation_compare: 체크포인트 비교

**설명**: 기준 모델을 포함한 여러 체크포인트를 같은 벤치마크, 행, decision 질문, MTEB 태스크로 채점해 기준 대비 차이와 불확실성을 한 표, 다각형 차트, Markdown 요약으로 쓴다. 각 채점 스테이지는 단일 모델 레시피와 같은 것이라 이미 잰 결과를 재사용한다.

**진입점**: `llmclick run configs/evaluation/compare.yaml`, `llmclick run configs/evaluation/small_llms.yaml`, `POST /api/jobs`.

**입력과 결과**

| 입력 | 결과 |
|---|---|
| `runs`(라벨별 모델, 하나는 기준), `benchmarks`, `rows`, `decision`, `embedding` 중 하나 이상, 비교 축 | `report` 스테이지의 `report.json`, `summary.md`, `axes.svg` |

**처리 흐름**: `CompareEvaluation` 에서 `score:<label>:<task>`(벤치마크), `rows` 와 `rows:<label>`(행 채점), `questions` 와 `decision:<label>`(decision 채점), `embedding:<label>:<task>`(MTEB), `report` 순. config 에 있는 섹션의 스테이지만 만든다.

**구현 파일**

| 파일 | 층 | 역할 |
|---|---|---|
| `evaluation/compare_evaluation.py` | stage | `CompareConfig`, 기준 대비 차이 판정, 차트, `CompareReport` |

**테스트**

| 종류 | 파일 |
|---|---|
| 단위 | `tests/unit/test_evaluation_compare_compare_evaluation.py` |
| 통합 | `tests/integ/test_evaluation_compare.py` |

**제약과 미구현**: run 라벨은 서로 달라야 하고, 하나는 시작 가중치(`checkpoint: base` 또는 Hub `name`)여야 한다. `benchmarks`, `rows`, `decision`, `embedding` 중 하나는 있어야 하고, 축은 그 섹션들이 내는 항목만 이름할 수 있다(`evaluation/compare_evaluation.py`).

---

### evaluation_custom: 레시피 지표로 임의 행 채점

**설명**: 학습된 실험의 체크포인트를 그 실험을 학습한 레시피 자신의 지표(`evaluate`)로 임의 행에서 채점하고, 학습 행과의 오염도를 함께 보고한다.

**진입점**: `llmclick run configs/evaluation/custom.yaml`, `POST /api/jobs`.

**입력과 결과**

| 입력 | 결과 |
|---|---|
| `model`(`experiment`, `checkpoint`: `validate`, `train`, `base` 또는 디렉터리), `data.sources`(실험 레시피 형식의 행) | `rows.jsonl`, 채점 결과, `report` 스테이지의 `report.json`(지표, 오염도) |

**처리 흐름**: `CustomEvaluation` 에서 `rows`, `score`, `report` 순.

**구현 파일**

| 파일 | 층 | 역할 |
|---|---|---|
| `evaluation/custom_stages.py` | stage | `RowsStage`, `ScoreStage`, `ReportStage`(모델과 행 묶음별로 `for_` 로 만들어 `evaluation_compare`, `evaluation_decision` 도 재사용) |

**테스트**

| 종류 | 파일 |
|---|---|
| 단위 | `tests/unit/test_evaluation_custom_contamination_check.py` |
| 통합 | `tests/integ/test_evaluation_custom.py` |

**제약과 미구현**: 실험 없는 Hub 모델(`model.name`)은 채점할 수 없다. 벤치마크와 임베딩 레시피만 Hub 모델을 받는다(`evaluation/source_model.py`). 실험 레시피 형식에 맞지 않는 행이나 빈 행 집합은 거부한다(`evaluation/custom_stages.py`).

---

### evaluation_decision: decision 모델 채점

**설명**: decision 실험의 체크포인트를 Jev 행이나 JevBench 공개 티어에서 추론 없이, 그리고 선택 시 추론 후 채점해 accuracy, NLL, ECE 를 티어별 우연 수준과 함께, 추론 길이와 질문당 지연과 함께 쓴다.

**진입점**: `llmclick run configs/evaluation/decision.yaml`, `llmclick run configs/evaluation/route.yaml`, `POST /api/jobs`.

**입력과 결과**

| 입력 | 결과 |
|---|---|
| `model`(decision 실험), `data.sources`(Jev 행 또는 `jevbench` 소스), `think`(`off`, `on`, `both`) | `items.jsonl`, 채점 결과, `report` 스테이지의 `report.json` |

**처리 흐름**: `DecisionEvaluation` 에서 `rows`, `score`(`DecisionScoreStage`), `report` 순.

**구현 파일**

| 파일 | 층 | 역할 |
|---|---|---|
| `evaluation/decision_evaluation.py` | stage | `DecisionConfig`, `jevbench` 소스, `DecisionScoreStage`, `DecisionReport`(`evaluation_compare` 도 재사용) |

**테스트**

| 종류 | 파일 |
|---|---|
| 단위 | `tests/unit/test_evaluation_decision_decision_evaluation.py` |
| 통합 | `tests/integ/test_evaluation_decision.py` |

**제약과 미구현**: decision 레시피가 아닌 실험은 거부한다. JevBench 티어는 `easy`, `standard`, `hard` 만 받는다(`evaluation/decision_evaluation.py`).

---

### evaluation_embedding: MTEB 채점

**설명**: 임베딩 모델을 `mteb` 패키지의 MTEB 태스크로 채점한다. 태스크마다 캐시되는 스테이지가 하나다.

**진입점**: `llmclick run configs/evaluation/embedding.yaml`, `POST /api/jobs`.

**입력과 결과**

| 입력 | 결과 |
|---|---|
| `model`(임베딩 실험 또는 Hub 모델), `tasks`, `max_length` | 태스크별 결과와 main_score, `report` 스테이지의 `report.json` |

**처리 흐름**: `EmbeddingEvaluation` 에서 `score:<task>`(태스크마다 하나, `EmbeddingStage`), `report` 순.

**구현 파일**

| 파일 | 층 | 역할 |
|---|---|---|
| `evaluation/embedding_evaluation.py` | stage | `EmbeddingConfig`, mteb 인코더 어댑터, `EmbeddingStage`, `EmbeddingReport`(`evaluation_compare` 도 재사용) |

**테스트**

| 종류 | 파일 |
|---|---|
| 단위 | 없음: 이 기능의 모듈 검사는 통합 테스트 파일이 함께 한다 |
| 통합 | `tests/integ/test_evaluation_embedding.py` |

**제약과 미구현**: 임베딩 모델이 아닌 레시피나 아키텍처, 알 수 없는 MTEB 태스크는 거부한다(`evaluation/embedding_evaluation.py`).

---

### evaluation_wakeword: 웨이크워드 검출기 평가

**설명**: 학습에 쓰지 않은 녹음을 여러 잡음 조건에서, 그리고 문구가 없는 장시간 오디오에서 검출기를 80ms 마다 채점해, 조건별 미검출률, 시간당 오수락, 임계값 스윕을 쓴다. 점수 추적은 저장해 임계값을 바꿔도 다시 듣지 않는다.

**진입점**: `llmclick run configs/evaluation/wakeword.yaml`, `POST /api/jobs`.

**입력과 결과**

| 입력 | 결과 |
|---|---|
| `model`(`wakeword` 실험과 `checkpoint`: `validate`, `train`, 또는 번들 디렉터리), 녹음 소스, 주변음과 잡음 WAV | `clips` 의 jsonl, `traces.npz`, `report` 스테이지의 `report.json` |

**처리 흐름**: `WakewordEvaluation` 에서 `clips`, `traces`, `report` 순.

**구현 파일**

| 파일 | 층 | 역할 |
|---|---|---|
| `evaluation/wakeword_evaluation.py` | stage | `WakewordEvaluationConfig`, `ClipsStage`, `TraceStage`, `WakewordReport` |

**테스트**

| 종류 | 파일 |
|---|---|
| 단위 | `tests/unit/test_evaluation_wakeword_wakeword_evaluation.py` |
| 통합 | `tests/integ/test_evaluation_wakeword.py` |

**제약과 미구현**:
- `checkpoint` 가 `validate` 나 `train` 이면 `model.experiment` 가 필요하다. 오디오 파일이 없는 행, `.wav` 가 없는 디렉터리는 거부한다(`evaluation/wakeword_evaluation.py`).
- `# ponytail:` 목표 오수락에서의 임계값은 스윕한 값 중 가장 민감한 것이고 사이를 보간하지 않는다. 0.01 간격이 거칠면 보간한다(`evaluation/wakeword_evaluation.py`).

---

### job_api: HTTP Job API

**설명**: 레시피 실행을 HTTP 로 연다. config 를 검증만 하거나, 백그라운드 워커에서 실행하고 상태와 진행률을 조회, 취소한다. 평가 레시피가 쓴 리포트를 목록과 전문으로 읽고, 서버가 아는 레시피 목록을 돌려준다. Job 은 SQLite 테이블에 남아 재시작 후에도 조회되고, 같은 config 를 실행 중에 다시 제출하면 같은 Job 을 돌려준다.

**진입점**: `llmclick serve` 또는 `uvicorn core.api.api_app:app`(`uvicorn main:app`), 그리고 아래 엔드포인트. 세부는 [api_spec.md](api_spec.md).

| 엔드포인트 | 라우터 |
|---|---|
| `DELETE /api/jobs/{job_id}` | `job_channel.py` |
| `GET /api/evaluations` | `evaluation_channel.py` |
| `GET /api/evaluations/{experiment}` | `evaluation_channel.py` |
| `GET /api/jobs` | `job_channel.py` |
| `GET /api/jobs/{job_id}` | `job_channel.py` |
| `GET /api/system/recipes` | `system_channel.py` |
| `GET /health` | `api_app.py` |
| `POST /api/config/load` | `config_channel.py` |
| `POST /api/config/validate` | `config_channel.py` |
| `POST /api/jobs` | `job_channel.py` |

**입력과 결과**

| 입력 | 결과 |
|---|---|
| `config_path`(서버의 YAML) 또는 본문의 config | `ConfigSummary`(레시피, 실험 키, 디렉터리, 스테이지 계획, config) |
| `POST /api/jobs` 의 `JobRequest` | `JobResponse`(202), `job_id` 는 실험 키와 스테이지 계획의 해시 |
| `job_id` | Job 상태(`pending`, `running`, `done`, `failed`, `cancelled`, `interrupted`)와 진행률 |
| `output_dir`, `recipe`, `experiment` | 평가 리포트 목록(`EvaluationSummary`) 또는 리포트 전문 |
| 없음 | 레시피별 스테이지와 레지스트리 키(`catalogue()`) |

**처리 흐름**: `core/api/api_app.py` 에서 `authorized`(토큰 검사)를 거쳐 라우터로. Job 은 `job_channel.create` 에서 `config_channel.parse`(경로 제한 `confined`, `pipeline.load` 또는 `pipeline.build`), `store().submit`, 워커가 `Pipeline.run` 순.

**구현 파일**

| 파일 | 층 | 역할 |
|---|---|---|
| `core/api/__init__.py` | package | API 패키지 |
| `core/api/api_app.py` | api | FastAPI 앱, `/health`, 라우터에 토큰 의존성 부착 |
| `core/api/access_guard.py` | api | `API_TOKEN` Bearer 검사, `API_PATHS` 밖 경로 403 |
| `core/api/api_schema.py` | api | 요청, 응답 본문(`JobRequest`, `JobResponse`, `ConfigSummary`, `EvaluationSummary`) |
| `core/api/job_store.py` | service | SQLite Job 테이블과 장치별 워커 풀, 취소, 재시작 시 `interrupted` 표시 |
| `core/api/routers/__init__.py` | api | 라우터 묶음 |
| `core/api/routers/config_channel.py` | api | `/api/config/load`, `/api/config/validate` |
| `core/api/routers/job_channel.py` | api | `/api/jobs` 제출, 목록, 조회, 취소 |
| `core/api/routers/evaluation_channel.py` | api | `/api/evaluations` 목록과 전문 |
| `core/api/routers/system_channel.py` | api | `/api/system/recipes` |
| `main.py` | api | 저장소 루트에서 `uvicorn main:app` 진입 |

**테스트**

| 종류 | 파일 |
|---|---|
| 단위 | 없음: 이 기능의 모듈 검사는 통합 테스트 파일이 함께 한다 |
| 통합 | `tests/integ/test_job_api.py` |
| 통합 | `tests/integ/test_evaluation_custom.py` |

`/api/evaluations` 조회는 학습된 실험을 공유하려고 `test_evaluation_custom.py` 에 있다.

**제약과 미구현**:
- `API_TOKEN` 이 비어 있으면 `/api/*` 가 열려 있다. `/health` 는 항상 토큰 없이 열린다(`core/api/access_guard.py`, `core/api/api_app.py`).
- 끝난 Job 은 취소할 수 없다(409). 취소는 다음 진행률 보고 시점에 멈춘다(`core/api/routers/job_channel.py`).
- `# ponytail:` `tts_piper` 의 fit 중에는 Piper 가 진행률을 보내지 않아 취소된 Job 이 fit 이 끝나야 멈춘다(`modeling/tts/tts_stages.py`).
- 서버가 Job 이 끝나기 전에 멈추면 그 Job 은 `interrupted` 로 표시되고 다시 제출해야 한다(`core/api/job_store.py`).

---

### llm_decision_cispo: decision 헤드 추론 강화

**설명**: 모델이 추론한 뒤 헤드로 선택지를 고르고, 그룹 안에서 정답을 더 확실히 고르게 한 추론을 CISPO 로 강화한다(Jeeves 2단계). 보통 `llm_decision_sft` 체크포인트에서 시작한다.

**진입점**: `llmclick run configs/llm/decision_cispo.yaml`, `POST /api/jobs`.

**입력과 결과**

| 입력 | 결과 |
|---|---|
| Jev 행 `{"state", "questions": {...}}` 또는 jeff 행(예: `samples/llm_decision.jsonl`), `model.head`(`pointer`, `readout`), `method`(`group_size`, `iterations`, `clip`, `temperature`, `max_think` 등) | `train` 체크포인트, `validation.json`(accuracy, nll, ece, questions, temperature, 선택 시 think_accuracy) |

**처리 흐름**: `DecisionCISPOPipeline` 에서 `data`, `train`, `validate` 순.

**구현 파일**

| 파일 | 층 | 역할 |
|---|---|---|
| `modeling/llm/methods/decision/cispo_method.py` | method | 추론 그룹 샘플링과 CISPO 손실 |

**테스트**

| 종류 | 파일 |
|---|---|
| 단위 | `tests/unit/test_llm_decision_sft_decision_method.py` |
| 통합 | `tests/integ/test_llm_decision_cispo.py` |

단위 테스트는 llm_decision_sft 와 공유하는 `decision/decision_method.py` 대상.

**제약과 미구현**: `model.head` 가 필수이고 `pointer`, `readout` 중 하나여야 한다(`modeling/llm/llm_config.py`). 학습 행이 2개 미만이면 학습과 보정 행으로 나눌 수 없어 거부하고, `training.max_length` 를 넘는 질문도 거부한다(`modeling/llm/methods/decision/decision_method.py`).

---

### llm_decision_sft: decision 헤드 지도 학습

**설명**: 헤드가 질문의 선택지마다 점수를 내고 그 위의 교차 엔트로피로 학습한다. 선택적으로 기준 모델의 추론을 문맥으로 넣고, 보정 행으로 온도를 맞춘다.

**진입점**: `llmclick run configs/llm/decision_pointer.yaml`, `llmclick run configs/llm/decision_readout.yaml`, `llmclick run configs/llm/route.yaml`, `POST /api/jobs`.

**입력과 결과**

| 입력 | 결과 |
|---|---|
| Jev 또는 jeff 행(예: `samples/llm_decision.jsonl`), `model.head`(`pointer`, `readout`), `method`(`calibration`, `eval_think`, `head_lr`, `max_think`, `think_fraction`) | `train` 체크포인트, `validation.json`(accuracy, nll, ece, questions, temperature, 선택 시 think_accuracy) |

**처리 흐름**: `DecisionSFTPipeline` 에서 `data`, `train`, `validate` 순.

**구현 파일**

| 파일 | 층 | 역할 |
|---|---|---|
| `modeling/llm/methods/decision/decision_sft_method.py` | method | 헤드 교차 엔트로피, 기준 모델 추론 캐시 |

**테스트**

| 종류 | 파일 |
|---|---|
| 단위 | `tests/unit/test_llm_decision_sft_decision_method.py` |
| 통합 | `tests/integ/test_llm_decision_sft.py` |

**제약과 미구현**: `model.head` 가 필수이고 `pointer`, `readout` 중 하나여야 한다(`modeling/llm/llm_config.py`). 질문 type 은 `choice`, `noul`, `score` 만 받는다. 학습 행이 2개 미만이면 거부한다(`modeling/llm/methods/decision/decision_method.py`).

---

### llm_dpo: 선호 최적화

**설명**: 건드리지 않은 기준 모델보다 `chosen` 을 `rejected` 보다 더 선호하도록 학습한다. 기준 모델의 로그 확률은 캐시한다.

**진입점**: `llmclick run configs/llm/dpo.yaml`, `POST /api/jobs`.

**입력과 결과**

| 입력 | 결과 |
|---|---|
| 행 `{"prompt", "chosen", "rejected"}`(예: `samples/llm_dpo.jsonl`), `method.beta` | `train` 체크포인트, `validation.json`(accuracy, margin) |

**처리 흐름**: `DPOPipeline` 에서 `data`, `train`, `validate` 순.

**구현 파일**

| 파일 | 층 | 역할 |
|---|---|---|
| `modeling/llm/methods/dpo_method.py` | method | DPO 손실, 기준 로그 확률 캐시 |

**테스트**

| 종류 | 파일 |
|---|---|
| 단위 | `tests/unit/test_llm_dpo_dpo_method.py` |
| 통합 | `tests/integ/test_tuning.py` |

**제약과 미구현**: `chosen`, `rejected` 는 빈 문자열이 아니고 서로 달라야 하며, `prompt` 는 문자열이나 메시지 목록이어야 한다(`modeling/llm/methods/dpo_method.py`).

---

### llm_grpo: 보상 기반 강화 학습

**설명**: 프롬프트마다 완성문 그룹을 샘플링해 보상 함수(`contains`, `exact_match`, `last_number`)로 점수 매기고, 그룹 평균을 넘는 완성문의 가능도를 높인다.

**진입점**: `llmclick run configs/llm/grpo.yaml`, `llmclick run configs/llm/grpo_gsm8k.yaml`, `POST /api/jobs`.

**입력과 결과**

| 입력 | 결과 |
|---|---|
| 행 `{"prompt", ...보상이 읽는 키(예: "answer")}`(예: `samples/llm_grpo.jsonl`), `method`(`reward`, `group_size`, `beta`, `clip`, `iterations`, `max_new_tokens`, `temperature`) | `train` 체크포인트, `validation.json`(reward) |

**처리 흐름**: `GRPOPipeline` 에서 `data`, `train`, `validate` 순.

**구현 파일**

| 파일 | 층 | 역할 |
|---|---|---|
| `modeling/llm/methods/grpo_method.py` | method | GRPO 손실, 보상 레지스트리 `REWARDS`(`data_synthetic` 의 verify 도 읽음) |

**테스트**

| 종류 | 파일 |
|---|---|
| 단위 | `tests/unit/test_llm_grpo_grpo_method.py` |
| 통합 | `tests/integ/test_llm_grpo.py` |
| 통합 | `tests/integ/test_tuning.py` |

**제약과 미구현**: 등록되지 않은 보상은 거부한다. `method.beta > 0` 이면 `training.adapter` 가 필요하다: KL 기준이 어댑터 아래의 기준 모델이다(`modeling/llm/llm_config.py`).

---

### llm_instruction: 지시 튜닝

**설명**: instruction/input/output 레코드를 대화로 바꿔 SFT 하고, 손실은 output 에만 건다. `method.system` 이 있으면 시스템 메시지로 붙인다.

**진입점**: `llmclick run configs/llm/instruction.yaml`, `POST /api/jobs`.

**입력과 결과**

| 입력 | 결과 |
|---|---|
| 행 `{"instruction", "input"?, "output"}`(예: `samples/llm_instruction.jsonl`), `method.system` | `train` 체크포인트, `validation.json`(loss, perplexity, tokens) |

**처리 흐름**: `InstructionPipeline` 에서 `data`, `train`, `validate` 순.

**구현 파일**

| 파일 | 층 | 역할 |
|---|---|---|
| `modeling/llm/methods/instruction_method.py` | method | 레코드를 대화로 바꾸는 SFT 확장(`modeling/llm/methods/sft_method.py` 의 `SFT` 상속) |

**테스트**

| 종류 | 파일 |
|---|---|
| 단위 | 없음: 이 기능의 모듈 검사는 통합 테스트 파일이 함께 한다 |
| 통합 | `tests/integ/test_tuning.py` |

단위 테스트 없음: 방법 코드가 `llm_sft` 의 SFT 를 그대로 쓴다.

**제약과 미구현**: `instruction`, `output` 은 문자열이어야 한다(`modeling/llm/methods/instruction_method.py`).

---

### llm_sft: 지도 미세조정

**설명**: 대화 데이터의 모든 assistant 턴에 교차 엔트로피를 걸어 LLM 을 미세조정한다. 하이브리드(Qwen3.5)와 일반 트랜스포머(Qwen3) 백본, 전체 미세조정과 LoRA 어댑터를 config 로 고른다.

**진입점**: `llmclick run configs/llm/sft.yaml`, `llmclick run configs/llm/sft_lora.yaml`, `llmclick run configs/llm/sft_transformer.yaml`, `POST /api/jobs`.

**입력과 결과**

| 입력 | 결과 |
|---|---|
| 행 `{"messages": [...]}`(예: `samples/llm_sft.jsonl`), `model.architecture`(`hybrid`, `transformer`), `training.adapter`(선택) | `train` 체크포인트, `validation.json`(loss, perplexity, tokens) |

**처리 흐름**: `SFTPipeline` 에서 `data`, `train`, `validate` 순.

**구현 파일**

| 파일 | 층 | 역할 |
|---|---|---|
| `modeling/llm/methods/sft_method.py` | method | assistant 턴 교차 엔트로피, loss, perplexity 평가(`llm_instruction` 이 상속) |

**테스트**

| 종류 | 파일 |
|---|---|
| 단위 | `tests/unit/test_llm_sft_sft_method.py` |
| 단위 | `tests/unit/test_llm_sft_training_loop.py` |
| 단위 | `tests/unit/test_llm_sft_training_method.py` |
| 단위 | `tests/unit/test_llm_sft_tuning_backbone.py` |
| 통합 | `tests/integ/test_llm_sft.py` |
| 통합 | `tests/integ/test_tuning.py` |

**제약과 미구현**: 대화는 assistant 턴으로 끝나야 하고 모든 메시지는 허용된 role 과 문자열 content 를 가져야 한다(`modeling/llm/methods/sft_method.py`). decision 이 아닌 레시피에 `model.head` 를 주면 거부한다(`modeling/llm/llm_config.py`).

---

### tts_piper: Piper 음성 미세조정

**설명**: 녹음과 텍스트로 Piper(VITS) 음성을 시작 가중치에서 미세조정해 음성 파일로 내보내고, 검증 텍스트를 읽혀 Whisper WER, 길이 비, UTMOS, DNSMOS, 화자 유사도로 청취 검증한다. `validation.phrases` 의 고정 응답은 잡음 제거 필터를 건 버전과 원본 중 잘 들리는 쪽을 `phrases/` 에 남긴다.

**진입점**: `llmclick run configs/tts/piper_jarvis.yaml`, `POST /api/jobs`.

**입력과 결과**

| 입력 | 결과 |
|---|---|
| `audio`, `text` 행 소스, 시작 가중치(`init`, Hub 파일은 커밋 `revision`), `quality`(`medium`, `high`), `training`, `validation.phrases` | `train` 의 음성 체크포인트, `validation.json`(wer, reference_wer, duration_ratio, utmos, dnsmos, speaker_similarity, phrase_wer), `phrases/`, `phrases.json` |

**처리 흐름**: `PiperPipeline` 에서 `data`(`VoiceDataStage`), `train`(`FitStage`), `validate`(`PiperListenStage`) 순.

**구현 파일**

| 파일 | 층 | 역할 |
|---|---|---|
| `modeling/tts/piper_process.py` | program | `PIPER_PYTHON` 프로세스로 Piper speak, fit, export 실행(`data_speech` 도 speak 사용) |

**테스트**

| 종류 | 파일 |
|---|---|
| 단위 | `tests/unit/test_tts_piper_audio_file.py` |
| 단위 | `tests/unit/test_tts_piper_quality_rater.py` |
| 통합 | `tests/integ/test_tts_piper.py` |

**제약과 미구현**:
- Hub 파일은 `file` 과 40자 커밋 `revision` 이 필요하다. `training.args` 로 레시피가 정하는 키를 덮어쓸 수 없다. `validation.phrases` 이름은 영문자, 숫자, `_`, `-` 만 쓴다(`modeling/tts/tts_config.py`).
- `# idempotent:` Piper 가 cudnn.deterministic 을 꺼 GPU 실행은 비트 단위로 같지 않다. 중단된 실행은 마지막 epoch 스냅샷에서 이어간다(`modeling/tts/tts_stages.py`).
- `# idempotent:` Piper speak 는 시드 없는 노이즈라 같은 텍스트도 매번 다른 파일이 된다(`modeling/tts/piper_process.py`).
- `# ponytail:` 취소된 API Job 은 fit 이 끝나야 멈춘다(`modeling/tts/tts_stages.py`).
- `# ponytail:` 오디오 지문을 매 실행 디렉터리에서 다시 계산한다. 느려지면 해시 manifest 로 바꾼다(`modeling/tts/tts_stages.py`).
- `# ponytail:` WER 에 숫자, 약어 정규화가 없고 Whisper 는 앞 30초만 듣는다(`modeling/tts/asr_listener.py`).

---

### tts_qwen: Qwen3-TTS 음성 복제

**설명**: 학습 없이 Qwen3-TTS 가 녹음 하나와 그 텍스트로 음성을 복제한다. 설정한 길이 안에서 UTMOS 가 가장 자연스럽다고 매긴 학습 클립을 참조로 고르고, `tts_piper` 와 같은 청취 검증을 한다.

**진입점**: `llmclick run configs/tts/qwen_jarvis.yaml`, `POST /api/jobs`.

**입력과 결과**

| 입력 | 결과 |
|---|---|
| `audio`, `text` 행 소스, `model.revision`(40자 커밋), `reference`(`min_seconds`, `max_seconds`), `validation.phrases` | `train` 의 복제 설정과 참조 클립, `validation.json`(tts_piper 와 같은 지표), `phrases/`, `phrases.json` |

**처리 흐름**: `QwenPipeline` 에서 `data`(`VoiceDataStage`), `train`(`CloneStage`), `validate`(`QwenListenStage`) 순.

**구현 파일**

| 파일 | 층 | 역할 |
|---|---|---|
| `modeling/tts/qwen_process.py` | program | `QWEN_PYTHON` 프로세스로 `environment/qwen/speak.py` 실행, 배치당 모델 1회 로드 |
| `environment/qwen/speak.py` | program | stdin JSON 요청으로 복제 음성 wav 생성 |
| `environment/qwen/pyproject.toml` | env | Qwen3-TTS 실행 환경 의존성 |
| `environment/qwen/uv.lock` | env | 위 환경 잠금 파일 |
| `environment/qwen/setup.sh` | env | `QWEN_PYTHON` 환경 구성 |

**테스트**

| 종류 | 파일 |
|---|---|
| 단위 | 없음: 이 기능의 모듈 검사는 통합 테스트 파일이 함께 한다 |
| 통합 | `tests/integ/test_tts_qwen.py` |

**제약과 미구현**:
- `model.revision` 은 40자 커밋이어야 하고 `reference.min_seconds` 는 `max_seconds` 보다 작아야 한다(`modeling/tts/tts_config.py`).
- `QWEN_PYTHON` 기본 환경은 MLX 기반 Qwen3-TTS 로 Apple silicon 용이다(`core/settings.py`).
- `# ponytail:` 오디오 지문을 매 실행 다시 계산한다(`modeling/tts/tts_stages.py`). WER 정규화 없음, Whisper 30초 제한(`modeling/tts/asr_listener.py`).

---

### wakeword: 웨이크워드 검출기 학습

**설명**: 문구 녹음과 그 밖의 녹음을 잡음 섞인 창으로 잘라 openWakeWord 의 고정 특징으로 바꾸고, 작은 DNN 검출기를 학습해 ONNX 로 내보낸 뒤, 잡음 속 보류 녹음 스트림과 "문구 뒤 명령" 시나리오로 검증한다. 창의 무작위성은 시드와 녹음에서 나와 재실행해도 같은 파일이 된다.

**진입점**: `llmclick run configs/wakeword/jarvis.yaml`, `POST /api/jobs`.

**입력과 결과**

| 입력 | 결과 |
|---|---|
| 문구(positive)와 그 밖(negative) 녹음 소스, 배경 잡음, `augment`(`rounds`, `snr`, `gain`), 검출기 학습 설정 | `features` 의 임베딩, `train` 의 `<slug>.onnx` 와 두 특징 모델 번들, `validation.json`(recall, false_accepts, scenario) |

**처리 흐름**: `WakewordPipeline` 에서 `data`(`ClipsStage`), `features`(`WindowsStage`), `train`(`FitDetectorStage`), `validate`(`StreamTestStage`) 순.

**구현 파일**

| 파일 | 층 | 역할 |
|---|---|---|
| `modeling/wakeword/__init__.py` | package | 웨이크워드 패밀리 |
| `modeling/wakeword/wakeword_pipeline.py` | pipeline | `WakewordPipeline` 등록 |
| `modeling/wakeword/wakeword_config.py` | config | `WakewordConfig`: 문구, 녹음, 증강, 검출기, 스트리밍 테스트 |
| `modeling/wakeword/wakeword_stages.py` | stage | `data`, `features`, `train`, `validate` 스테이지와 스트림 판정(`evaluation_wakeword` 도 재사용) |
| `modeling/wakeword/feature_embedder.py` | model | openWakeWord 고정 전처리: 16kHz 오디오를 80ms 마다 96차원 임베딩으로 |
| `modeling/wakeword/wakeword_detector.py` | model | DNN 검출기 학습과 ONNX 내보내기 |

**테스트**

| 종류 | 파일 |
|---|---|
| 단위 | `tests/unit/test_wakeword_wakeword_detector.py` |
| 통합 | `tests/integ/test_wakeword.py` |

**제약과 미구현**:
- `augment.snr`, `augment.gain` 은 `[low, high]` 범위여야 한다(`modeling/wakeword/wakeword_config.py`).
- `# ponytail:` ONNX 내보내기에 torch 2.9 부터 deprecated 인 TorchScript exporter 를 쓴다. torch 가 지우면 `dynamo=True` 로 옮긴다(`modeling/wakeword/wakeword_detector.py`).
- `# ponytail:` 배경 파일 전부를 16kHz 로 메모리에 올린다. 잡음셋이 GB 단위가 되면 디스크에서 스트리밍한다(`modeling/wakeword/wakeword_stages.py`).

---

## 공통 모듈

여러 기능이 함께 쓰는 파일.

| 파일 | 역할 | 쓰는 기능 |
|---|---|---|
| `core/__init__.py` | 프레임워크 패키지 | 전체 |
| `core/cli_app.py` | `llmclick run`, `validate`, `recipes`, `serve` | 전체 |
| `core/config/experiment_store.py` | 지문으로 이름 붙인 공유 스테이지 디렉터리와 실험 디렉터리 | 전체 |
| `core/config/config_schema.py` | 모든 레시피가 공유하는 config 키(`BaseConfig`) | 전체 |
| `core/logger.py` | structlog 로깅, `LOG_FORMAT` 에 따라 콘솔 또는 JSON | 전체 |
| `core/pipeline_runner.py` | 스테이지를 순서대로 실행하고 이미 만든 것은 건너뛰는 러너, `@recipe` 등록 | 전체 |
| `core/progress_reporter.py` | 진행률 보고(터미널 바, Job 상태), 취소 | 전체 |
| `core/recipe_registry.py` | 이름 레지스트리, `RECIPES`, `catalogue()` | 전체 |
| `core/settings.py` | 환경 변수와 `LLMCLICK_ENV` 파일 설정 | 전체 |
| `core/stage_base.py` | 입력으로 지문을 매기고 두 번 실행해도 안전한 스테이지 단위 | 전체 |
| `core/device_selector.py` | cuda, mps, cpu 순 장치 선택 | 전체 |
| `core/file_store.py` | 내용 해시, 원자적 쓰기, JSON, 잠금 | 전체 |
| `data/__init__.py` | data 채널 패키지, import 시 레시피 등록 | `data_speech`, `data_synthetic` |
| `data/data_pipeline.py` | `SyntheticData`, `SpeechData` 레시피 | `data_speech`, `data_synthetic` |
| `data/data_config.py` | 교사, 시드, 스테이지별 config(`SyntheticConfig`, `SeedConfig`) | `data_speech`, `data_synthetic` |
| `data/seeds_stage.py` | `seeds` 스테이지: 시드마다 안정적인 id 를 가진 행 | `data_speech`, `data_synthetic` |
| `data/prompts_stage.py` | `prompts` 스테이지와 생성기 레지스트리 `GENERATORS` | `data_speech`, `data_synthetic` |
| `data/teacher_client.py` | 교사 모델 4종(`TEACHERS`), 토큰과 비용 보고 | `data_speech`, `data_synthetic` |
| `data/row_store.py` | 자연 키와 원자적 행 파일 | `data_speech`, `data_synthetic` |
| `evaluation/__init__.py` | evaluation 채널 패키지, import 시 레시피 등록 | `evaluation_*` 전체 |
| `evaluation/evaluation_pipeline.py` | 평가 레시피 6종과 리포트 트래커 전송 | `evaluation_*` 전체 |
| `evaluation/evaluation_config.py` | 평가 대상 모델(`SourceModel`), 행, 측정 설정 | `evaluation_*` 전체 |
| `evaluation/source_model.py` | 실험 config 재구성과 체크포인트 위치, 또는 Hub 모델 로드 | `evaluation_benchmark`, `evaluation_compare`, `evaluation_custom`, `evaluation_decision`, `evaluation_embedding` |
| `evaluation/contamination_check.py` | 학습 행과 평가 항목의 정확 일치, n-gram 근사 중복 측정 | `data_synthetic`, `evaluation_benchmark`, `evaluation_compare`, `evaluation_custom`, `evaluation_decision` |
| `modeling/__init__.py` | modeling 채널 패키지, import 시 레시피 등록 | `llm_*`, `embedding_contrastive`, `tts_*`, `wakeword` |
| `modeling/modeling_pipeline.py` | 학습하면 반드시 검증하는 `ModelingPipeline` | `llm_*`, `embedding_contrastive`, `tts_*`, `wakeword` |
| `modeling/modeling_stages.py` | `TrainStage`, 통과해야 체크포인트 링크를 남기는 `ValidateStage` | `llm_*`, `embedding_contrastive`, `tts_*`, `wakeword` |
| `modeling/modeling_config.py` | 트래커와 검증 범위 config, 소스 키 | `llm_*`, `embedding_contrastive`, `tts_*`, `wakeword`, `data_*`, `evaluation_*` |
| `modeling/metric_tracker.py` | wandb 로 학습 지표와 평가 리포트 전송 | `llm_*`, `embedding_contrastive`, `tts_*`, `wakeword`, `evaluation_*` |
| `modeling/tuning/__init__.py` | 프로세스 내 미세조정 패키지 | `llm_*`, `embedding_contrastive` |
| `modeling/tuning/tuning_pipeline.py` | `TuningPipeline`: `data`, `train`, `validate` | `llm_*`, `embedding_contrastive` |
| `modeling/tuning/tuning_stages.py` | `DataStage`, `TuneStage`, `MeasureStage` | `llm_*`, `embedding_contrastive` |
| `modeling/tuning/training_loop.py` | 결정적 배치, 누적, 워밍업 후 선형 감쇠, 재개 스냅샷 | `llm_*`, `embedding_contrastive` |
| `modeling/tuning/training_method.py` | 학습 방법 기반: 행 검사, 손실, 평가 | `llm_*`, `embedding_contrastive`, `evaluation_custom`, `evaluation_decision` |
| `modeling/tuning/tuning_backbone.py` | 모델 카탈로그 기반: 로드, 저장, 학습 가중치 | `llm_*`, `embedding_contrastive`, `evaluation_*`(wakeword 제외) |
| `modeling/tuning/tuning_config.py` | 백본, 행, 최적화 설정, LoRA 어댑터 | `llm_*`, `embedding_contrastive`, `tts_*` |
| `modeling/tuning/row_sources.py` | 행 소스 레지스트리(`delimited`, `huggingface`, `jevbench`, `local_jsonl`) | 전체 레시피 |
| `modeling/llm/__init__.py` | LLM 패밀리 패키지 | `llm_*` |
| `modeling/llm/llm_pipeline.py` | LLM 레시피 6종 등록 | `llm_*` |
| `modeling/llm/llm_config.py` | 레시피별 config 와 head 검사 | `llm_*` |
| `modeling/llm/methods/__init__.py` | LLM 학습 방법 패키지 | `llm_*` |
| `modeling/llm/methods/llm_method.py` | 대화를 토큰 id 와 채점 위치로 | `llm_sft`, `llm_instruction`, `llm_dpo`, `llm_grpo`, `llm_decision_*` |
| `modeling/llm/models/__init__.py` | `LLM_BACKBONES` 카탈로그 | `llm_*`, `evaluation_*` |
| `modeling/llm/models/llm_backbone.py` | LLM 백본 기반: 채팅 렌더링, 로짓, 생성 | `llm_*`, `data_synthetic`, `evaluation_benchmark`, `evaluation_decision` |
| `modeling/llm/models/backbone_config.py` | LLM `model:` 섹션 | `llm_*`, `evaluation_*` |
| `modeling/llm/models/hybrid_backbone.py` | 하이브리드 디코더(Qwen3.5-0.8B, 2B) | `llm_*` |
| `modeling/llm/models/transformer_backbone.py` | 일반 디코더 트랜스포머(Qwen3-0.6B, 1.7B) | `llm_*` |
| `modeling/llm/methods/decision/__init__.py` | decision 방법 패키지 | `llm_decision_sft`, `llm_decision_cispo` |
| `modeling/llm/methods/decision/decision_method.py` | Jev 행 형식, 헤드 손실, 보정 지표(ECE), 온도 맞춤 | `llm_decision_sft`, `llm_decision_cispo`, `evaluation_decision` |
| `modeling/llm/models/heads/__init__.py` | `HEADS` 레지스트리 | `llm_decision_sft`, `llm_decision_cispo` |
| `modeling/llm/models/heads/decision_head.py` | 선택지마다 점수를 내는 decision 헤드 기반 | `llm_decision_sft`, `llm_decision_cispo` |
| `modeling/llm/models/heads/pointer_head.py` | pointer 헤드(Jeeves) | `llm_decision_sft`, `llm_decision_cispo` |
| `modeling/llm/models/heads/readout_head.py` | readout 헤드(Jev, jeff) | `llm_decision_sft`, `llm_decision_cispo` |
| `modeling/tts/__init__.py` | TTS 패밀리 패키지 | `tts_piper`, `tts_qwen` |
| `modeling/tts/tts_pipeline.py` | `PiperPipeline`, `QwenPipeline` 등록 | `tts_piper`, `tts_qwen` |
| `modeling/tts/tts_config.py` | `PiperConfig`, `QwenConfig`, ASR, Hub 파일 설정 | `tts_piper`, `tts_qwen`, `data_speech` |
| `modeling/tts/tts_stages.py` | `VoiceDataStage`, `FitStage`, `CloneStage`, `ListenStage` 와 하위 2종 | `tts_piper`, `tts_qwen` |
| `modeling/tts/quality_rater.py` | UTMOS, DNSMOS P.835, 화자 유사도 | `tts_piper`, `tts_qwen` |
| `modeling/tts/asr_listener.py` | Whisper 받아쓰기와 WER | `tts_piper`, `tts_qwen`, `data_speech`, `wakeword` |
| `modeling/tts/audio_file.py` | WAV 위치, 샘플, 리샘플링, 음성 검사 측정 | `tts_piper`, `tts_qwen`, `data_speech`, `wakeword`, `evaluation_wakeword` |
| `environment/piper/pyproject.toml` | Piper 실행 환경 의존성 | `tts_piper`, `data_speech` |
| `environment/piper/uv.lock` | Piper 환경 잠금 파일 | `tts_piper`, `data_speech` |
| `environment/piper/setup.sh` | `PIPER_PYTHON` 환경 구성 | `tts_piper`, `data_speech` |
| `environment/Dockerfile` | 서비스 이미지 빌드 | 전체 |
| `compose.yaml` | 커밋 태그 이미지로 서비스 실행, `/health` 헬스체크 | 전체 |
| `environment/.env.sample` | 환경 변수 예시 | 전체 |

공통 모듈의 테스트: 단위 `tests/unit/test_framework_file_store.py`, `tests/unit/test_framework_logger.py`, `tests/unit/test_framework_recipe_registry.py`, `tests/unit/test_framework_settings.py`, `tests/unit/test_framework_modeling_stages.py`, 통합 `tests/integ/test_framework.py`. 테스트 보조: `tests/conftest.py`(외부 접속 차단, 작은 무작위 LM, 공유 fixture), `tests/recipe_inputs.py`(레시피별 config 와 입력 행), `tests/recipe_fakes.py`(가짜 wandb, lm-evaluation-harness), `tests/speech_fakes.py`(가짜 Piper, Whisper, openWakeWord 전처리).
