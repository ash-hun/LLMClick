# LLMClick 기능 정의서

| 항목 | 값 |
|---|---|
| Last Updated | 2026-11-01 |

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
| `llm_distill` | 큰 교사 모델이 학생의 답을 토큰마다 채점하는 on-policy distillation | `llmclick run configs/llm/distill.yaml`, `POST /api/jobs` | 구현됨 |
| `llm_dpo` | LLM 선호 최적화(DPO) | `llmclick run configs/llm/dpo.yaml`, `POST /api/jobs` | 구현됨 |
| `llm_grpo` | LLM 보상 기반 강화 학습(GRPO) | `llmclick run configs/llm/grpo.yaml`, `POST /api/jobs` | 구현됨 |
| `llm_instruction` | instruction/input/output 레코드로 LLM 지시 튜닝 | `llmclick run configs/llm/instruction.yaml`, `POST /api/jobs` | 구현됨 |
| `llm_kto` | 짝 없이 바람직함/아님 라벨만으로 하는 선호 최적화(KTO) | `llmclick run configs/llm/kto.yaml`, `POST /api/jobs` | 구현됨 |
| `llm_orpo` | 기준 모델 없는 선호 최적화(ORPO): chosen SFT 와 odds 비 항 | `llmclick run configs/llm/orpo.yaml`, `POST /api/jobs` | 구현됨 |
| `llm_quantize` | 학습된 체크포인트나 Hub 모델 양자화(AWQ, GPTQ, W8A8, FP8, bitsandbytes NF4/INT8, GGUF)와 기준 대비 손실 검증 | `llmclick run configs/llm/quantize_awq.yaml`, `llmclick run configs/llm/quantize_gguf.yaml`, `POST /api/jobs` | 구현됨 |
| `llm_reward` | chosen/rejected 짝으로 답마다 점수 하나를 내는 보상 모델(Bradley-Terry) 학습 | `llmclick run configs/llm/reward.yaml`, `POST /api/jobs` | 구현됨 |
| `llm_sft` | 대화 데이터로 LLM 지도 미세조정 | `llmclick run configs/llm/sft.yaml`, `POST /api/jobs` | 구현됨 |
| `llm_simpo` | 기준 모델 없는 선호 최적화(SimPO): 토큰 평균 로그 확률 마진 | `llmclick run configs/llm/simpo.yaml`, `POST /api/jobs` | 구현됨 |
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
| config 의 `teacher`, `seeds`, `prompts`(생성기), `evolve`(선택), `respond`, `verify`(스키마, 보상, 다수결, 판정자), `select.assemble`(`sft`, `dpo`, `grpo`, `decision`, `kto`: judge 점수가 `select.kto_threshold` 이상이면 바람직함, 프롬프트의 판정된 답마다 한 행) | `select` 스테이지의 `rows.jsonl`(목표 레시피 형식), manifest, 행 수와 제외 사유별 카운트 |

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

### llm_distill: on-policy distillation

**설명**: 학생 모델이 프롬프트에 직접 답을 샘플링하고, 같은 어휘를 쓰는 큰 교사 모델이 그 답의 토큰마다 로그 확률을 매긴다. 토큰마다 학생과 교사의 로그 확률 차(reverse KL 추정)를 그 토큰의 손실 가중치로 써서, 학생이 쓰지 않을 텍스트를 흉내 내는 대신 자기 답이 교사와 갈라지는 자리를 고친다(Qwen3 보고서, Thinking Machines 의 on-policy distillation).

**진입점**: `llmclick run configs/llm/distill.yaml`, `POST /api/jobs`.

**입력과 결과**

| 입력 | 결과 |
|---|---|
| 행 `{"prompt"}`(예: `samples/llm_grpo.jsonl`, 다른 키는 읽지 않음), `method.teacher`(`architecture`, `name`, `revision`), `method.samples`, `method.max_new_tokens`, `method.temperature` | `train` 체크포인트, `validation.json`(teacher_logp: 학생 greedy 답의 교사 토큰 평균 로그 확률, reverse_kl: 같은 토큰의 학생과 교사 로그 확률 차 평균) |

**처리 흐름**: `DistillPipeline` 에서 `data`, `train`, `validate` 순. `train` 이 학습 전에 교사를 학생과 같은 장치에 올리고 학습이 끝나면 내린다. validate 단계는 교사를 다시 올려 채점한 뒤 내린다.

**구현 파일**

| 파일 | 층 | 역할 |
|---|---|---|
| `modeling/llm/methods/distill_method.py` | method | 교사 로드와 어휘 검사, 샘플링, reverse KL 손실, 평가 |

**테스트**

| 종류 | 파일 |
|---|---|
| 단위 | `tests/unit/test_llm_distill_distill_method.py` |
| 통합 | `tests/integ/test_llm_distill.py` |
| 통합 | `tests/integ/test_tuning.py` |

**제약과 미구현**: 교사와 학생의 토크나이저 어휘가 다르면 로드할 때 거부한다(Qwen3-0.6B 와 Qwen3-1.7B 는 같고, Qwen3 와 Qwen3.5 는 다르다). 교사는 학생의 `model.template` 으로 같은 토큰을 읽는다. 교사는 학습 내내 학생 옆 메모리에 있다(Qwen3-1.7B FP32 약 7GB). 교사 `name` 이 로컬 디렉터리면 그 내용이 train 지문에 들어간다(`modeling/tuning/tuning_config.py` 의 `local_models`). 토큰마다 자기 차이만 가중치로 쓰고 뒤 토큰의 차이를 앞으로 할인해 더하지 않는다. MPS 에서 Qwen3-0.6B 를 Qwen3-1.7B 로 18 스텝(lr 5e-6, 샘플 프롬프트) 학습하자 held-out reverse_kl 이 1.65 에서 0.59 로, teacher_logp 가 -1.79 에서 -0.60 으로 움직였다.

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
| `modeling/llm/methods/preference_method.py` | method | 선호 행 검사, 길이, accuracy/margin 평가(`llm_orpo`, `llm_simpo` 와 공유) |

**테스트**

| 종류 | 파일 |
|---|---|
| 단위 | `tests/unit/test_llm_dpo_dpo_method.py` |
| 통합 | `tests/integ/test_tuning.py` |

**제약과 미구현**: `chosen`, `rejected` 는 빈 문자열이 아니고 서로 달라야 하며, `prompt` 는 문자열이나 메시지 목록이어야 한다(`modeling/llm/methods/preference_method.py`).

---

### llm_grpo: 보상 기반 강화 학습

**설명**: 프롬프트마다 완성문 그룹을 샘플링해 보상 함수로 점수 매기고, 그룹 평균을 넘는 완성문의 가능도를 높인다. 보상은 정답 비교(`contains`, `exact_match`, `last_number`), 형식(`regex`), 길이(`length`), 교사 판정(`judge`), 이들의 가중합(`weighted`)이다.

**진입점**: `llmclick run configs/llm/grpo.yaml`, `llmclick run configs/llm/grpo_gsm8k.yaml`, `llmclick run configs/llm/grpo_weighted.yaml`, `POST /api/jobs`.

**입력과 결과**

| 입력 | 결과 |
|---|---|
| 행 `{"prompt", ...보상이 읽는 키(예: "answer")}`(예: `samples/llm_grpo.jsonl`), `method`(`reward`, `group_size`, `beta`, `clip`, `clip_high`, `iterations`, `max_new_tokens`, `temperature`, `loss_norm`, `scale_rewards`, `reward_workers`, `policy_loss`) | `train` 체크포인트, `validation.json`(reward) |

**처리 흐름**: `GRPOPipeline` 에서 `data`, `train`, `validate` 순.

**구현 파일**

| 파일 | 층 | 역할 |
|---|---|---|
| `modeling/llm/methods/grpo_method.py` | method | GRPO 손실 |
| `modeling/llm/methods/grpo_reward.py` | method | 보상 레지스트리 `REWARDS` 와 규칙 보상, `weighted` 가중합(`data_synthetic` 의 verify 도 읽음) |
| `data/judge_reward.py` | reward | `judge` 보상: 교사가 루브릭으로 1-10 점, 0-1 로 환산, 프로세스 메모리 캐시. data 채널에서 등록해 modeling 이 data 를 import 하지 않는다 |

**테스트**

| 종류 | 파일 |
|---|---|
| 단위 | `tests/unit/test_llm_grpo_grpo_method.py` |
| 단위 | `tests/unit/test_llm_grpo_grpo_reward.py` |
| 단위 | `tests/unit/test_llm_grpo_judge_reward.py` |
| 통합 | `tests/integ/test_llm_grpo.py` |
| 통합 | `tests/integ/test_tuning.py` |

**제약과 미구현**: 등록되지 않은 보상은 거부하고, `weighted` 는 각 `parts` 의 이름도 검사하며 `weighted` 를 중첩하지 않는다(`modeling/llm/methods/grpo_reward.py`). `method.beta > 0` 이면 `training.adapter` 가 필요하다: KL 기준이 어댑터 아래의 기준 모델이다(`modeling/llm/llm_config.py`). `reward_model` 은 `llm_reward` 체크포인트가 완성문을 채점한다([llm_reward](#llm_reward-보상-모델bradley-terry)). `judge` 는 새 완성문마다 교사를 한 번 부른다(스텝당 최대 `group_size` x `batch_size` 회, 순차 호출). 판정 응답은 신뢰하지 않는 입력이라 1-10 범위 밖의 숫자나 숫자 없는 응답은 최저점이다. `reward_workers` 가 1보다 크면 한 배치의 보상을 스레드로 동시에 계산하고 결과 순서와 값은 순차 계산과 같다. `local` 교사는 프로세스 안의 모델 하나라 호출이 잠금으로 차례를 지킨다(`data/teacher_client.py`). `loss_norm: token` 은 배치의 채점 토큰 전체로 평균을 내 긴 완성문의 토큰이 짧은 것과 같은 무게를 갖는다(Dr. GRPO, DAPO). 평균은 마이크로 배치 안에서 내므로 `accumulation` 이 1보다 크면 스텝 전체의 토큰 평균과 조금 다르다. `scale_rewards: false` 는 그룹 표준편차로 나누지 않는다. `policy_loss` 는 `ppo`(토큰마다 비율 클립), `gspo`(완성문마다 토큰 평균 로그 비의 지수 하나를 클립, Qwen3), `cispo`(토큰 비율을 위에서만 자르고 그래디언트 없는 가중치로 씀, MiniMax-M1) 중 하나이고, `clip_high` 는 위쪽 클립만 넓힌다(DAPO 의 clip-higher). 이 레시피는 샘플을 만든 가중치로 바로 학습하므로 그룹을 처음 쓸 때 비율이 1 이고 세 손실의 그래디언트가 같다. 차이는 `iterations` 가 1보다 클 때 재사용하는 스텝에서만 나고, 그때 `clipped`(클립된 토큰이나 완성문 비율)를 기록한다. `gspo` 는 `loss_norm: sequence` 만 받는다. MPS 에서 Qwen3-0.6B(lr 2e-6, `iterations: 2`)로 돌리면 재사용 스텝의 clipped 가 ppo(clip 0.2) 0.07에서 0.32, gspo(clip 3e-4, clip_high 4e-4) 0.38에서 1.0, cispo(clip_high 4) 0에서 0.01 이었다. 논문의 gspo 클립 값은 이 lr 에서 재사용 스텝을 대부분 멈추므로 `clipped` 를 보고 고른다. 이 다섯 키는 기본값이면 지문에서 빠진다.

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

### llm_kto: 라벨 선호 최적화(KTO)

**설명**: 답 하나마다 바람직함(`true`) 또는 아님(`false`) 라벨만 있으면 학습한다. chosen, rejected 짝이 필요 없어 평점이나 좋아요/싫어요 기록을 그대로 쓴다. 기준 모델 대비 로그 확률 비를 바람직한 답은 올리고 아닌 답은 내리며, 기준점은 행끼리 답을 맞바꾼 쌍에서 추정한 KL 이다(KTO 논문).

**진입점**: `llmclick run configs/llm/kto.yaml`, `POST /api/jobs`.

**입력과 결과**

| 입력 | 결과 |
|---|---|
| 행 `{"prompt", "completion", "label"}`(예: `samples/llm_kto.jsonl`, `data_synthetic` 의 `kto` 조립), `method.beta`, `method.desirable_weight`, `method.undesirable_weight` | `train` 체크포인트, `validation.json`(desirable_logp, undesirable_logp, gap: 토큰 평균 로그 확률 차) |

**처리 흐름**: `KTOPipeline` 에서 `data`, `train`, `validate` 순. `train` 이 학습 전에 기준 모델의 로그 확률(행의 답, 다음 행의 답)을 한 번 계산해 `reference_logps.json` 에 캐시한다.

**구현 파일**

| 파일 | 층 | 역할 |
|---|---|---|
| `modeling/llm/methods/kto_method.py` | method | KTO 손실, 기준 로그 확률 캐시, 라벨별 평가 |

**테스트**

| 종류 | 파일 |
|---|---|
| 단위 | `tests/unit/test_llm_kto_kto_method.py` |
| 통합 | `tests/integ/test_tuning.py` |
| 통합 | `tests/integ/test_data_synthetic.py`(kto 조립) |

**제약과 미구현**: `label` 은 bool, `completion` 은 빈 문자열이 아니어야 한다. KL 추정용 맞바꾼 쌍은 학습 행 순서로 정해지고 `max_length` 에서 잘린다. validate 단계에는 기준 모델이 없어 지표는 절대 로그 확률이고, held-out 행에 한쪽 라벨이 없으면 그 지표와 `gap` 은 빠진다. 짝이 없어 행끼리 공유하는 턴 끝 토큰을 바람직하지 않은 행이 함께 내리므로 lr 에 민감하다: Qwen3.5-0.8B(MPS, 6 스텝)에서 2e-5 는 모델이 무너졌고(gap 2.87 → -0.56), 5e-6 은 바람직한 답의 확률도 내렸고, 1e-6 은 gap 을 7.69 로 넓히며 유지했다.

---

### llm_orpo: 기준 모델 없는 선호 최적화(ORPO)

**설명**: chosen 답의 토큰당 NLL 에 chosen 대 rejected 의 log-odds 비 항을 `lam` 만큼 더해 학습한다. SFT 와 선호 정렬을 한 손실로 하고 기준 모델 패스가 없다.

**진입점**: `llmclick run configs/llm/orpo.yaml`, `POST /api/jobs`.

**입력과 결과**

| 입력 | 결과 |
|---|---|
| DPO 와 같은 행 `{"prompt", "chosen", "rejected"}`(예: `samples/llm_dpo.jsonl`), `method.lam` | `train` 체크포인트, `validation.json`(accuracy, margin: log-odds 차) |

**처리 흐름**: `ORPOPipeline` 에서 `data`, `train`, `validate` 순.

**구현 파일**

| 파일 | 층 | 역할 |
|---|---|---|
| `modeling/llm/methods/orpo_method.py` | method | ORPO 손실, 토큰 평균 로그 확률의 log-odds |
| `modeling/llm/methods/preference_method.py` | method | 선호 행 검사, 길이, 평가 |

**테스트**

| 종류 | 파일 |
|---|---|
| 단위 | `tests/unit/test_llm_orpo_orpo_method.py` |
| 통합 | `tests/integ/test_tuning.py` |

**제약과 미구현**: 행 규칙은 `llm_dpo` 와 같다(`modeling/llm/methods/preference_method.py`).

---

### llm_quantize: 학습 후 양자화

**설명**: 학습된 체크포인트나 Hub 모델을 양자화하고, 같은 held-out 행에서 양자화 전후 loss 를 비교한다. 양자화하는 스테이지 이름이 `train` 이라 `validate/checkpoint` 링크, `evaluation_*` 레시피, 다른 실험의 `model.name` 이 학습된 모델처럼 읽는다.

**진입점**: `llmclick run configs/llm/quantize_awq.yaml`, `llmclick run configs/llm/quantize_gguf.yaml`, `POST /api/jobs`.

**입력과 결과**

| 입력 | 결과 |
|---|---|
| `model`(디렉터리나 Hub ID+revision), SFT 대화 행(`data.sources`: train 몫이 보정, held-out 몫이 측정), `quantization.name`(`awq`, `gptq`, `w8a8`, `fp8`, `bnb_nf4`, `bnb_int8`, `gguf`), `quantization.rows`, 그 밖의 키는 scheme 인자(`scheme`, `smoothing`, gptq 와 w8a8 의 `dampening`, gguf 의 `type`) | `train/checkpoint` 양자화 모델(compressed-tensors, bitsandbytes 직렬화, 또는 `model.gguf` 와 tokenizer), `summary.json`(base_loss, base_perplexity, size_mb), `validation.json`(loss, perplexity, tokens, base_loss, loss_delta, size_mb) |

**처리 흐름**: `QuantizePipeline` 에서 `data`(`DataStage` 재사용), `train`(기준 모델 held-out loss 측정 후 scheme 실행, 결과는 임시 디렉터리에 쓰고 끝나면 이름 변경), `validate`(양자화 모델 loss 와 `loss_delta`) 순.

**구현 파일**

| 파일 | 층 | 역할 |
|---|---|---|
| `modeling/quantize/__init__.py` | package | 양자화 패키지, import 시 레시피 등록 |
| `modeling/quantize/quantize_pipeline.py` | pipeline | `QuantizePipeline` 등록, catalogue 에 scheme 목록 |
| `modeling/quantize/quantize_config.py` | config | `QuantizeConfig`(SFT 행, `quantization:` 섹션), adapter 거부 |
| `modeling/quantize/quantize_scheme.py` | scheme | `SCHEMES` 레지스트리, llm-compressor 레시피(AWQ, GPTQ, SmoothQuant+GPTQ, FP8), bitsandbytes 직렬화, llama.cpp 변환기 호출(GGUF) |
| `environment/llama_cpp/setup.sh` | environment | llama.cpp v0.7.0 소스를 sha256 확인 후 `LLAMA_CPP_DIR` 에 받는다. 다시 실행하면 아무것도 하지 않는다 |
| `modeling/quantize/quantize_stages.py` | stage | `QuantizeStage`, `QuantizeMeasureStage` |

**테스트**

| 종류 | 파일 |
|---|---|
| 단위 | `tests/unit/test_llm_quantize_quantize_scheme.py` |
| 통합 | `tests/integ/test_llm_quantize.py` |

**제약과 미구현**: `quantize` extra(llm-compressor, bitsandbytes)가 필요하다. `bnb_nf4`, `bnb_int8`, `fp8` 은 CUDA 에서만 돈다(다른 장치는 시작할 때 거부). `awq`, `gptq`, `w8a8` 은 보정 행이 필요하고 순서를 섞지 않는다. GPTQ(`gptq`, `w8a8`)는 레이어 Hessian 이 양의 정부호가 아니면 `quantization.dampening` 을 올리라는 에러로 멈춘다: Qwen3-0.6B 는 샘플 행에서 기본값 0.01 로 실패하고 0.1 로 통과한다. `evaluation_benchmark`, `evaluation_compare` 는 lm-eval 이 compressed-tensors 체크포인트를 직접 읽어 따로 처리하지 않는다(MPS, CPU 에서 AWQ 로 확인). validate 단계는 compressed-tensors 체크포인트를 dense 로 풀어 로드하므로 측정 loss 는 양자화 오차를 반영하지만 속도는 반영하지 않는다(`modeling/llm/models/llm_backbone.py`). 출력층(`lm_head`)과 백본의 `frozen` 모듈(Qwen3.5 비전 타워)은 양자화하지 않는다. `gguf` 는 llama.cpp 의 Python 변환기를 앱 환경의 별도 프로세스로 실행하고(`environment/llama_cpp/setup.sh` 가 고정 릴리스를 받음), `type` 은 `q8_0`, `f16`, `bf16` 이다. k-quant(q4_k_m 등)는 llama.cpp 의 C++ 도구가 필요해 없다. transformers 가 GGUF 를 다시 읽어 validate 하므로 그 목록에 없는 아키텍처(Qwen3.5 hybrid)는 거부한다. 고정한 llama.cpp 커밋이 지문에 들어간다. `evaluation_benchmark` 와 `evaluation_compare` 는 `gguf_file` 로 GGUF 체크포인트를 읽는다(MPS 에서 q8_0 의 arc_challenge 40 문항 acc_norm 0.40, 원본 0.425). `training.adapter` 는 거부한다.

---

### llm_reward: 보상 모델(Bradley-Terry)

**설명**: DPO 와 같은 chosen, rejected 짝으로 답마다 점수 하나를 내는 보상 모델을 학습한다. 손실은 `-logsigmoid(r_chosen - r_rejected)` 다. 체크포인트는 출력 하나짜리 sequence classifier(`AutoModelForSequenceClassification`)라 보상 모델 서버나 다른 학습기가 그대로 읽고, `llm_grpo` 는 `reward_model` 보상으로 읽는다.

**진입점**: `llmclick run configs/llm/reward.yaml`, `POST /api/jobs`.

**입력과 결과**

| 입력 | 결과 |
|---|---|
| 행 `{"prompt", "chosen", "rejected"}`(예: `samples/llm_dpo.jsonl`) | `train` 체크포인트(sequence classifier, `num_labels: 1`, `pad_token_id` 기록), `validation.json`(accuracy: chosen 점수가 더 높은 짝의 비율, margin: 점수 차 평균) |

**처리 흐름**: `RewardPipeline` 에서 `data`, `train`, `validate` 순. 백본을 sequence classifier 로 올리고(`LLMBackbone.scorer`), 각 답의 마지막 토큰 hidden state 에 `score` 층을 씌워 점수를 읽는다. LoRA 를 쓰면 새로 만든 `score` 층은 어댑터 옆에서 전부 학습하고(`modules_to_save`), 저장할 때 어댑터와 함께 합친다.

**구현 파일**

| 파일 | 층 | 역할 |
|---|---|---|
| `modeling/llm/methods/reward_method.py` | method | 짝별 점수, Bradley-Terry 손실 |
| `modeling/llm/methods/scorer_reward.py` | reward | `llm_grpo` 의 `reward_model` 보상: 체크포인트를 프로세스당 하나 올려 완성문을 채점 |
| `modeling/llm/models/llm_backbone.py` | model | `scorer`, `sequence_scores`, sequence classifier 로드와 저장 |

**테스트**

| 종류 | 파일 |
|---|---|
| 단위 | `tests/unit/test_llm_reward_reward_method.py` |
| 통합 | `tests/integ/test_llm_reward.py` |
| 통합 | `tests/integ/test_tuning.py` |

**제약과 미구현**: 점수는 범위가 없고, GRPO 는 그룹 안에서 비교하므로 순서와 간격만 쓴다. `reward_model` 보상은 프로세스당 보상 모델 하나를 최적 장치에 올려 두고 다른 `path` 를 요청받으면 바꾼다. Job 서버에서는 다음 요청까지 메모리에 남는다. 완성문마다 forward 한 번씩 순서대로 채점한다(`reward_workers` 를 늘려도 잠금으로 차례를 지킨다). `path` 가 로컬 디렉터리면 그 내용이 `llm_grpo` 의 train 지문에 들어간다. 과정 보상 모델(PRM)은 없다. MPS 에서 Qwen3.5-0.8B 를 LoRA 로 9 스텝 학습하자 held-out 8 짝의 accuracy 1.0, margin 1.66 이었고, 그 체크포인트를 보상으로 쓴 GRPO 6 스텝이 돌았다(보상 6.4 안팎).

---

### llm_sft: 지도 미세조정

**설명**: 대화 데이터의 모든 assistant 턴에 교차 엔트로피를 걸어 LLM 을 미세조정한다. 하이브리드(Qwen3.5)와 일반 트랜스포머(Qwen3) 백본, 전체 미세조정, LoRA 어댑터, QLoRA(`training.adapter.quantize: nf4 | int8`), gradient checkpointing(`training.gradient_checkpointing`), 한 서버 여러 GPU 의 데이터 병렬 학습(`training.devices`)을 config 로 고른다. QLoRA 는 모든 `llm_*` 학습 레시피에 같은 키로 적용된다.

**진입점**: `llmclick run configs/llm/sft.yaml`, `llmclick run configs/llm/sft_lora.yaml`, `llmclick run configs/llm/sft_qlora.yaml`, `llmclick run configs/llm/sft_transformer.yaml`, `POST /api/jobs`.

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
| 단위 | `tests/unit/test_llm_sft_data_parallel.py` |
| 통합 | `tests/integ/test_llm_sft.py` |
| 통합 | `tests/integ/test_tuning.py` |

**제약과 미구현**: 대화는 assistant 턴으로 끝나야 하고 모든 메시지는 허용된 role 과 문자열 content 를 가져야 한다(`modeling/llm/methods/sft_method.py`). decision 이 아닌 레시피에 `model.head` 를 주면 거부한다(`modeling/llm/llm_config.py`). QLoRA 는 CUDA 에서만 로드하고(다른 장치는 학습 전에 거부), 저장할 때 기준 모델을 `dtype`(FP32)로 CPU 에 다시 올려 어댑터를 병합하므로 체크포인트는 일반 모델이다. 병합하는 동안 기준 모델 전체가 호스트 메모리에 올라간다(`modeling/llm/models/llm_backbone.py`). QLoRA 는 `quantize` extra(bitsandbytes)가 필요하다. 키를 비워 두면 config 지문에서 빠져 이전 LoRA 실험의 지문이 그대로다(`modeling/tuning/tuning_config.py`). gradient checkpointing 은 non-reentrant 방식이라 LoRA, QLoRA 에서도 그대로 동작하고, 같은 config 의 가중치는 켜고 끈 것이 같다. Qwen3-0.6B LoRA, 1832 토큰 배치의 MPS 측정에서 forward activation 메모리가 9645MB 에서 2011MB 로 줄었다. 모든 `llm_*` 학습 레시피와 `embedding_contrastive` 에 같은 키로 적용된다(`modeling/tuning/tuning_backbone.py`). `training.devices` 가 1보다 크면 train 단계가 장치마다 프로세스를 띄운다(CUDA 는 NCCL, CPU 는 gloo, MPS 는 거부). 각 프로세스가 배치를 `rows[rank::devices]` 로 나눠 받고, 손실에 몫의 행 수 비율을 곱한 뒤 gradient 를 평균한다. 그래서 행 평균 손실(DPO, ORPO, SimPO)은 단일 프로세스와 같은 가중치가 나오고(CPU 2 프로세스, DPO 로 확인), 토큰 평균(SFT, instruction, GRPO `loss_norm: token`)이나 배치 통계(KTO 의 KL 기준점)를 쓰는 손실은 근사다. 준비(DPO, KTO 기준값)와 파일 쓰기, 중간 평가, 진행 보고는 첫 프로세스만 하고 진행과 스텝 지표는 큐로 부모에 전달한다. 취소하면 모든 프로세스를 멈춘다. 프로세스마다 모델 전체를 올리므로 한 장에 들어가는 모델만 빨라진다. NCCL 경로와 다중 프로세스 재개는 확인하지 않았다(`modeling/tuning/data_parallel.py`).

---

### llm_simpo: 기준 모델 없는 선호 최적화(SimPO)

**설명**: chosen 답의 토큰 평균 로그 확률이 rejected 의 것보다 마진 `gamma` 이상 크도록 학습한다. 합이 아닌 평균이라 긴 답이 길이로 이기지 않고, 기준 모델 패스가 없다.

**진입점**: `llmclick run configs/llm/simpo.yaml`, `POST /api/jobs`.

**입력과 결과**

| 입력 | 결과 |
|---|---|
| DPO 와 같은 행 `{"prompt", "chosen", "rejected"}`(예: `samples/llm_dpo.jsonl`, `data_synthetic` 의 `dpo` 조립), `method.beta`, `method.gamma` | `train` 체크포인트, `validation.json`(accuracy, margin: 토큰 평균 로그 확률 차) |

**처리 흐름**: `SimPOPipeline` 에서 `data`, `train`, `validate` 순.

**구현 파일**

| 파일 | 층 | 역할 |
|---|---|---|
| `modeling/llm/methods/simpo_method.py` | method | SimPO 손실, 토큰 평균 마진 |
| `modeling/llm/methods/preference_method.py` | method | 선호 행 검사, 길이, 평가 |

**테스트**

| 종류 | 파일 |
|---|---|
| 단위 | `tests/unit/test_llm_simpo_simpo_method.py` |
| 통합 | `tests/integ/test_tuning.py` |

**제약과 미구현**: 행 규칙은 `llm_dpo` 와 같다(`modeling/llm/methods/preference_method.py`). 손실에 SFT 항은 없다.

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
| `core/config/config_schema.py` | 모든 레시피가 공유하는 config 키(`BaseConfig`), 나중에 더한 키는 기본값이면 지문에서 빼는 `added_keys` | 전체 |
| `core/logger.py` | structlog 로깅, `LOG_FORMAT` 에 따라 콘솔 또는 JSON | 전체 |
| `core/pipeline_runner.py` | 스테이지를 순서대로 실행하고 이미 만든 것은 건너뛰는 러너, `@recipe` 등록 | 전체 |
| `core/progress_reporter.py` | 진행률 보고(터미널 바, Job 상태), 취소 | 전체 |
| `core/recipe_registry.py` | 이름 레지스트리, `RECIPES`, `catalogue()` | 전체 |
| `core/settings.py` | 환경 변수와 `LLMCLICK_ENV` 파일 설정(`PIPER_PYTHON`, `LLAMA_CPP_DIR` 등) | 전체 |
| `core/stage_base.py` | 입력으로 지문을 매기고 두 번 실행해도 안전한 스테이지 단위 | 전체 |
| `core/device_selector.py` | cuda, mps, cpu 순 장치 선택 | 전체 |
| `core/file_store.py` | 내용 해시, 원자적 쓰기, JSON, 잠금 | 전체 |
| `data/__init__.py` | data 채널 패키지, import 시 레시피와 `judge` 보상 등록 | `data_speech`, `data_synthetic`, `llm_grpo` |
| `data/data_pipeline.py` | `SyntheticData`, `SpeechData` 레시피 | `data_speech`, `data_synthetic` |
| `data/data_config.py` | 교사, 시드, 스테이지별 config(`SyntheticConfig`, `SeedConfig`) | `data_speech`, `data_synthetic` |
| `data/seeds_stage.py` | `seeds` 스테이지: 시드마다 안정적인 id 를 가진 행 | `data_speech`, `data_synthetic` |
| `data/prompts_stage.py` | `prompts` 스테이지와 생성기 레지스트리 `GENERATORS` | `data_speech`, `data_synthetic` |
| `data/teacher_client.py` | 교사 모델 4종(`TEACHERS`), 토큰과 비용 보고 | `data_speech`, `data_synthetic`, `llm_grpo`(`judge`) |
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
| `modeling/tuning/tuning_stages.py` | `DataStage`, `TuneStage`, `MeasureStage`, 한 프로세스나 여러 프로세스에서 같은 학습 본문 `tune` | `llm_*`, `embedding_contrastive` |
| `modeling/tuning/training_loop.py` | 결정적 배치, 누적, 워밍업 후 선형 감쇠, 재개 스냅샷, 프로세스별 배치 몫과 gradient 평균 | `llm_*`, `embedding_contrastive` |
| `modeling/tuning/data_parallel.py` | `training.devices` 프로세스 실행, 장치 배정, gradient 평균, 첫 프로세스의 진행 전달 | `llm_*`, `embedding_contrastive` |
| `modeling/tuning/training_method.py` | 학습 방법 기반: 행 검사, 손실, 평가 | `llm_*`, `embedding_contrastive`, `evaluation_custom`, `evaluation_decision` |
| `modeling/tuning/tuning_backbone.py` | 모델 카탈로그 기반: 로드, 저장, 학습 가중치 | `llm_*`, `embedding_contrastive`, `evaluation_*`(wakeword 제외) |
| `modeling/tuning/tuning_config.py` | 백본, 행, 최적화 설정(`gradient_checkpointing` 포함), LoRA 어댑터와 QLoRA `quantize` 키, 지문에 들어가는 추가 로컬 모델(`local_models`) | `llm_*`, `embedding_contrastive`, `tts_*` |
| `modeling/tuning/row_sources.py` | 행 소스 레지스트리(`delimited`, `huggingface`, `jevbench`, `local_jsonl`) | 전체 레시피 |
| `modeling/llm/__init__.py` | LLM 패밀리 패키지 | `llm_*` |
| `modeling/llm/llm_pipeline.py` | LLM 학습 레시피 9종 등록 | `llm_*` |
| `modeling/llm/llm_config.py` | 레시피별 config 와 head 검사 | `llm_*` |
| `modeling/llm/methods/__init__.py` | LLM 학습 방법 패키지 | `llm_*` |
| `modeling/llm/methods/llm_method.py` | 대화를 토큰 id 와 채점 위치로, 샘플링한 완성문의 채점 표시와 토큰별 로그 확률 | `llm_sft`, `llm_instruction`, `llm_dpo`, `llm_orpo`, `llm_simpo`, `llm_grpo`, `llm_distill`, `llm_decision_*`, `llm_quantize` |
| `modeling/llm/models/__init__.py` | `LLM_BACKBONES` 카탈로그 | `llm_*`, `evaluation_*` |
| `modeling/llm/models/llm_backbone.py` | LLM 백본 기반: 채팅 렌더링, 로짓, 생성, QLoRA 4/8bit 로드와 병합 저장, 양자화된 체크포인트 로드 | `llm_*`, `data_synthetic`, `evaluation_benchmark`, `evaluation_decision` |
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
| `environment/Dockerfile` | 서비스 이미지 빌드, `EXTRAS` 빌드 인자로 선택 의존성 그룹 설치(`quantize` 면 llama.cpp 변환기도) | 전체 |
| `compose.yaml` | 커밋 태그 이미지로 서비스 실행, `/health` 헬스체크, 포트는 loopback 만 | 전체 |
| `compose.gpu.yaml` | NVIDIA GPU 를 서비스에 붙이고 `quantize` extra 로 빌드하는 compose 덧붙임 파일 | 전체, QLoRA, `llm_quantize` |
| `environment/.env.sample` | 환경 변수 예시 | 전체 |

공통 모듈의 테스트: 단위 `tests/unit/test_framework_file_store.py`, `tests/unit/test_framework_logger.py`, `tests/unit/test_framework_recipe_registry.py`, `tests/unit/test_framework_settings.py`, `tests/unit/test_framework_config_schema.py`, `tests/unit/test_framework_modeling_stages.py`, 통합 `tests/integ/test_framework.py`. 테스트 보조: `tests/conftest.py`(외부 접속 차단, `CORPUS` 에 고정한 샘플 파일로 만든 작은 무작위 LM, 공유 fixture), `tests/recipe_inputs.py`(레시피별 config 와 입력 행), `tests/recipe_fakes.py`(가짜 wandb, lm-evaluation-harness), `tests/speech_fakes.py`(가짜 Piper, Whisper, openWakeWord 전처리).
