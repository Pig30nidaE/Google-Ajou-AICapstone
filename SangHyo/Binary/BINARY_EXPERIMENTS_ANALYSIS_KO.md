# Binary 실험 종합 분석 — 다음 Ablation Study를 위한 현황 문서

최종 갱신: **2026-07-30**

## 0. 이 문서의 목적과 다른 문서와의 관계

이 문서는 `SangHyo/`의 모든 `Binary_*` 실험 결과를 실제 산출물(`FINAL_REPORT.json` 등)
기준으로 재점검하고, **다음 ablation study를 어디에 투자해야 하는지** 판단할 근거를
정리한다. 특히 2026-07-29에 완료된 신규 LLM 기반 실험 3종
(`Binary_Gemma_CognitiveFeature_AUC`, `GemmaFeatureProgramPipeline`,
`GeminiFeaturePipeline`)은 어느 기존 문서에도 결과가 정리되어 있지 않아 이번에
처음 종합했다.

기존 문서와의 역할 분담:

| 문서 | 역할 | 이 문서와의 차이 |
| --- | --- | --- |
| `AGENTS.md` | 실행/누수 방지 계약, 2026-07-28까지의 실험 authoritative 기록 | 이 문서 이후 완료된 3개 신규 실험이 반영되어 있지 않음 |
| `EXPERIMENT_SUMMARY_KO.md` | `Binary_Google_FinalBest`(검증 정확도 0.909) 스토리 한 장 정리 | accuracy·threshold 중심, ROC-AUC 중심 비교가 아님 |
| `Binary_AUC_TWO_MODELS_ANALYSIS_KO.md` | 신규 2개 실험(YDF/Gemma) **실행 전** 설계 문서 | 실행 후 실측 결과가 없음(작성 시점엔 아직 안 돌았음) |
| `BINARY_MODEL_PROPOSAL_KO.md` | MMSE 포함/미포함 트랙별 **최종 챔피언 1개씩** 선정 | "무엇이 최고인가"에 집중, "무엇을 배웠고 다음에 뭘 바꿀까"는 다루지 않음 |
| **이 문서** | 전체 실험의 성능·교훈 종합 + 다음 ablation 후보 | — |

과제 정의는 모든 문서와 동일하다: `CN=0`, `MCI 또는 Dem=1`, Training 141명
(CN 85 / MCI 47 / Dem 9). Dem screening 등 다른 라벨 과제는 6절에서 별도로 다루며
이 문서의 다른 어떤 숫자와도 직접 비교하지 않는다.

---

## 1. 전체 실험 성능 표

숫자는 각 실험의 `FINAL_REPORT.json`(또는 동급 보고서)에서 그대로 가져왔다.
"nested"는 후보/가중치 선택이 outer held-out 라벨과 분리되어 있었는지를 뜻한다.

### 1-1. 웨어러블 전용 (MMSE 미사용)

| 실험 | 방법 | Subject-mean OOF ROC-AUC | Repeat 평균 | nested | 비고 |
| --- | --- | ---: | ---: | :---: | --- |
| `Binary_Google_ROCAUC_Champion` (wearable track) | 28일·8-view Sequence Transformer anchor | **0.6293** | 0.5955 | O | 95% CI [0.5343, 0.7245]. 현재 wearable-only 최고 |
| `Binary_Wearable_SequenceFusion_Google` | 개별 Sequence Transformer | 0.6254(추정치) | — | — | 위 anchor의 원조 가설, 대체로 재현됨 |
| `GemmaFeatureProgramPipeline` | wearable_only 블록 | — | 0.5381 | O | |
| `Binary_Wearable_GoogleModels` | 1,077개 후보 탐색 | 0.5370 | — | — | 대규모 탐색이 일반화 안 됨 |
| `Binary_PaperLGBM_NoMMSE` | 논문 특징+LightGBM | 0.5214 | — | — | 사람 단위 평가 시 무작위 수준 |
| `Binary_Wearable_BalancedFusion_Google` | gated multi-model | 0.4756 | — | — | |
| `Binary_Wearable_TabNet_Google` | 1,077개 요약+TabNet | 0.4462 | — | — | 소표본 TabNet 실패 |

### 1-2. MMSE 포함 (전통적 특징·모델)

| 실험 | 방법 | OOF ROC-AUC | nested | 비고 |
| --- | --- | ---: | :---: | --- |
| `Binary_MMSE_MaxAUC` | MMSE 39개, 규제 LR+SVM | **0.765756** | O(정식) | CI [0.6840, 0.8457]. 기존 "정식" 최고 anchor |
| `Binary_Google_ROCAUC_Champion` (MMSE track, 앵커) | MMSE-only immutable anchor | 0.762290 | — | |
| `Binary_Google_ROCAUC_Champion` (MMSE track, 선택 정책) | 앵커+wearable 결합 정책 | 0.759979 | O | CI [0.6754, 0.8361]. **앵커보다 낮음** — 결합이 오히려 손해 |
| `Binary_Google_OrdinalStable` (binary\_\_fold_topk) | 3 label strategy × 2 selection 비교 | 0.756933 | O | 7개 arm 중 승자 선택 bias 포함 |
| `Binary_EDA_Selective` | 엄선 14개, LR+얕은GBT | 0.7174 | — | 적은 특징이 복잡한 모델과 동등 이상 |
| `Binary_MMSE_DomainFusion` | MMSE 영역+wearable | 0.7095 | — | |
| `Binary_MetaEnsemble_Google` | 7개 base learner 메타 | 0.7082 | — | 복잡한 결합의 추가 이득 없음 |
| `Binary_Google_FinalBest` | MMSE 중심 14개+YDF | 0.7116 | — | 검증 정확도 0.909(threshold 성적, AUC 아님) |
| `Binary_Clinical_MMSE_Fusion` | MMSE+wearable 요약 | 0.6973 | — | |
| `Binary_PaperLGBM_MMSE` | 논문 특징+MMSE | 0.6924 | — | |
| `Binary_Google_YDF_Ensemble` | 일반 YDF GBT+RF | 0.6727 | — | |
| `Binary_Google_MaxAUC_Tuned` | MMSE39+wearable112, 대규모 탐색 | 0.717227 | O | non-nested 0.801681, optimism +0.0845 |

### 1-3. LLM 기반 특징 3종 — 이번에 새로 정리 (2026-07-29 완료)

| 실험 | LLM 역할 | 챔피언(LLM 포함) AUC | 대응 MMSE-only baseline AUC | 차이 | Paired bootstrap 95% CI | nested |
| --- | --- | ---: | ---: | ---: | --- | :---: |
| `Binary_Gemma_CognitiveFeature_AUC` | 사람별 인지 불일치 특징 10개(MMSE만 입력) | 0.7817(subject-mean) / 0.7785(repeat-mean) | 0.7748(subject-mean) / 0.7747(repeat-mean) | +0.0069 / +0.0037 | **미보고** | X |
| `Binary_Google_YDF_AUC` | (LLM 미사용, YDF 전용 — 비교용) | 0.7834(subject-mean) / 0.7661(repeat-mean) | — | — | — | X |
| `GemmaFeatureProgramPipeline` | 전역 feature-program 1회 생성(라벨 미노출) | 0.7439(subject-mean, full) / 0.7291(repeat-mean, full) | 0.7492(subject-mean) / 0.7351(repeat-mean) | **-0.0053** / -0.0060 | **[-0.01926, 0.00896]**(포함 0) | **O** |
| `GeminiFeaturePipeline`(본인 구현) | 사람별 진단중립 행동 특징 12개(wearable만 입력) | 0.7181(with MMSE+logreg) | 0.6941(with MMSE+logreg) | +0.0240 | **[-0.0157, 0.0633]**(포함 0) | X(튜닝 없음) |

**세 실험 모두, LLM이 만든 특징을 추가해도 MMSE 단독 대비 통계적으로 유의미한
개선을 보이지 못했다.** 유일하게 진짜 nested(inner-OOF) 선택을 쓴
`GemmaFeatureProgramPipeline`은 점수가 오히려 **하락**했고, 나머지 두 실험의
소폭 상승(+0.004~+0.024)은 95% 구간이 0을 포함해 "우연과 구분 안 됨" 수준이다.
자세한 내용은 2절.

---

## 2. 핵심 발견 (다음 ablation 설계에 직접 영향을 주는 것들)

### 발견 1 — LLM 특징 추가는 세 가지 독립 설계 모두에서 실패했다

- `Binary_Gemma_CognitiveFeature_AUC`: MMSE 항목을 Gemma(`gemma-4-31b-it`)에게
  보여주고 인지 불일치 특징 10개를 만들게 함. 챔피언(0.7817)과 MMSE-only
  baseline(0.7748)의 차이는 subject-mean 기준 +0.0069에 불과하고, 이 실험은
  **paired bootstrap CI 자체를 계산하지 않았다** — 즉 "유의미하다"고 주장할
  근거가 보고서에 없다.
- `GemmaFeatureProgramPipeline`: Gemma가 라벨을 보지 않고 wearable 원시 채널로부터
  전역 특징 프로그램(10개, `signed_mean`/`signed_product`/`absolute_gap` 3개
  연산만 허용)을 1회 생성. **nested inner-OOF 선택에서 100번 중 29번(최빈값)이
  wearable=0, program=0 가중치를 골랐다** — 즉 정직한 선택 절차가 자기 스스로
  "이 특징은 버리는 게 낫다"고 판단한 것과 같다. `full` arm은 MMSE-only보다
  **-0.0053** 낮았고 CI [-0.01926, 0.00896]는 0을 포함한다.
- `GeminiFeaturePipeline`(본인 구현): wearable 행동 패턴 특징 12개를 만들어
  MMSE+wearable summary에 추가. with-MMSE+logreg에서만 +0.024 상승했지만
  CI [-0.0157, 0.0633]는 0을 포함하고, with-MMSE+gbdt에서는 오히려 **-0.008**
  하락했다. without-MMSE 조합에서는 두 모델 모두 큰 의미가 없다(±0.02 이내).

세 실험의 프롬프트·입력·downstream 모델이 모두 다른데도 결론이 일관된다는 점이
중요하다. **"LLM이 새 특징을 만들어 붙이면 MMSE 위에 신호가 더 얹힌다"는
가설은 이번 표본에서 세 번 독립적으로 반증됐다.**

### 발견 2 — MMSE 단독이 여전히 최고 수준이고, 정교화가 그 위에 거의 못 얹힌다

MMSE를 쓰는 모든 정직한 실험의 점수가 **0.73~0.78 사이에 촘촘히 몰려 있다**
(`Binary_MMSE_MaxAUC` 0.7658, `Binary_Google_ROCAUC_Champion` 앵커 0.7623,
`GemmaFeatureProgramPipeline` MMSE-only 0.7351, YDF 챔피언 0.7834, Gemma
CognitiveFeature 챔피언 0.7817). 모델 복잡도(규제 LR/SVM → YDF 앙상블 →
LLM 특징 추가)가 크게 달라져도 성능 밴드가 거의 안 바뀐다는 것은, 현재 141명
표본과 특징 집합에서 얻을 수 있는 정보량 자체가 이 근처에서 막혀 있다는
신호다(`EXPERIMENT_SUMMARY_KO.md`가 지적한 "전처리 5종 모두 0.879
accuracy에서 멈춤" 현상과 같은 패턴이 AUC 지표에서도 반복됨).

`Binary_Google_ROCAUC_Champion`의 MMSE 트랙에서는 **앵커 단독(0.7623)이
wearable을 결합한 nested 선택 정책(0.7600)보다 오히려 높다** — 정직한 선택
절차가 복잡화를 스스로 거부한 두 번째 사례(발견 1의 `GemmaFeatureProgram`과
같은 패턴).

### 발견 3 — non-nested 선택의 낙관 편향은 실측으로도 꽤 크다

`Binary_Google_MaxAUC_Tuned`(nested 0.7172 vs non-nested 0.8017, optimism
+0.0845)와 `Binary_Google_OrdinalStable`(nested 0.7569 vs 전체 non-nested 최고
0.8103, optimism +0.0534)에서 이미 확인된 패턴이, 새 실험에서도 간접적으로
드러난다. `Binary_Gemma_CognitiveFeature_AUC`(non-nested, 0.7817)와
`Binary_Google_YDF_AUC`(non-nested, 0.7834)의 점수는, 유일하게 nested로 평가한
`GemmaFeatureProgramPipeline`의 MMSE-only 0.7351보다 0.04~0.05 높다. 세 실험이
서로 다른 방법이라 직접 비교는 조심해야 하지만, **"nested로 다시 평가하면
0.78 근처의 점수들이 0.73대로 내려올 가능성"**은 이번 표에서 계속 반복되는
패턴과 일치한다.

### 발견 4 — 웨어러블 전용 신호는 여전히 약하고 불확실성이 크다

현재 wearable-only 최고는 `Binary_Google_ROCAUC_Champion`의 고정 Transformer
anchor, 0.6293(95% CI [0.5343, 0.7245] — **구간 폭이 0.19**). 이전
`SequenceFusion`의 0.6254와 대체로 재현되지만, CI가 넓어 "무작위보다 확실히
낫다"고 말하기엔 아직 여유가 없다. `GemmaFeatureProgramPipeline`의
wearable_only(0.5381)와 program_only(0.5715)도 같은 범위(0.5~0.63)에 머문다.

### 발견 5 — 세 LLM 실험 중 본인 구현(`GeminiFeaturePipeline`)이 가장 낮다

| 실험 | MMSE-only(또는 그에 준하는) 기준선 |
| --- | ---: |
| `Binary_Gemma_CognitiveFeature_AUC` | 0.7748 |
| `GemmaFeatureProgramPipeline` | 0.7351 |
| `GeminiFeaturePipeline`(본인) | **0.6941** |

같은 MMSE 정보를 쓰는데도 `GeminiFeaturePipeline`의 "with MMSE" 기준선이
가장 낮다. 확인되지 않은 가설이지만 유력한 원인 후보:

1. `with` 모드가 MMSE 37개 + wearable BASE 35개(단순 mean/sd) = 72개를 그대로
   합쳐 쓴다 — 다른 두 실험처럼 MMSE 단독 앵커를 따로 유지하지 않는다.
2. downstream 모델이 고정 config 1개(logreg/gbdt, 튜닝 없음)뿐이라, 다른 실험의
   LR C-grid/SVM RBF-grid/rank-blend 같은 선택 여지가 없다.
3. BASE wearable 블록이 mean/sd 요약뿐이라(35개), 다른 실험이 쓰는 112개
   커리티드 wearable 특징보다 정보가 적다.

이는 정확히 5절의 ablation 후보 4번으로 이어진다.

### 발견 6 — 세 신규 실험 폴더의 README가 실행 상태를 잘못 기술하고 있다

`Binary_Gemma_CognitiveFeature_AUC`, `Binary_Google_YDF_AUC`,
`GeminiFeaturePipeline`(본인 것 포함) 세 폴더의 `README_KO.md`가 모두 "아직
실제 API/전체 실행을 하지 않았다"는 취지로 남아 있지만, 실제로는 완료된 run이
존재한다. 이 문서의 수치가 README보다 최신이며, 다음 작업자는 README가 아니라
`*_result/<UTC_ID>/FINAL_REPORT.json`을 1차 근거로 삼아야 한다(README 갱신은
별도 정리 필요).

---

## 3. 별도 참고 — `Binary_Google_ROCAUC_Champion`

`Codex_Dementia_ROCAUC`가 재사용하는 공유 라이브러리인 동시에, 자체 완료 결과도
갖고 있다(`20260728_101249_utc`). MMSE 트랙과 wearable 트랙 모두 "앵커 단독"과
"선택 정책 결합" 두 숫자를 함께 보고하는데, **두 트랙 모두 앵커 단독이 결합보다
높거나 같다** — 발견 2의 핵심 증거원이다.

---

## 4. 다음 Ablation Study 후보 (근거 기반 우선순위)

1. **LLM의 역할을 "특징 생성"에서 "특징 선택/정제"로 전환.** 세 실험 모두
   "새 특징 추가"는 실패했지만, 아무도 "기존 151개(MMSE 39+wearable 112) 뱅크에서
   LLM이 무엇을 유지·제거할지 고르게 하는" 실험은 하지 않았다.
   `GemmaFeatureProgramPipeline`의 nested 선택이 wearable/program을 최빈값으로
   버렸다는 사실은, 사람이 아니라 LLM이 그 "버리는 판단"을 프롬프트로
   수행하도록 만드는 것이 다음으로 시도해볼 자연스러운 방향임을 시사한다.
2. **세 LLM 실험을 동일한 nested 절차로 재평가.** 현재 `Gemma_CognitiveFeature_AUC`와
   `GeminiFeaturePipeline`은 non-nested/미조정이라 "진짜 이득"과 "선택 낙관"이
   섞여 있다. `GemmaFeatureProgramPipeline`이 이미 쓰는 inner-OOF 선택 골격을
   재사용해 나머지 둘도 nested로 다시 돌리면, 발견 3의 가설(0.78대가 0.73대로
   내려오는지)을 직접 검증할 수 있다.
3. **CN vs MCI 서브셋만 분리 평가.** 지금까지의 모든 headline AUC는
   CN(85) vs MCI+Dem(56) 전체를 섞은 값이라, 상대적으로 쉬운 Dem 9명이 점수를
   끌어올린다. `Binary_AUC_TWO_MODELS_ANALYSIS_KO.md`가 인용한 "CN vs MCI ≈
   0.7258, CN vs Dem ≈ 0.9386"이 맞다면, LLM 특징이 정말 유용한지는 **어려운
   CN-vs-MCI 경계에서만 다시 봐야** 판단할 수 있다. 지금까지 이 서브셋 AUC를
   보고한 신규 실험은 없다.
4. **`GeminiFeaturePipeline`의 BASE 블록을 다른 두 실험 수준으로 맞춰 재실행.**
   발견 5의 원인 후보를 검증하려면 (a) MMSE-only 순수 앵커 arm을 별도로 추가,
   (b) BASE wearable을 35개 mean/sd에서 112개 커리티드 특징으로 교체, (c)
   downstream에 최소 2~3개 규제 강도 grid를 허용 — 이 세 가지를 개별
   ablation으로 분리해서 어느 것이 격차를 만드는지 확인.
5. **새 외부 코호트 확보가 구조적으로 최우선.** 이 문서를 포함한 거의 모든
   문서가 같은 141명(그리고 33명 Validation)을 반복 재사용한다는 한계를
   지적한다. 위 1~4번 ablation은 "같은 데이터에서 방법을 바꾸는" 실험이므로
   개선 폭에 원천적 상한이 있다. 방법론 ablation과 별개로, 새 피험자 확보
   경로를 계속 확인해야 한다.
6. **다음 LLM 실험은 시작 전에 `--stage models`(또는 동등한 사전 조회)로
   API 키의 실제 가용 모델을 확인.** `GeminiFeaturePipeline` 개발 중
   `gemini-1.5-flash`(서비스 종료), `gemini-2.5-flash-lite`(신규 키 404),
   `gemini-3.6-flash`(thinking 토큰이 출력 예산을 잠식해 JSON 잘림) 순서로
   3번의 시행착오를 겪은 뒤에야 `gemma-4-31b-it`로 안정화됐다. 이 운영 교훈
   (모델 가용성은 문서가 아니라 키 자신에게 물어야 하고, thinking 지원 모델은
   `max_output_tokens`에 thinking 여유분을 반드시 포함해야 함)을 다음 LLM
   실험 설계 초기에 반영하면 같은 디버깅 반복을 피할 수 있다.

---

## 5. 별도 과제 — Dem Screening (CN+MCI vs Dem, 이 문서의 다른 숫자와 비교 금지)

| 실험 | ROC-AUC | nested | 비고 |
| --- | --- | :---: | --- |
| `Binary_Google_DemRankAUC_select1` | **0.9199**(sd 0.0152) | O(20×5 반복) | 174명(CN+MCI 162 vs Dem 12), 기존 `DemScreen` 0.8284 대비 +0.0915. Dem 양성 12명뿐이라 CI가 이 delta보다 넓을 수 있음에 주의 |
| `Binary_Google_DemScreen` | 0.8284 | — | 이전 참조값 |
| `Codex_Dementia_ROCAUC` | — | — | 코드만 존재, **완료된 실행 결과 없음**(2026-07-30 기준) |

이 과제는 표본 수·난이도가 CN vs MCI+Dem과 다르므로, 위 어떤 점수도 1~4절의
숫자와 나란히 놓고 "더 낫다/못하다"를 말할 수 없다.

---

## 6. 근거 파일 (재현·추적용)

```text
Binary_Gemma_CognitiveFeature_AUC/Binary_Gemma_CognitiveFeature_AUC_result/20260729_225209_utc/
  FINAL_REPORT.json, API_AUDIT.json, LEAKAGE_AUDIT.json, RUN_CONFIG.json

Binary_Google_YDF_AUC/Binary_Google_YDF_AUC_result/20260729_225231_utc/
  FINAL_REPORT.json, POLICY_RESULTS.json, CANDIDATE_RESULTS.json, LEAKAGE_AUDIT.json

GemmaFeatureProgramPipeline/.../20260729_152223_utc/
  FINAL_REPORT.json, RUN_CONFIG.json, FEATURE_PROGRAM.json, FEATURE_PROGRAM_MANIFEST.json

GeminiFeaturePipeline/GeminiFeaturePipeline_result/20260729_133444_utc/
  FINAL_REPORT.json, GEMINI_REPORT.json, RUN_CONFIG.json, TRAINING_REPORT.json

Binary_Google_ROCAUC_Champion/Binary_Google_ROCAUC_Champion_result/20260728_101249_utc/
  training/FINAL_REPORT.json, training/wearable/nested_oof_report.json

Binary_Google_DemRankAUC_select1/Binary_Google_DemRankAUC_result/20260728_135149_utc/
  training/FINAL_REPORT.json
```

*이 문서는 각 실험의 저장된 `FINAL_REPORT.json` 등 실측 산출물을 근거로 작성했다.
새로운 실험이 완료되면 1·2·6절을 함께 갱신하고, 오래된 해석을 그대로 남겨두지
않는다.*
