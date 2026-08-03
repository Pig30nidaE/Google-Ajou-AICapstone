# reproduction_spec.md — 이민지·이석훈(2025) VAE 재현 사양서

작성일: 2026-08-02
최근 갱신: 2026-08-03 (실제 산출물 감사 후 교정 사양 반영)
대상 논문: 이민지·이석훈, 「VAE 기반 데이터 불균형 개선을 통한 치매 조기 탐지 기법」,
*Journal of KIIT*, Vol.23, No.7, pp.1-12, Jul. 31, 2025.
DOI [10.14801/jkiit.2025.23.7.1](https://doi.org/10.14801/jkiit.2025.23.7.1)
파일: `papers/Reproduction/이민지_이석훈_VAE.pdf` (12쪽)

---

## 1. 연구질문

| # | 질문 | 답을 주는 실험 |
| --- | --- | --- |
| RQ1 | 논문 보고 성능이 어느 정도 재현되는가 | A |
| RQ2 | 피험자 중복·전처리·증강 누수를 제거하면 성능이 어떻게 변하는가 | A → B delta |
| RQ3 | VAE 증강이 **새로운 독립 피험자**에 대한 일반화를 실제로 향상시키는가 | B, C |
| RQ4 | non-nested와 nested 평가의 차이는 얼마인가 | B → C delta |

목표는 최고 성능이 아니라 **검증설계에 따른 성능 변화의 정량화**다.

### 1.1 2026-08-03 실제 산출물 감사 판정

`reproduction_lee_lee_2025_vae_result/20260803_013228_full/`은 **재현 실패**로 판정한다.

| 핵심 증거 | 기존 산출물 | 비교값/판정 |
| --- | ---: | ---: |
| A Wide & Deep+VAE, 기록 단위 macro-F1 | 0.4524 | 논문 0.8556 |
| A Wide & Deep+VAE, Dem recall | 0.1633 | 논문 0.8235 |
| B VAE 합성/실제 평균 표준편차 비율 | 약 0.0856 | 분산 붕괴 |
| B TSTR Dem recall | 모든 모델 0 | 합성자료 일반화 실패 |
| C SMOTE 선택 | 2개 fold에서 합성행 0개 | 증강 비교로 무효 |

기존 산출물은 실패 원인을 추적하는 감사 증거일 뿐이며, 재현 결과로 인용하지 않는다.
아래 교정 사양은 코드 정적 수정의 기준이다. 이번 감사에서는 요청에 따라 학습·평가·테스트를
실행하지 않았으므로, 교정본의 유효성은 재실행한 새 산출물로 다시 판단해야 한다.

---

## 2. 논문에서 확인된 설정 (본문·표·그림 실측)

### 2.1 데이터 (§3.1–3.2, 표 1·2·3)

| 항목 | 논문 | 실측 검증 |
| --- | --- | :---: |
| 데이터셋 | AI-Hub 「치매 고위험군 웨어러블 라이프로그」 | — |
| 피험자 | 174명 (CN 111 / MCI 51 / Dem 12) | ✅ 정확 일치 |
| 일별 기록 | 12,183건 (7,737 / 3,661 / 785) | ✅ 정확 일치 |
| 클래스 | CN / MCI / Dem 3-class | ✅ |
| 입력 변수 | 활동 22 + 수면 24 = **46개** | ✅ 컬럼명 100% 일치 |
| MMSE | **사용하지 않음** | 로더가 읽지 않음 |
| AAD | CN에 포함 | 배포본에 이미 반영 |

### 2.2 전처리 (§4.2, §5.1, 그림 1)

| 항목 | 논문 |
| --- | --- |
| 이상치 | §4.2·그림 1 = Isolation Forest / §5.1 = 상·하위 10% (**충돌**, I-1) |
| 이상치 후 행 수 | CN 7,075 / MCI 3,374 / Dem 515 (합 10,964 = 전체의 정확히 90.0%) |
| 스케일링 | StandardScaler, 증강 **이후** 적용으로 읽힘 |
| 범주형 인코딩 | 라벨 인코딩 (적용 대상 없음, I-14) |
| 분할 | train : valid : test = 8 : 1 : 1, **행 단위** (I-6) |

### 2.3 VAE (§5.1, 그림 2)

| 항목 | 논문 |
| --- | --- |
| 입력 차원 | 특성 수와 동일 (= 46) |
| latent | 본문 500 / **그림 2 = 50** (**충돌**, I-2) |
| encoder | 512 → 256 |
| decoder | 256 → 512 |
| 정규화 | Batch Normalization + Dropout 0.3 |
| 손실 | 재구성 오차 + KL 발산의 **가중합** (가중치 미보고) |
| 최적화 | Adam, lr = 1e-4 |
| 생성 | 표준정규 `N(0, I)`에서 latent 샘플링 → decoder |
| 재구성 오차 | 0.0002 (척도 미보고, I-15) |
| 생성 개수 | 미보고 → 표 5에서 **4,000**으로 유도 (I-5) |
| epoch / batch / seed | **전부 미보고** |

### 2.4 분류기 (§5.1)

| 모델 | 논문 보고 | 미보고 |
| --- | --- | --- |
| XGBoost | multi:softmax, max_depth=6, lr=0.1 | n_estimators, subsample, reg_*, seed |
| DNN | 512-256-128-64-32, ReLU, L2, BatchNorm, Dropout 0.5, softmax | L2 계수, lr, epoch, batch, early stopping |
| TabNet | n_d = n_a = 64, n_steps = 5 | gamma, lambda_sparse, epoch, batch |
| Wide & Deep | Deep 256-128-64, ReLU, Dropout 0.3 | **Wide 입력이 무엇인지**, lr, epoch, batch |
| 공통 | cross-entropy, Adam | lr, epoch, batch, early stopping, seed |

### 2.5 보고 결과 (표 6 = 정본, §5.2 본문은 오기 I-3)

Wide & Deep, 증강 전/후:

| Class | P (before) | R (before) | F1 (before) | P (VAE) | R (VAE) | F1 (VAE) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CN | 0.9107 | 0.9223 | 0.9165 | 0.9332 | 0.8501 | 0.8897 |
| MCI | 0.8398 | 0.8373 | 0.8385 | 0.7340 | 0.8843 | 0.8022 |
| Dem | 0.9070 | 0.7647 | 0.8298 | 0.9333 | 0.8235 | 0.8750 |
| **Avg (macro)** | 0.8858 | 0.8414 | **0.8616** | 0.8668 | 0.8526 | **0.8556** |

그림 3 (증강 후 조건으로 판단, I-12):

| 모델 | CN F1 | MCI F1 | Dem F1 | Avg F1 |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 0.8914 | **0.7501 (그림) / 0.7581 (본문)** | 0.7816 | 0.8103 (보고값) |
| DNN | 0.8958 | 0.7770 | 0.7527 | 0.8085 |
| TabNet | 0.8762 | 0.7485 | 0.7391 | 0.7879 |
| Wide & Deep | 0.8897 | 0.8022 | 0.8750 | 0.8556 |

**평가 단위: 일별 기록(row).** 논문은 피험자 단위 집계를 하지 않는다.
복원된 평가셋 크기: 증강 전 N=1,097 / 증강 후 N=1,095 (**서로 다름**, I-4).
XGBoost MCI 값은 그림과 본문이 충돌하며(I-18), 보고 Avg 0.8103은 본문 값 0.7581을 넣을 때만
산술적으로 맞는다. 또한 Wide & Deep의 macro-F1은 증강 전 0.8616이 VAE 후 0.8556보다 높다.
따라서 논문 자체도 “VAE가 전체 평균 성능을 개선했다”는 근거를 제공하지 않는다.

---

## 3. 채택한 primary reproduction 설정

사용자 지시 "논문 본문의 구체적인 실험 설명을 우선"에 따라:

```yaml
# configs/paper_percentile_latent500.yaml  ← primary
outlier.method: percentile      # §5.1 본문
outlier.percentile.q: 0.10
vae.latent_dim: 500             # §5.1 본문
```

**이 설정을 "원 저자 코드의 확정 사양"이라고 부르지 않는다.**
문서·로그·결과표 어디에서나 **primary reported-method reconstruction**으로만 표기한다.
`configs/*.yaml`의 `label:` 필드와 결과 CSV의 `config_label` 열에 이 문자열이 그대로 들어간다.

> ⚠️ `report_inconsistencies.md` I-1의 산술 증거는 **A3(Isolation Forest + latent 500)**이
> 논문의 행 수를 재현할 가능성이 높음을 시사한다. primary를 A1로 두는 것은 사용자 지시를
> 따른 것이며, A1이 행 수를 재현하지 못하는 것은 **예상된 결과이자 보고 대상**이다.

### 민감도 config (실험 A)

| config | outlier | latent | 역할 |
| --- | --- | ---: | --- |
| `paper_percentile_latent500.yaml` | percentile 10% | 500 | **primary** |
| `paper_percentile_latent50.yaml` | percentile 10% | 50 | 그림 2 해석 |
| `paper_isoforest_latent500.yaml` | IsolationForest(0.1) | 500 | raw VAE forensic 변형 + 기존 실패 산출물 |
| `paper_isoforest_latent50.yaml` | IsolationForest(0.1) | 50 | §4.2 + 그림 2 |
| `paper_isoforest_scaled_latent500.yaml` | IsolationForest(0.1) | 500 | **감사 후 교정 기본 A5** |

추가 축(각 config에서 override 가능):

- `outlier.percentile.scope`: `global` | `per_class`
- `outlier.percentile.action`: `drop_row` | `clip`
- `preprocessing.scaler_scope`: `all_data` | `train_with_synthetic` | `train_real_only`
- `augmentation.vae.fit_scope`: 현재 교정 실행은 `train_dem_only`만 허용
- `augmentation.vae.input_space`: `raw` | `scaled`

2026-08-03 감사 후 교정 기본 실행은 **Isolation Forest(0.1) + scaled VAE 입력 +
`train_dem_only`**를 사용한다. 기존 raw 입력 variant는 논문 서술 순서를 추적하는 forensic
민감도 실험으로만 보존한다. 이는 원 저자 설정의 확정이 아니라 실패 원인을 통제하기 위한
재구성 가정이다.

---

## 4. 실험 A — `paper_reported_reconstruction`

**목적**: 논문이 서술한 절차를 그대로 재구성한다. 검증방법이 적절하다고 전제하지 않는다.

```
1. activity + sleep 합본 (12,183행 × 46변수)
2. 이상치 제거          ← 전체 데이터에서 fit  (누수, 의도적)
3. 행 단위 8:1:1 분할   ← 피험자 중복 발생 (누수, 의도적)
4. 교정 설정의 scaler로 train Dem을 변환하고 VAE 학습 (`train_dem_only`만 허용, fit 감사 기록)
5. 합성 Dem 4,000행 생성 → train에만 추가
6. 합성자료를 원 단위로 복원·유효성 검사하고 분류기 입력 공간으로 변환
7. XGBoost / DNN / TabNet / Wide&Deep 학습
8. 기록 단위 metric 산출 → 논문 표 6·그림 3과 병기
```

**누수 감사기는 `observe` 모드**로 동작한다. 위반을 예외로 던지지 않고 **측정해 보고**한다:

- train/test 피험자 교집합 크기와 클래스별 실제 평가 피험자 수
- 이상치 detector가 본 test 행 수
- scaler가 본 test 행 수
- VAE source/test 피험자 교집합과 VAE가 직접 본 test 원시행 수(교정본 기대: 0)

증강 전/후는 **반드시 같은 split·같은 seed**에서 비교한다
(논문은 그러지 않았다, I-4 — 이 사실은 결과표 각주로 보고).

**산출물**: `outputs/A_<label>/` 아래
`record_level_metrics.csv`, `paper_comparison.csv`, `row_counts.csv`,
`leakage_observation.json`, `per_model/*.json`.

---

## 5. 실험 B — `leakage_controlled_non_nested`

**목적**: 논문 하이퍼파라미터를 **고정**한 채 피험자 분리와 전처리·VAE 범위만 통제한다.
하이퍼파라미터를 다시 선택하지 않는다.

### 강제 순서 (감사기가 각 단계를 기록·검증)

```
1. 피험자 테이블 기준 fold 분리      subject_stratified
2. Isolation Forest fit ← train 피험자만 (A와 같은 방법·contamination)
3. imputer fit       ← train 피험자만
4. scaler fit        ← train 피험자만
5. VAE fit           ← train fold의 실제 Dem 기록만
6. 합성 Dem 추가     ← train fold에만
7. 분류기 학습
8. 평가 피험자에는 학습된 변환만 적용
9. 평가 피험자에 합성자료 절대 미추가
```

- 기본 3-fold, `n_repeats` 지원, 모든 모델이 **동일 split** 사용.
- 각 fold에 CN·MCI·Dem 피험자가 모두 존재하는지 검증 (없으면 `SplitError`).
- 감사기 `enforce` 모드: 위반 시 즉시 `LeakageError`.

### 평가

- 일별 예측확률을 **피험자별 평균**으로 집계 → 피험자 단위가 **주 평가**.
- 기록 단위는 보조 결과로 함께 보고.
- synthetic row는 평가 단위에 절대 포함하지 않는다.

---

## 6. 실험 C — `nested_subject_independent`

```
Outer: subject_stratified(n_splits=3), n_repeats=1 (10까지 설정 가능)
Inner: subject_stratified(n_splits=3)
group: 피험자 ID
```

Dem 피험자가 12명이므로 **outer 5-fold를 기본값으로 쓰지 않는다**
(3-fold에서 fold당 Dem 4명, 5-fold면 2~3명).

### inner CV 안에서만 선택하는 항목

`outlier.method`, `percentile.q`(제한된 후보), `isoforest.contamination`,
`augmentation.method`, `vae.latent_dim`(50/500), `synthetic_ratio`, `vae.epochs`,
`vae.lr`, `vae.batch_size`, `classifier`, 분류기 하이퍼파라미터, `class_weight`,
필요시 `decision_rule`.

탐색은 config의 `search.space`로 제한하고 `search.max_evals`로 상한을 둔다.
상한을 넘으면 분류기×증강법 arm을 균형 있게 순회하고 각 arm 내부만 seed 기반으로
무작위화한다(Optuna 미사용).

후보를 만들 때는 선택된 `augmentation.method`와 `outlier.method`에 실제로 활성화되는 축만
남긴 뒤 정규화하고, 의미가 같은 후보는 중복 제거한다. 예를 들어 SMOTE 후보에 VAE의 latent,
epoch, ratio 축을 곱하지 않는다. 증강법을 선택한 후보가 `n_synthetic=0`이면 유효한 증강 후보로
간주하지 않고 오류로 처리한다. `max_evals`가 전체 유효 후보보다 작으면
분류기×증강법 조합별 round-robin으로 예산을 배분한다.

### outer test에 절대 사용 금지

이상치 임계값 선택 / scaler fit / VAE fit / synthetic ratio 선택 / 모델 선택 /
early stopping / threshold 선택. 감사기가 각각을 `record_selection()`으로 추적한다.

---

## 7. 평가 규약

**주 평가 단위: 피험자.** 일별 예측확률을 피험자별로 평균해 3-class 확률을 만든다.

| 구분 | 지표 |
| --- | --- |
| 주 | macro-F1, balanced accuracy, macro OvR ROC-AUC, macro PR-AUC, 클래스별 P/R/F1 |
| 보조 | accuracy, weighted-F1, confusion matrix, 기록 단위 metric, multiclass Brier, log loss |
| Dem 전용 | Dem recall / precision / F1, fold별 Dem 피험자 수, 정분류된 Dem 피험자 수 |

모든 결과표에 **"독립 Dem 피험자 12명"** 제한을 명시 출력한다.
불확실성은 **피험자 단위 bootstrap 95% CI**로 보고한다.

---

## 8. 실행

```bash
python run.py --config configs/paper_percentile_latent500.yaml
python run.py --config configs/leakage_controlled_non_nested.yaml
python run.py --config configs/nested_subject_independent.yaml
```

옵션: `--inspect-data`, `--audit-only`, `--dry-run`, `--fold`, `--seed`, `--resume`,
`--skip-vae`, `--augmentation {none,vae,class_weight,random_oversampling,smote}`.

---

## 9. 재현 수준 선언

본 재현이 달성하는 수준은 **method-level reconstruction**이다.
**exact reproduction은 원리적으로 불가능**하며 근거는 다음과 같다.

| # | 근거 | 상세 |
| --- | --- | --- |
| 1 | 이상치 기법이 본문과 그림에서 충돌하고, 본문 방법으로는 보고된 행 수가 나오지 않음 | I-1 |
| 2 | latent dimension이 본문 500 / 그림 50 | I-2 |
| 3 | 증강 전/후가 서로 다른 평가셋에서 측정됨 (N=1,097 vs 1,095) | I-4 |
| 4 | random seed 전면 미보고 (split·VAE·분류기 모두) | — |
| 5 | epoch / batch size / early stopping 기준 미보고 | — |
| 6 | VAE의 fit 범위(전체 Dem인지 train Dem인지) 미보고 | I-9 |
| 7 | scaler의 fit 범위 미보고 | I-8 |
| 8 | 10% 임계값 선택에 쓴 자료 미보고 | I-10 |
| 9 | Wide 컴포넌트 입력 미보고 | — |
| 10 | KL 가중치(beta) 미보고 | — |

따라서 본 재현은 **논문 수치와의 일치가 아니라, 논문 절차를 재구성했을 때 얻어지는 수치와
검증설계를 강화했을 때의 변화량**을 결과로 제시한다.

2026-08-03 기존 실행은 이 method-level 목표에도 도달하지 못했다. 교정본 재실행 전까지
현재 상태는 **“재현 실패, 원인 교정 완료 후 검증 대기”**이며, 실패 산출물과 새 산출물을
혼합 집계하지 않는다.
