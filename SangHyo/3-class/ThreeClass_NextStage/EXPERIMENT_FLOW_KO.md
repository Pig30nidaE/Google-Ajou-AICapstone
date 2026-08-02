# 3-class 분류 다음 단계 실험 설계

기준일: 2026-07-14  
대상: `CN(0) vs MCI(1) vs DEM(2)` subject-level 동시점 분류  
실행 환경: Google Colab Pro+ / A100 / High-RAM

## 1. 출발점과 이번 실험의 목적

기존 `ThreeClass_Classification/3class_subject_ensemble_colab.ipynb`의 leakage-safe nested CV 결과는 다음과 같다.

| 지표 | 결과 |
| --- | ---: |
| subject-level nested-CV Macro F1 | 0.3579 |
| balanced accuracy | 0.3589 |
| accuracy | 0.5319 |
| macro OVR AUROC | 0.5300 |
| multiclass log loss | 0.9576 |
| MCI recall / F1 | 0.2128 / 0.2500 |
| DEM recall / F1 | 0.1111 / 0.1429 |

최종 training OOF 선택 점수는 0.4559였지만 nested CV는 0.3579였다. `+0.098`의 차이는 후보·blend·class scale 선택이 training OOF에 과적합했을 가능성을 보여준다. 또한 DEM scale이 탐색 상한 2.5에 붙었고, prior-only log loss 약 0.8469보다 nested-CV log loss가 나빴다. 따라서 이번 실험은 단순히 후보 수를 늘리는 것이 아니라 다음을 동시에 해결한다.

1. 작은 표본에 강한 최신 tabular foundation model을 추가한다.
2. MCI/DEM의 낮은 분리력을 직접 확인할 수 있는 balanced·pairwise 후보를 비교한다.
3. 2026년 동일 코호트 연구가 제시한 35일 이상 변동성 특징과 35일 sequence를 동일 subject fold에서 비교한다.
4. fold 밖에서 feature selection, calibration, blend, threshold를 학습하지 않는다.
5. class별 사후 배율 탐색을 제거해 selection overfitting을 줄인다.

## 2. 논문·최신 자료를 반영한 모델 선택

### 채택

- **TabPFN-3**: 이번 단계의 핵심 신규 후보다. `tabpfn 8.1.0`에서 `ModelVersion.V3`와 `inference_precision=torch.float32`를 명시적으로 요청하며, 현재 데이터의 `n=141`과 최대 약 3,700개인 feature view는 공식적으로 제시된 `1,000행 × 20,000열` 범위 안이다. 외부 scaling/OHE를 하지 않는 native compact view와 fold-local stability-filtered view를 비교한다. imbalance는 공식 `balance_probabilities`의 on/off만 비교한다.
- **LDA와 Elastic-Net Logistic**: 2026년 동일 코호트 CN/MCI 연구와 기존 실험 모두 작은 표본에서 선형 모델의 가치를 지지한다. LDA는 compact 35일 view, Elastic-Net은 compact 35일·multi-window와 legacy anchor에 사용한다.
- **얕은 CatBoost**: compact 35일 view에서만 강하게 정규화한 비선형 기준선으로 사용한다. 후보 수를 통제하기 위해 다른 boosting 계열은 이번 고정 후보군에 넣지 않는다.
- **pairwise TabPFN-3**: `CN-MCI`, `CN-DEM`, `MCI-DEM` 세 binary model의 log-probability vote를 하나의 사전 고정 후보로 평가한다. rare-class 개선 여부는 inner/outer CV에서만 판단한다.
- **35일 MiniRocket**: 작은 표본에서도 강한 시계열 분류 기준선이다. 값·관측 mask·관측 후 경과일을 입력한다.
- **35일 mask-aware tiny TCN**: dilation `1, 2, 4, 8`, block당 convolution 2개와 masked mean/last-observed pooling을 사용한다. hidden dimension 24, dropout 0.35의 작은 모델로 제한한다.

MiniRocket 실행에는 `sktime==1.0.1`과 `numba==0.62.1`을 사용한다. 이 `sktime` 버전의 공식 의존성 상한이 `scikit-learn<1.8`이므로 설치 충돌을 피하기 위해 두 notebook 모두 `scikit-learn==1.7.2`로 고정한다.

### 이번 notebook에서 보류

- **TabM**은 ICLR 2025의 강한 tabular DL 후보지만 141명에서 추가 HPO 후보를 늘리면 selection overfitting 위험이 커진다. TabPFN-3가 개선되지 않을 때 stable 64/128에 한해 다음 독립 ablation으로 진행한다.
- **MantisV2**는 최신 분류용 시계열 foundation model이지만 35일을 권장 길이 512로 보간해야 해 강한 pretraining mismatch가 생긴다. TCN/MiniRocket이 보완성을 보인 뒤 frozen embedding만 별도 notebook에서 평가한다.
- **PatchTST/MOMENT/TimesNet/MambaSL**은 각각 forecasting 중심 근거, 지나치게 긴 권장 입력, 짧은 35-step에서 불안정한 주기 추정, 설치 복잡도 때문에 1차 후보에서 제외한다.
- **SMOTE/GAN/diffusion 증강**은 DEM 9명의 관측을 복제·합성해 CV 변동성을 가릴 위험이 있어 사용하지 않는다.

로컬 `Papers/`의 자료도 다음처럼 설계 근거와 한계를 분리해 사용했다.

- 「라이프로그 데이터를 활용한 LSTM 모델 기반의 치매 예측」: activity/sleep 연속열을 함께 쓰는 sequence branch의 가설 근거
- 「라이프로그 데이터를 활용한 랜덤포레스트 및 SHAP 기반 인지기능 장애 예측 모델」: feature 축소, sleep/activity family, 성능 동결 후 XAI의 근거
- 「라이프로그를 이용한 치매 고위험군 예측 시스템 제안」, 「웨어러블 디바이스를 활용한 치매 예측 라이프로그 분석」, 「설명가능 인공지능을 활용한 라이프로그 기반 치매 위험도 산정 방법에 관한 연구」: wearable/lifelog 표현과 설명가능성의 배경 자료

이 자료들은 주로 binary 분류이며 subject-grouped 3-class 결과가 아니므로 논문 점수를 목표 성능으로 재사용하지 않는다.

2026년 Yonsei Medical Journal의 CN/MCI 연구는 최소 35일 wearable 자료에서 단순 평균보다 sleep·heart-rate의 longitudinal variability와 distributional feature를 결합했을 때 분리가 좋아졌다고 보고했다. 특히 median/trimmed mean, 범위·MAD·IQR·CV, consecutive difference, 7일 rolling variability, time-bin 변화와 `HR_drop` 계열을 이번 compact feature 가설에 반영한다. 그러나 논문의 `111 CN + 51 MCI = 162명`은 이 저장소의 training/validation CN·MCI 합계와 정확히 같아 **동일 AI-Hub 코호트일 가능성이 매우 높다**. 따라서 이 논문은 독립 외부 근거가 아니라 feature hypothesis로만 사용한다. 논문을 보고 설계한 뒤 이 저장소의 historical validation을 평가하면 그 benchmark 역시 문헌을 통한 간접 오염 가능성이 있으므로, 최종 일반화 확인에는 별도 외부 코호트가 필요하다.

주요 외부 근거:

- [Longitudinal variability of wearable-derived sleep and heart rate for MCI detection, Yonsei Medical Journal 2026](https://eymj.org/DOIx.php?id=10.3349%2Fymj.2025.0575)
- [TabPFN-3 technical report](https://arxiv.org/abs/2605.13986)
- [TabPFN 8.1.0 official release](https://github.com/PriorLabs/TabPFN/releases/tag/v8.1.0)
- [TabPFN official repository and usage constraints](https://github.com/PriorLabs/TabPFN)
- [TabPFNv2, Nature 2025](https://www.nature.com/articles/s41586-024-08328-6)
- [TabArena, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/1697e3fb412da11dc9488249f9e7bbc9-Abstract-Datasets_and_Benchmarks_Track.html)
- [TabM, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html)
- [MiniRocket](https://arxiv.org/abs/2012.08791)
- [TCN](https://arxiv.org/abs/1803.01271)
- [GRU-D missingness representation](https://www.nature.com/articles/s41598-018-24271-9)
- [MantisV2](https://arxiv.org/abs/2602.17868)
- [Stability selection](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2010.00740.x)
- [scikit-learn probability calibration guidance](https://scikit-learn.org/stable/modules/calibration.html)

## 3. 고정된 후보군

후보 수와 설정은 official validation label을 보기 전에 아래처럼 고정한다.

| 후보 | 입력 | 목적 |
| --- | --- | --- |
| `lda_compact35_k32` | compact 35일 stable 32 | 저분산 선형 판별 기준선 |
| `elastic_compact35_k32` | compact 35일 stable 32 | sparse 선형 기준선 |
| `elastic_compact_multi_k48` | compact 35/50/70일 stable 48 | multi-window 변동성 효과 |
| `cat_compact35_k48` | compact 35일 stable 48 | 얕은 비선형 기준선 |
| `elastic_legacy_all_k64` | 기존 전체 summary stable 64 | 직전 실험과 연결되는 anchor |
| `tabpfn3_compact35_native_raw` | compact 35일 전체 usable feature | 외부 scaling/imputation 없는 native 기준 |
| `tabpfn3_compact35_k64_raw` | compact 35일 stable 64 | 고차원 잡음 제거 효과 |
| `tabpfn3_compact35_k64_balanced` | compact 35일 stable 64 | 공식 prior balancing 효과 |
| `tabpfn3_compact_multi_k96_balanced` | compact 35/50/70일 stable 96 | multi-window와 balancing 결합 |
| `tabpfn3_pairwise_compact35_k64` | compact 35일 stable 64 | MCI/DEM pairwise 분해 |
| `minirocket_35d` | 35일 값+mask+delta | 비신경 sequence 기준선 |
| `mask_tcn_35d` | 35일 값+mask+delta | 작은 supervised temporal model |

후보 이름, feature budget, balancing 여부와 blend 탐색 공간은 training-only discovery 시작 전에 고정한다. 새로운 후보를 결과를 본 뒤 같은 run에 추가하지 않는다.

## 4. feature와 sequence 계약

- prediction index: subject별 마지막 activity date
- prediction horizon: 해당 시점의 cognitive status 동시점 분류이며 미래 치매 전환 예측이 아님
- legacy lookback: 직전 실험과 비교하기 위한 7/14/28일 subject summary
- compact lookback: 마지막 35/50/70 calendar days의 분포·변동성 summary, primary window는 35일
- sequence lookback: 마지막 35일 daily sequence
- sleep date: `sleep_bedtime_end`의 날짜
- 같은 wake-date의 sleep이 여러 개면 가장 긴 main sleep을 대표 행으로 고정하고 episode 수만 별도 daily count로 남기며 `sleep_period_id` 자체는 사용하지 않음
- 절대 날짜, 이메일, sample 순서, 진단, MMSE 총점/문항, 의사 정보는 사용하지 않음
- raw sequence placeholder `...`는 feature로 사용하지 않음
- validation에만 나타난 feature는 training contract에 맞춰 제거

compact 후보는 sleep duration/stage/efficiency, HR·HR drop, RMSSD, breathing, onset/restless/temperature, stage ratio·entropy·transition과 activity MET·steps·intensity·nonwear·entropy·relative amplitude를 일별로 만든다. 각 35/50/70일 calendar grid에서 mean, median, 10% trimmed mean, robust mode, min/max/range, MAD, kurtosis, SD/CV/IQR, consecutive-difference, 7일 rolling SD/CV, 4개 time-bin 변화량을 계산한다. 관측이 없는 날짜도 calendar grid에 남겨 missingness가 시간 간격과 뒤섞이지 않게 한다.

stable feature bank는 각 train fold 내부에서만 다음을 수행한다.

1. 결측률/상수 feature 제거
2. training median과 0.5/99.5 percentile 기록
3. class-stratified 80% subsampling을 full mode 16회 수행
4. multiclass ANOVA와 `CN-MCI`, `MCI-DEM`, `CN-DEM` binary contrast ranking의 선택 빈도와 점수 결합
5. 매우 높은 상관 feature를 완화한 deterministic ranking 생성
6. 후보별 32/48/64/96개 slice 생성

이는 DEM 표본이 매우 작기 때문에 고전 stability selection의 오류율 보장을 주장하지 않는 **stability filter**다. 선택 빈도와 최종 feature 목록은 artifact로 저장한다.

35일 sequence는 training에서 고정한 domain feature 목록만 사용한다. fold train의 median/IQR로 값 채널을 변환하고, 원래 관측 mask와 channel별 `days-since-observed / 35`를 추가한다. outer-valid나 benchmark 통계는 fit에 사용하지 않는다.

Training source만 사용한 사전 coverage 점검에서는 마지막 activity date를 기준으로 한 35 calendar-day 창에 sleep night가 35개 미만인 subject가 `125/141`이었다. 따라서 논문의 “최소 35일 wearable 자료” 조건이 이 데이터에서 그대로 충족된다고 가정하지 않는다. 실제 run은 subject별 activity-day, sleep-night, paired-day 수와 class별 coverage를 artifact로 저장하고, mask/delta 및 coverage feature로 불완전 관측을 명시한다.

## 5. 평가 흐름

두 notebook은 물리적으로 역할을 분리한다.

```text
01_train_only_discovery_colab.ipynb
  └─ Data/1.Training만 접근: validation source·label 경로 자체가 없음
       ├─ outer repeated StratifiedKFold: 2 seeds × 3 folds
       │    └─ 각 outer-train 안에서 inner 2 seeds × 3 folds
       │         ├─ fold-local stability bank/sequence scaler fit
       │         ├─ 고정 12개 후보 OOF 확률 생성
       │         └─ 사전 고정 sparse blend + global temperature 선택
       ├─ outer-valid에서 선택 절차와 모든 개별 후보 평가
       ├─ 전체 Training repeated OOF: 2 seeds × 3 folds
       └─ 선택 component별 전체 Training refit 2 seeds 확률 평균
            └─ sparse rule·feature contract·모델 동결 및 hash 저장

02_frozen_benchmark_colab.ipynb
  └─ 완료된 frozen training artifact와 hash 검증
       ├─ label을 열지 않고 validation source를 deterministic transform
       ├─ frozen model로 확률 생성 및 입력 manifest 동결
       └─ 명시적 historical-reuse 확인문 뒤 label을 단 한 번 로드해 평가
```

첫 notebook은 `Data/2.Validation`에 전혀 접근하지 않는다. 두 번째 notebook도 모델, feature, blend, temperature를 변경할 수 없으며 완료 marker가 있으면 같은 frozen run의 재평가를 거부한다. 이 benchmark는 과거 반복 사용되었고 2026년 동일 코호트 논문을 통한 간접 오염 가능성도 있으므로 결과를 독립 holdout 성능으로 부르지 않는다.

primary metric은 subject-level Macro F1이다. Nested 성능의 주 추정치는 각 outer seed가 만든 완전한 OOF 결과 2개의 평균과 sample SD이며, 각 outer-valid 후보도 실제 배포와 동일하게 2개 full-training refit seed의 확률을 평균한다. 6개 outer fold mean/SD와 legacy anchor 대비 paired delta 방향도 함께 저장한다. 두 outer repeat의 subject별 확률까지 다시 평균한 지표는 실제 frozen 2-refit ensemble보다 더 많은 split-model을 섞으므로 `averaged repeated-OOF diagnostic`으로만 보고한다. 모든 모델 입력은 subject당 한 행 또는 한 sequence이므로 window vote를 별도로 만들지 않는다. 함께 저장하는 지표는 balanced accuracy, accuracy, per-class precision/recall/F1, confusion matrix, OVR AUROC/AUPRC, multiclass log loss다. Bootstrap CI는 고정된 OOF prediction에 조건부인 구간이며 split/model 변동성은 outer repeat/fold SD로 별도 해석한다.

blend 탐색은 single model, equal-weight pair/triple, `0.75/0.25`, `0.50/0.25/0.25`만 사용한다. 최고 OOF Macro F1에서 0.01 이내면 더 단순한 rule을 선택한다. class별 scale/threshold grid와 isotonic calibration은 사용하지 않는다. global temperature 하나도 inner-OOF log loss가 최소 0.005 개선될 때만 채택하며 argmax/Macro F1 개선으로 해석하지 않는다.

Nested inner selection, 전체 Training 최종 rule OOF, frozen deployment refit을 모두 2 seeds로 통일한다. 따라서 outer-valid에서 평가되는 선택·ensemble 절차와 실제 배포 계약의 반복 수가 일치한다. 단계별 split/model seed 값은 독립적으로 고정해 저장한다.

Full 설정은 inner selection과 deployment-matched outer 평가를 합친 nested `576`회, final OOF `72`회, 합계 약 `648` candidate fit에 선택 component의 2-seed full refit이 추가된다. A100에서도 한 Colab 세션을 넘길 수 있으므로 fold checkpoint를 Drive에 두고 같은 run ID로 재실행해 이어간다. `FAST_MODE=True`는 2 folds, 각 seed 1개, full refit 1개로 줄인 약 96-fit 기능 검증이지만 TabPFN이 포함되어 즉시 끝나는 unit test는 아니다. Smoke artifact는 benchmark notebook이 거부한다.

CV의 임시 TabPFN 후보는 fold 확률을 얻는 즉시 CPU 이동·참조 해제·CUDA cache 정리를 수행한다. 최종 refit의 각 TabPFN은 학습 직후 공식 `.tabpfn_fit` 파일로 내려 GPU 메모리를 비우며, benchmark에서도 한 refit만 GPU에 올려 예측한 뒤 해제한다. 선택 component마다 seed `910000–910001`의 확률을 먼저 평균한 후 동결 sparse rule과 temperature를 적용한다. V3 foundation checkpoint는 어떤 CV checkpoint보다 먼저 resolve/download·SHA-256 계산하며 이 identity를 run ID에 포함한 뒤 즉시 snapshot한다. 학습 종료 시 cache와 snapshot hash를 다시 확인하므로, 같은 package/filename 아래 가중치가 바뀌거나 partial run 중 checkpoint가 교체된 혼합 run을 완료할 수 없다.

## 6. 결과 채택 기준

다음 단계 모델을 “개선”으로 부르려면 nested-CV에서 아래를 함께 확인한다.

- 선택 pipeline Macro F1이 기존 0.3579보다 최소 약 `+0.02` 개선
- outer-fold paired delta가 대부분 같은 방향
- MCI recall/F1이 개선되고 DEM 개선이 한두 subject의 우연한 적중만으로 설명되지 않음
- log loss가 prior-only 기준보다 더 나빠지지 않거나 명확히 개선
- selection OOF와 nested CV의 간격이 기존 `0.098`보다 감소
- 선택 tabular feature가 coverage/valid-day 변수에 과도하게 집중되거나 sequence branch가 mask만으로 개선되는 정황이 있으면 acquisition-protocol shortcut으로 간주하고 benchmark 전에 no-coverage/no-mask ablation을 별도 동결
- 새 모델이 anchor를 이기지 못하면 더 단순한 LDA/Elastic-Net/CatBoost 모델을 유지

DEM은 training 9명, historical validation 3명이므로 어떤 결과에도 넓은 불확실성과 per-class support를 함께 보고한다.

## 7. Colab 실행 순서

Notebooks:

1. `ThreeClass_NextStage/01_train_only_discovery_colab.ipynb`
2. `ThreeClass_NextStage/02_frozen_benchmark_colab.ipynb`

1. Prior Labs 계정에서 TabPFN-3 non-commercial model license를 확인한다.
2. Colab Secrets에 Prior Labs용 `TABPFN_TOKEN`, gated model 다운로드용 별도 `HF_TOKEN`, 32자 이상의 임의 비밀 문자열 `SUBJECT_HASH_KEY`를 등록하고 notebook access를 허용한다. 해당 Hugging Face model repository access도 먼저 승인받는다. 비밀값은 출력하거나 artifact에 저장하지 않으며, benchmark에도 동일한 `SUBJECT_HASH_KEY`를 사용한다.
3. A100 GPU와 High-RAM runtime을 선택한다. 두 notebook은 A100이 아니면 중단하며 benchmark는 discovery에 기록된 accelerator 문자열, package/Python 버전, 명시적 TabPFN float32 계약까지 동일한지 확인한다.
4. discovery notebook 설정 cell에서 `PROJECT_ROOT_OVERRIDE`, `DATA_ROOT_OVERRIDE`, `RESULT_BASE_OVERRIDE`를 필요한 경우만 바꾼다.
5. `01_train_only_discovery_colab.ipynb`를 먼저 `FAST_MODE=True`로 smoke test하고, 별도 output root에서 `FAST_MODE=False` full run을 실행한다.
6. `nested_cv_report.json`, `nested_candidate_metrics.csv`, `nested_fold_metrics.csv`, `candidate_oof_metrics.csv`, feature stability와 frozen contract를 검토한다. 이 시점까지 validation source와 label은 모두 접근하지 않는다.
7. `FAST_MODE=False`인 full training 완료 marker와 artifact hash가 생성된 뒤에만 `02_frozen_benchmark_colab.ipynb`에 frozen run 경로를 지정한다. Benchmark notebook은 smoke run을 거부한다.
8. benchmark notebook의 경고문을 읽고 정확한 acknowledgment 문자열을 입력했을 때만 historical validation을 단 한 번 평가한다. 가능하면 이 단계보다 독립 외부 cohort 평가를 우선한다.

동일한 공용 TabPFN cache를 쓰는 discovery/benchmark run을 동시에 실행하지 않는다. Benchmark는 frozen snapshot과 cache hash가 다르면 해당 run의 정확한 V3 checkpoint로 cache를 복원하기 때문이다.

Discovery notebook은 완료 run을 덮어쓰지 않고 partial nested-CV/final-OOF checkpoint를 재개한다. Benchmark notebook은 frozen artifact의 hash와 training subject hash를 검증하고, label을 열기 전에 unlabeled prediction과 입력 manifest를 저장한다. TabPFN 모델은 공식 fitted-model 포맷으로 저장하고 정확한 V3 foundation checkpoint도 hash와 함께 보존한다. subject 식별자는 keyed salted hash만 artifact에 남긴다.

## 8. 산출물

- 환경·입력 fingerprint·data audit JSON
- 실행한 discovery notebook과 core module의 exact code snapshot/hash
- training feature와 sequence cache
- feature manifest와 full-train stability ranking CSV
- nested fold assignment/metric, fold별 candidate paired metric, OOF 확률
- 최종 sparse blend rule과 selection OOF report
- frozen model bundle 및 TabPFN fitted-model files
- exact TabPFN-3 foundation checkpoint와 SHA-256 manifest
- `FINAL_TRAINING_REPORT.json`, frozen configuration, model/hash manifest와 training 완료 marker
- 별도 benchmark notebook이 만든 validation-source cache, label 없는 prediction CSV와 frozen-input manifest
- acknowledgment 후 단 한 번 생성되는 historical validation metric/CI/confusion matrix와 benchmark 완료 marker
