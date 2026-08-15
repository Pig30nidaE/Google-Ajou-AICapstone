# 재현 검증 코드

`REPORT.md`의 모든 수치를 생성하는 스크립트. 실행 전 각 파일 상단의 `DATA_ROOT`를
AI Hub 원본 데이터 경로로 수정할 것.

```
의존성: numpy pandas scipy scikit-learn statsmodels
```

## YMJ 논문(2026) 재현 — 실행 순서

| 순서 | 스크립트 | 하는 일 | 대략 소요 |
|---|---|---|---|
| 1 | `ymj_faithful_build.py` | 원본 수면 CSV → 일별 파생지표 20종 → 환자단위 5계열 피처(339개) | ~5초 |
| 2 | `ymj_validate_table3.py` | 논문 Table 3(t-test/LMM/FDR) 재현 → 파생지표 정확성 독립 검증 | ~2초 |
| 3 | `ymj_faithful_experiments.py` | Table 1의 12개 실험을 non-nested / nested 두 방식으로 평가 | ~1분 |
| 4 | `ymj_missed_variants.py` | 놓친 명세 변형 4종(V0~V3) 테스트 | ~3분 |
| 5 | `ymj_final_robustness.py` | 윈도우·최소야간수·시드 감도 (총 51회 nested 평가) | ~1분 |
| 6 | `ymj_perm_bootstrap.py` | Permutation test(1000회) + Bootstrap CI(2000회) | ~30분 |
| 7 | `rq2_validate_and_ladder.py` | Part1: 논문 Suppl. Table 6(7개 모델) 재현 검증. Part2: 누수 사다리(L0/L1/L2) | ~3분 |
| 8 | `rq2_nested_vs_nonnested.py` | Non-nested vs Repeated Nested (10회 반복), 5개 지표 + 95% CI, 하이퍼파라미터 탐색 포함 | ~20분 |

7·8은 순서 무관, 6 이후 아무 때나 실행 가능. 둘 다 `ymj_faithful_build.py`가 만든
`ymj_faithful_features.csv`가 같은 폴더에 있어야 한다.

## 자체 실험

| 스크립트 | 하는 일 |
|---|---|
| `build_patient_data.py` | 원본 활동+수면 → 환자단위 집계(`patient_level_all_v2.csv`) |
| `run_experiment.py` | V26/V29 파이프라인 재현, `nia+219` 포함/제외 비교 |
| `run_nested_cv_leakfree.py` | CN vs MCI+Dem, 누수 없는 nested CV |
| `compute_circadian_features.py` | 1분 MET 시퀀스 → 서캐디안 지표(IS/IV/RA/M10/L5) |
| `run_dementia_vs_rest_nested.py` | CN+MCI vs Dementia, 서캐디안 피처 nested CV |
| `permutation_test.py` | 위 결과의 permutation test |
| `verify_own_final_model.py` | `EXPERIMENT_LOG.md` 최종 모델(AUC 0.9087)의 permutation 검증 |

## 핵심 구현 포인트

**nested vs non-nested의 차이는 단 한 곳이다.** `ymj_faithful_experiments.py`의
`evaluate()`를 보면:

```python
if not nested:
    # 전체 데이터로 impute/scale/RFE를 먼저 확정 → 라벨 누수
    Xs = StandardScaler().fit_transform(SimpleImputer().fit_transform(Xn))
    selector.fit(Xs, y)          # <-- 여기서 전체 라벨을 봄
    Xf = selector.transform(Xs)

for tr, te in skf.split(Xn, y):
    if nested:
        # outer-train 안에서만 impute/scale/RFE 수행
        ...
        selector.fit(Xtr, y[tr])  # <-- train 라벨만 봄
```

**주의 — 기본 인자 함정**: `ymj_faithful_build.summarize()`의 윈도우 파라미터는
반드시 명시적으로 전달해야 한다. `def f(x, w=W)` 형태로 모듈 전역을 기본값에 묶으면
정의 시점 값이 고정되어, 나중에 `B.W = 5`로 바꿔도 반영되지 않는다.
(이 버그로 첫 감도분석이 조용히 무력화되었고, 윈도우 5/7/14/21 결과가 전부 동일하게
나오는 것을 보고 발견했다.)
