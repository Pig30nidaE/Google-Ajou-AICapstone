# Binary_NewCohort_Augmented_Nested — Stage B 증강 모델 (Phase 3 · NOW-6)

**단일 노트북**: [`Binary_NewCohort_Augmented_Nested.ipynb`](Binary_NewCohort_Augmented_Nested.ipynb) (7셀, run-4 엔진 인라인) · **사전등록**: [`PREREGISTRATION_KO.md`](PREREGISTRATION_KO.md)

## 실행 (Stage A 의 ledger 가 실 평가 1건을 보인 뒤에만)
```bash
cd SangHyo/Binary/Binary_NewCohort_Augmented_Nested
B0AU_SCORES=<scores_<sha8>.csv> B0AU_COVARIATES=<steward 공변량/라벨 파일> \
B0AU_NB_PATH=$PWD/Binary_NewCohort_Augmented_Nested.ipynb OMP_NUM_THREADS=1 \
  ../../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 Binary_NewCohort_Augmented_Nested.ipynb
```
공변량 파일 컬럼: `sid, DIAG_NM, age, sex, edu_years` 필수; `gait_speed, site, enroll_month` 선택. 라벨(DIAG_NM)은 outer 층화와 평가에만 쓰이고, 인지검사 열은 가드가 차단한다.
배선 테스트: `B0AU_SYNTHETIC=1 B0AU_QUICK=1` + `B0_Score_NewCohort/make_synthetic_cohort.py` 산출물(`labels_SEALED.csv`).

예상 시간(n≈310, 1 스레드): main K=2~3 ~1분, 진단 ~1분, permutation 2000회 ~15~25분.
결과: `Binary_NewCohort_Augmented_Nested_result/<UTC_RUN_ID>/{FINAL_REPORT.json, oof_predictions.csv, null_delta1.npy, null_max.npy}`.
