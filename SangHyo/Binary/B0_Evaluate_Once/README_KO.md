# B0_Evaluate_Once — 봉인 1회 평가 + ledger (Phase 3 · NOW-3)

**단일 노트북**: [`B0_Evaluate_Once.ipynb`](B0_Evaluate_Once.ipynb) (6셀) · **사전등록**: [`PREREGISTRATION_KO.md`](PREREGISTRATION_KO.md) · **ledger**: `EVALUATION_LEDGER.json` (비어 있음; 실 평가는 정확히 1건만 허용)

## 실행 (steward 권장)
```bash
cd SangHyo/Binary/B0_Evaluate_Once
B0EV_SCORES=<scores_<sha8>.csv> B0EV_LABELS=<steward 라벨 파일> \
B0EV_SCORING_RECORD=<SCORING_RECORD.json> B0EV_QC_REPORT=<QC_REPORT.json> \
B0EV_NB_PATH=$PWD/B0_Evaluate_Once.ipynb OMP_NUM_THREADS=1 \
  ../../../.venv/bin/jupyter nbconvert --to notebook --execute --output /tmp/evaluation_run.ipynb B0_Evaluate_Once.ipynb
```
라벨 파일 컬럼: `sid, DIAG_NM` 필수(`CN/MCI/Dem`); `age, sex, edu_years, edu_cat, gait_speed, site, enroll_month, mci_subtype` 선택(secondary 용).
결과: `B0_Evaluate_Once_result/<UTC_RUN_ID>/{REPORT.json, REPORT_KO.md, report.png}` + ledger 갱신.

## 합성 배선 테스트 (결과 무효)
`B0EV_SYNTHETIC=1 B0EV_LEDGER=/tmp/LEDGER_SYN.json B0EV_N_BOOT=1000` 로 실행. 실 ledger 는 건드리지 않는다.

## 규칙 요약
primary = AUC_std, tier R/T/F/I 는 AUC_std CI 로 기계 판정, ledger 가 2번째 실 평가를 거부, 결과 후 채점기 수정 금지. 자세한 것은 사전등록 §2–§5.
