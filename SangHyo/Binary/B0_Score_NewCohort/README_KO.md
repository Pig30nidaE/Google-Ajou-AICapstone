# B0_Score_NewCohort — 봉인 채점 노트북 (Phase 3 · NOW-2)

**단일 노트북**: [`B0_Score_NewCohort.ipynb`](B0_Score_NewCohort.ipynb) (7셀, 인라인, sklearn 불필요) · **사전등록**: [`PREREGISTRATION_KO.md`](PREREGISTRATION_KO.md)
**도구**: `check_label_blind.py`(CT6) · `make_manifest_from_sourcedata.py`(SourceData 만으로 manifest 생성) · `make_synthetic_cohort.py`(배선 테스트용 합성 코호트) · `manifest_aihub174.csv`

## 실행
```bash
cd SangHyo/Binary/B0_Score_NewCohort
B0SC_DATA_ROOT=<새 코호트 루트> B0SC_MANIFEST=<steward manifest.csv> \
B0SC_NB_PATH=$PWD/B0_Score_NewCohort.ipynb OMP_NUM_THREADS=1 \
  ../../../.venv/bin/jupyter nbconvert --to notebook --execute --output /tmp/scoring_run.ipynb B0_Score_NewCohort.ipynb
```
환경변수: `B0SC_DATA_ROOT`(필수) · `B0SC_MANIFEST`(기본 `<root>/manifest.csv`) · `B0SC_ARTIFACT`(기본 `../B0_Frozen_Scorer/b0_frozen_v1.json`) · `B0SC_OUT_ROOT` · `B0SC_CT5_REFERENCE`(174명 참조 점수와 비교, 검증 전용).
결과: `B0_Score_NewCohort_result/<UTC_RUN_ID>/{scores_<sha8>.csv, QC_REPORT.json, SCORING_RECORD.json}`.

## 검증 재현
```bash
# CT6
../../../.venv/bin/python check_label_blind.py B0_Score_NewCohort.ipynb
# CT5 (AI-Hub 174명 재현)
B0SC_DATA_ROOT=../../../Data B0SC_MANIFEST=$PWD/manifest_aihub174.csv \
B0SC_CT5_REFERENCE=../B0_Frozen_Scorer/B0_Frozen_Scorer_result/<run>/b0_scores_174_insample.csv ... (위 명령)
# CT7 (합성 dirty)
../../../.venv/bin/python make_synthetic_cohort.py --out /tmp/synth_dirty --dirty && B0SC_DATA_ROOT=/tmp/synth_dirty ... (위 명령)
```

## 거부 코드
`no_data`(manifest 에만 있음) · `schema_fail`(행 스키마 위반) · `insufficient_days`(<35 조인일) · `feature_unavailable`(창 안 activity_low 결측) · `scored`.

## 주의
- 이 노트북은 라벨 파일을 열지 않으며, 열 수 있는 경로 자체를 차단한다. 공변량 컬럼이 같은 파일에 있어도 읽지 않는다(존재만 기록).
- drift(PSI/KS) 는 보고 전용이다. 점수는 어떤 QC 결과로도 바뀌지 않는다.
- 다른 기기/스키마의 export 는 SCHEMA_CONTRACT 위반으로 거부된다. MET-대체 재구현 채점기는 만들지 않는다(사용자 결정).
