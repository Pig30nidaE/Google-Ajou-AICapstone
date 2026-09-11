# Binary_WakeOnsetCoupling_Nested — Experiment 1

**단일 노트북**: [`Binary_WakeOnsetCoupling_Nested.ipynb`](Binary_WakeOnsetCoupling_Nested.ipynb) (14셀, 코드 전부 인라인, 외부 모듈 import 없음)
**사전등록**: [`PREREGISTRATION_KO.md`](PREREGISTRATION_KO.md) (실행 전 동결, 코드셀 sha256 포함)

CN(111) vs MCI+Dem(63), 174명. **nested CV + 피험자 단위 분할 + MMSE 제외.**
질문: run 2 의 `cpl_wake2onset_slope` 가 동결 D블록(`low_SD`, `low_MED`, subject-mean AUC 0.6491) 위에 증분을 주는가?

---

## 실행 방법

### 로컬 (권장 — B0 가 Colab 과 비트 동일하게 재현됨을 확인)
```bash
cd SangHyo/Binary/Binary_WakeOnsetCoupling_Nested
EXP1_NB_PATH=$PWD/Binary_WakeOnsetCoupling_Nested.ipynb OMP_NUM_THREADS=1 \
  ../../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 Binary_WakeOnsetCoupling_Nested.ipynb
```
`.venv` 에 `nbclient`/`nbconvert` 가 필요하다(`pip install nbclient nbconvert`). 데이터는 저장소 `Data/` 를 자동 탐색한다.
결과는 이 폴더의 `Binary_WakeOnsetCoupling_Nested_result/<UTC_RUN_ID>/`. 진행 상황은 그 안의 `PROGRESS.json`.

### Colab
1. 런타임은 CPU 로 충분하다 (GPU 불필요). 2 vCPU 면 permutation 에 ~60분, 전체 ~75분.
2. 셀을 위에서부터 전부 순서대로 실행. 셀 1 에서 Drive 마운트를 요청하고 `Data/` 를 자동 탐색한다.
3. 결과는 `/content/drive/MyDrive/Binary_WakeOnsetCoupling_Nested_result/<UTC_RUN_ID>/`.

### 배선 테스트 (결과 무효)
환경변수 `EXP1_QUICK=1`(repeat 2, permutation 20) 과 `EXP1_SYNTHETIC=1`(B0 재현 assert 건너뜀), `EXP1_DATA_ROOT=<합성 데이터>` 로
전 셀을 약 40초에 통과시킬 수 있다. 이 모드의 숫자는 어디에도 보고하지 않는다 (RUN_ID 에 `_QUICK`/`_SYN` 접미사가 붙는다).

### 예상 시간 (로컬 M1, 1 스레드 실측 기준)
| 구간 | 시간 |
|---|---|
| 로드·MET 파싱·특징·신뢰도 | ~20 s |
| MAIN K=2 (20×5, inner 4×2, C 4개) | ~16 s |
| 기전 진단 7 arm + 기준 2 arm (20 repeats) | ~2.5 min |
| permutation 2000회 (0.41 s/회, 순차) | ~14 min |
| 후보 열 permutation 1000회 | ~4 min |
| 결정성 재실행·저장 | ~30 s |
| **합계** | **≈ 22 min** |

---

## run 2 와 무엇이 다른가

| 항목 | run 2 (`Binary_DaytimeVariety_Nested`) | 이 실험 |
|---|---|---|
| 후보 | lean-6 (신규 4개 묶음) | **v0 하나** (K=2, P vs B0) |
| B0 | 동결 D블록 | 동일 + **run-2 값과 비트 일치 assert** (per-repeat 20개) |
| 헤드라인 p | p_max (B0 포함 max → 후보가 0이어도 0.001) | **p_Δ** (Δ 의 permutation null) + p_col co-statistic |
| 진단 | 8 repeats, 기준 B0REF 별도 | **20 repeats, 동일 fold**, paired Δ |
| 실패 fold | 진단에서 카운터 폐기 | 모든 호출부에서 수집, main = 0 assert, 진단은 N/A 처리 |
| threshold 지표 | OOF 라벨로 최적화(낙관) | 제거 (threshold-free 만) |
| 금지어 가드 | FEAT 에만 | **모든 arm 의 특징 컬럼명**에 적용, nuisance 는 잔차화 전용 허용 목록 |
| 결정성 | P repeat 0 만 | **두 main arm 전체 OOF 행렬** + null 배열 |
| 교란 감사 | 수집량·배치 5개 | + wake_SD, 기상 중앙시각, 취침, 수면시간, n_earlywake (wrap 기전 감사) |
| TabPFN | 진단 arm (토큰 하드코딩) | 제거. **토큰을 노트북에 넣지 않는다** |
| 결과 폴더 | Colab Drive 또는 cwd | 노트북 폴더 기준 |

## 정직성 장치

| 장치 | 셀 |
|---|---|
| MMSE 경로 fail-closed + 컬럼명 가드 자기검증(`_MUST_PASS`/`_MUST_FAIL`) | 2 |
| non-wear = MET < 0.5, first-35 창 전원 성립 assert | 3 |
| 90명 부분집합 재구축 → **모든 컬럼 비트 동일** | 6 |
| 20 repeat 전수: train∩test = ∅, test Dem ≥ 2, MCI ≥ 8 | 7 |
| B0 재현 assert (run-2 subject-mean·per-repeat, 1e-9) · 분해 항등식 · 실패 fold 0 | 9 |
| permutation: 라벨 셔플 Δ null (p_Δ) + 후보 열 셔플 (p_col), null 중심 0.5 확인 | 11 |
| 사전등록 T1/T2/T3 기계 판정 + 기전 규칙 (a)/(b)/(c) 기계 판정 | 12 |
| 두 main arm 전체 재실행 해시 일치, 코드셀 sha256 기록 | 13 |

## 실행 전 공시 (자세한 내용은 PREREGISTRATION §11)

- v0 는 `mod 1440` wrap 때문에 실질적으로 "04:00 이전 기상 여부" 지표다 (ρ = −0.83). 그래서 F2/F3/F4 로 기전을 분리한다.
- 계획 단계 감사에서 대안 정의의 in-sample AUC 와, 한 agent 가 재실행한 P arm 의 nested 결과(Δ_merged ≈ −0.014)가 실험자에게
  노출됐다. 판정 기준은 노출 전에 확정됐고 바꾸지 않았다. **이 run 은 blind test 가 아니라 사전등록 기준에 따른 확정·기록용 run 이다.**

## 주의
- 저장소 `.gitignore` 가 `*.ipynb`/`*.csv` 를 제외한다 → 커밋하려면 `git add -f`.
- 진단 arm 의 어떤 숫자도 대표값으로 주장하지 않는다. 결과가 나빠도 산출물을 삭제하지 않는다.
