# Binary_SedentaryBout_Nested — Experiment 2 (confirmatory-record)

**단일 노트북**: [`Binary_SedentaryBout_Nested.ipynb`](Binary_SedentaryBout_Nested.ipynb) (14셀, 코드 전부 인라인, 외부 모듈 import 없음)
**사전등록**: [`PREREGISTRATION_KO.md`](PREREGISTRATION_KO.md) (실행 전 동결, 코드셀 sha256 포함)

CN(111) vs MCI+Dem(63), 174명. **nested CV + 피험자 단위 분할 + MMSE 제외.**
질문: 깨어있는 창 안의 **최장 정좌 bout(분)의 일간 SD** (`longest_w_SD`) 가 동결 D블록(`low_SD`, `low_MED`, subject-mean AUC 0.6491) 위에 증분을 주는가?
이것은 `low_SD` 가설("인지저하에서 하루하루가 더 같아진다")을 분 총량에서 **bout 구조**로 확장한, 사전 선언된 **마지막** 기전 가설이다.
T3 이면 stop rule 이 발동해 이 코호트에서의 특징 탐색을 종료한다 (사용자 결정, 2026-09-04).

---

## 실행 방법

### 로컬 (권장 — B0 가 run 2 / Experiment 1 과 비트 동일하게 재현됨을 노트북이 assert)
```bash
cd SangHyo/Binary/Binary_SedentaryBout_Nested
EXP2_NB_PATH=$PWD/Binary_SedentaryBout_Nested.ipynb OMP_NUM_THREADS=1 \
  ../../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 Binary_SedentaryBout_Nested.ipynb
```
`.venv` 에 `nbclient`/`nbconvert` 가 필요하다. 데이터는 저장소 `Data/` 를 자동 탐색한다.
결과는 이 폴더의 `Binary_SedentaryBout_Nested_result/<UTC_RUN_ID>/`. 진행 상황은 그 안의 `PROGRESS.json`.

### Colab
1. CPU 런타임으로 충분하다. 2 vCPU 면 permutation 에 ~60분, 전체 ~80분.
2. 셀을 위에서부터 전부 순서대로 실행. 셀 1 에서 Drive 마운트를 요청하고 `Data/` 를 자동 탐색한다.
3. 결과는 `/content/drive/MyDrive/Binary_SedentaryBout_Nested_result/<UTC_RUN_ID>/`.

### 배선 테스트 (결과 무효)
`EXP2_QUICK=1`(repeat 2, permutation 20) 과 `EXP2_SYNTHETIC=1`(B0 재현 assert 건너뜀), `EXP2_DATA_ROOT=<합성 데이터>`,
`EXP2_OUT_ROOT=<임시 폴더>` 로 전 셀을 약 1분에 통과시킬 수 있다. 이 모드의 숫자는 어디에도 보고하지 않는다 (RUN_ID 에 `_QUICK`/`_SYN` 접미사).

### 예상 시간 (로컬, 1 스레드)
| 구간 | 시간 |
|---|---|
| 로드·전체일 MET 파싱·bout 지표·특징·신뢰도 | ~30 s |
| MAIN K=2 (20×5, inner 4×2, C 4개) | ~16 s |
| 진단 6 arm + 기준 2 arm (20 repeats) | ~2.5 min |
| permutation 2000회 (순차) | ~14 min |
| 후보 열 permutation 1000회 | ~4 min |
| 결정성 재실행·저장 | ~30 s |
| **합계** | **≈ 22~25 min** |

---

## 후보 정의 (실행 전 동결)

```
worn      = MET ≥ 0.5                       (Oura 비착용 0.1 / 착용 바닥 0.9; 0.0 no-data sentinel 도 잡힌다)
sedentary = worn & MET < 1.5                (비착용 분은 False → run 이 끊긴다)
onset     = 착용분만의 30분 rolling MET 평균 ≥ 1.5 가 처음 성립하는 창의 시작 (04:00 기준 분)
offset    = 그 조건이 마지막으로 성립하는 창의 끝(배타)
day valid = 착용률 ≥ 0.8 and (offset − onset) ≥ 240분
longest_w = [onset, offset) 안 sedentary 연속 run 길이의 최댓값 (분)
longest_w_SD = 35일 창의 valid 일에 대한 across-day SD (ddof 0)
```
`activity_class_5min` 은 쓰지 않는다. `low_SD`/`low_MED` 는 run 2 와 동일하게 35일 전부(게이트 없음)로 계산해 비트 동일성을 지킨다.

## Experiment 1 과 무엇이 다른가

| 항목 | Experiment 1 | 이 실험 |
|---|---|---|
| 후보 | `cpl_wake2onset_slope` (run-2 사후 단변량 위) | `longest_w_SD` (**label-free 자격만**: SB 0.75, ρ(low_SD) 0.22, ρ(low_MED) 0.17) |
| 노출 상태 | 감사 agent 의 nested 결과 노출 → confirmatory-record | 구성개념 인접 변형이 Training-141 스크린 표에 존재 → confirmatory-record (AUC 열은 읽지 않음) |
| MET 파싱 | 첫 35일만 | 전체 조인일(S1 B0_all 용) + 첫 35일 |
| 잔차화 | 앞 n_feat 열 전부 (D블록 포함) | **후보 열만** (`make_resid_col_arm`) → 기준은 B0ref 그대로 |
| 진단 | F1~F5 (wrap 기전) | F1 단독 · F2 ⟂low_MED · F3 가족 2순위(`meanbout_w_SD`) · F4 ⟂nonwear · F5 ⟂유효일수 · S1 B0_all(기술용) |
| 교란 감사 | 수면 타이밍 축 | 창 구조 축 (창 길이·onset·유효일수·최장 bout 수준) |
| 저장 | 20개 | + `daily_metrics_first35.csv` (일별 bout 지표, label-free) |

## 정직성 장치

| 장치 | 셀 |
|---|---|
| MMSE 경로 fail-closed + 컬럼명 가드 자기검증(`_MUST_PASS`/`_MUST_FAIL`) | 2 |
| non-wear = MET < 0.5, 0.0 sentinel run 검출, first-35 창 전원 성립 assert | 3 |
| 깨어있는 창·bout run 이 비착용에서 끊김 (SED 마스크가 worn 을 포함) | 4 |
| 90명 부분집합 재구축 → **모든 컬럼 비트 동일** | 6 |
| 20 repeat 전수: train∩test = ∅, test Dem ≥ 2, MCI ≥ 8 | 7 |
| B0 재현 assert (run-2 subject-mean·per-repeat, 1e-9) · 분해 항등식 · 실패 fold 0 | 9 |
| permutation: 라벨 셔플 Δ null (p_Δ) + 후보 열 셔플 (p_col), null 중심 0.5 확인 | 11 |
| 사전등록 T1/T2/T3 기계 판정 + 기전·가족 규칙 (a)/(b)/(c1)/(c2) 기계 판정 | 12 |
| 두 main arm 전체 재실행 해시 일치, 코드셀 sha256 기록 | 13 |

## 주의
- 저장소 `.gitignore` 가 `*.ipynb`/`*.csv` 를 제외한다 → 커밋하려면 `git add -f`.
- 진단·sensitivity arm 의 어떤 숫자도 대표값으로 주장하지 않는다. S1(B0_all) 은 절대 baseline 으로 채택하지 않는다.
- 결과가 나빠도 산출물을 삭제하지 않는다.
