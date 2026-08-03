# reproduction_lee_lee_2025_vae

이민지·이석훈(2025), 「VAE 기반 데이터 불균형 개선을 통한 치매 조기 탐지 기법」
(*Journal of KIIT* 23(7), pp.1-12, DOI [10.14801/jkiit.2025.23.7.1](https://doi.org/10.14801/jkiit.2025.23.7.1))
재현 + 누수 통제 검증 + Nested Group CV.

**목표는 최고 성능이 아니라 재현성과 검증설계에 따른 성능 변화의 정량화다.**

---

## 상태

| 항목 | 상태 |
| --- | --- |
| 논문 분석·데이터 대조 | ✅ 완료 |
| 사전 문서 6종 | ✅ 완료 |
| 기존 코드·config 검사 | 정적 검사 완료; 이번 교정 후 테스트 미실행 |
| **2026-08-03 실제 산출물 감사** | ❌ **재현 실패** |
| 감사 후 코드 교정 | 🔧 정적 수정 — 재실행 필요 |
| 교정본 학습·평가·테스트 | ⏳ **미실행** — 이번 감사에서는 실행하지 않음 |

---

## 2026-08-03 산출물 감사 결론

`reproduction_lee_lee_2025_vae_result/20260803_013228_full/`은 논문의 성능을 재현하지 못했다.
차이는 난수 변동으로 설명할 수 있는 범위를 크게 넘고, 합성자료 붕괴와 실험 배선 결함도
함께 확인되었다.

| 감사 대상 | 기존 산출물 | 논문 보고값 또는 판정 |
| --- | ---: | ---: |
| A, Wide & Deep + VAE, 기록 단위 macro-F1 | **0.4524** | **0.8556** |
| A, Wide & Deep + VAE, Dem recall | **0.1633** | **0.8235** |
| B, VAE 합성자료의 평균 표준편차 비율(합성/실제) | **약 0.0856** | 심각한 분산 붕괴 |
| B, TSTR Dem recall | **모든 모델 0** | 합성자료 일반화 실패 |
| C, 선택된 SMOTE | 2개 fold에서 `n_synthetic=0` | 이름만 SMOTE인 no-op이므로 결과 무효 |
| A 대조표 `n_dem_subjects` | **12로 고정 표기** | 실제 A3 test Dem 피험자 **8명** |

따라서 이 디렉터리의 기존 산출물은 **실패 원인을 보여 주는 감사 증거**로만 보존한다.
논문 재현 성능 또는 교정된 파이프라인의 성능 증거로 인용하면 안 된다. 교정본은 다음을
전제로 다시 실행해야 한다.

- VAE 입력은 표준화 공간으로 통일하고, 재구성 손실과 KL 모두 차원 평균으로 정규화한다.
- VAE는 해당 fold의 실제 train Dem에만 적합하며, fit 대상은 감사 로그에 반드시 기록한다.
- 전처리/VAE fit 범위는 피험자 집합뿐 아니라 원시 row ID로도 검사해, 행 단위 split에서
  all-data fit이 `n_violations=0`으로 가려지지 않게 한다.
- B는 A와 같은 Isolation Forest를 사용하고, B·C는 `subject_stratified` 분할을 사용한다.
- C 탐색 후보는 활성화된 증강법에 유효한 축만 남겨 정규화·중복 제거하고, 증강량 0인
  증강 후보는 유효한 증강으로 취급하지 않는다. 제한된 예산은 분류기×증강법 조합마다
  round-robin 배분한다.

이번 수정에서는 사용자의 지시에 따라 학습·평가·테스트를 실행하지 않았다. 따라서 위 교정이
성능을 회복시켰다고 아직 주장할 수 없으며, **교정본 재실행 후 새 산출물을 별도로 감사해야 한다.**

---

## 먼저 읽을 것

| 문서 | 내용 |
| --- | --- |
| [reproduction_spec.md](reproduction_spec.md) | 재현 사양, 실험 A/B/C 정의, 재현 수준 선언 |
| [report_inconsistencies.md](report_inconsistencies.md) | **논문 내부 불일치 18건**. 가장 중요한 문서 |
| [paper_data_mapping.md](paper_data_mapping.md) | 논문 변수 ↔ 실제 컬럼 대응 (46개 전부 일치) |
| [assumptions.md](assumptions.md) | 미보고 항목의 가정값과 근거 |
| [unresolved_questions.md](unresolved_questions.md) | 미해결 질문 15건 (저자 문의용) |
| [synthetic_data_risk.md](synthetic_data_risk.md) | 합성자료 해석 위험과 진단 절차 |
| [leakage_audit.md](leakage_audit.md) | 누수 통제 설계와 자동검사 대조표 |

---

## 주요 발견 요약

### 1. 데이터 코호트는 완전히 재현된다 ✅

`Data/1.Training` + `Data/2.Validation` 합본이 논문 표 3과 **정확히 일치**한다.

| | CN | MCI | Dem | 합 |
| --- | ---: | ---: | ---: | ---: |
| 피험자 (논문=실측) | 111 | 51 | 12 | **174** |
| 기록 (논문=실측) | 7,737 | 3,661 | 785 | **12,183** |

논문 표 1·2의 **46개 변수가 실제 컬럼명과 문자 단위로 100% 일치**한다. 대체·추정이 필요 없다.

### 2. 이상치 행 수는 Isolation Forest를 강하게 지지한다 🔴

논문은 §4.2·그림 1에서 Isolation Forest, §5.1에서 "상·하위 10%"라고 서로 다르게 서술한다.
행 수와 잔존율은 **Isolation Forest(`contamination=0.1`)를 강하게 지지**하지만,
논문이 seed와 세부 설정을 보고하지 않아 원 구현으로 확정할 수는 없다.

| 방식 | 잔존 행 | CN | MCI | Dem | 논문과의 L1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| percentile q=0.10 (**§5.1 본문**) | 372 (3.05%) | 240 | 126 | **6** | 10,592 |
| **IsolationForest c=0.1 (§4.2)** | **10,964 (89.99%)** | **7,075** | 3,401 | 488 | 54 |
| 논문 §5.1 보고값 | 10,964 | 7,075 | 3,374 | 515 | — |

- IF의 **합계는 seed와 무관하게 항상 정확히 10,964** = 논문 값.
- seed에 따라 L1 거리 8까지 감소 (잔여 차이는 미보고 `random_state` 탓).
- **§5.1 본문 방식은 Dem이 6행만 남아 8:1:1 분할 자체가 불가능하다.**
  `configs/paper_percentile_latent500.yaml`은 이를 설명하는 `InfeasibleSplitError`를 던진다.

### 3. 합성 Dem 행 수는 정확히 4,000개다 (표 5에서 유도)

```
Dem train 4,412 = 실제 412 (= 515 − 51 − 52) + 합성 4,000
```
→ Dem train이 **9.7배**로 부풀려졌다. 출처는 여전히 **12명**이다.

### 4. 논문 표 6의 증강 전/후는 서로 다른 평가셋에서 측정되었다 🔴

precision·recall에서 역산한 결과(혼동행렬 주변합까지 닫힘):

| | 평가셋 (CN, MCI, Dem) | N | 정확도 |
| --- | --- | ---: | ---: |
| 증강 전 | (708, 338, 51) | **1,097** | 0.8888 |
| 증강 후 | (707, 337, 51) | **1,095** | 0.8594 |

→ "동일 조건 비교"라는 전제가 성립하지 않는다. 본 재현은 **동일 split·동일 seed**로 비교한다.

### 5. 본문 §5.2는 Dem F1과 macro F1을 뒤바꿔 적었다

본문 "Dem 0.8556 / 평균 0.875" ↔ 표 6 "Dem 0.8750 / 평균 0.8556".
표 6이 `F1 = 2PR/(P+R)`과 macro 평균 모두에서 자기정합하므로 **표 6이 정본**이다.

### 6. 분할은 행 단위다 → train·test 피험자가 대규모로 중복 등장

논문 표 5의 행 수는 행 단위 8:1:1 분할과 정합한다. 실제 A3 산출물에서는 test 피험자
167명 전원이 train에도 있었고, test에 포함된 Dem 피험자는 8명이었다. 논문의 정확한
subject 교집합은 split seed가 미보고라 확정할 수 없지만, 반복측정 행 누수는 구조적으로
발생한다. 이것이 논문 성능의 주된 낙관 편향 후보이며 실험 B·C가 통제하는 대상이다.

### 7. `sleep_temperature_delta` == `sleep_temperature_deviation`

12,183행 전체에서 원소 단위로 동일하다. 논문의 "46개 변수"는 **실질 45개**다.

---

## 설치

```bash
pip install -r requirements.txt
```

무거운 의존성(torch, xgboost, pytorch-tabnet, imbalanced-learn)은 **지연 임포트**된다.
`--inspect-data`, `--dry-run`, `pytest`는 numpy / pandas / scikit-learn / pyyaml만으로 동작한다.

---

## 실행

### 1) 데이터 점검 (학습 없음)

```bash
python run.py --inspect-data
```

### 2) dry-run — 학습 전에 반드시 확인

```bash
python run.py --config configs/paper_isoforest_scaled_latent500.yaml --dry-run
python run.py --config configs/leakage_controlled_non_nested.yaml --dry-run
python run.py --config configs/nested_subject_independent.yaml --dry-run
```

변수 존재 여부 / 피험자 수 / 클래스별 피험자 수 / 분할 가능 여부 / fold별 Dem 피험자 수 /
preprocessing fit 범위 / VAE 학습 대상 범위 / 예상 합성행 수 / 누수 검사를 출력한다.

### 3) 이상치 방식 검증

```bash
python run.py --config configs/paper_isoforest_scaled_latent500.yaml --audit-only
```

### 4) 실제 실행

```bash
python run.py --config configs/paper_isoforest_scaled_latent500.yaml
python run.py --config configs/leakage_controlled_non_nested.yaml
python run.py --config configs/nested_subject_independent.yaml
```

### 옵션

| 옵션 | 설명 |
| --- | --- |
| `--inspect-data` | 데이터 구조·계약만 점검 |
| `--audit-only` | 누수 검사 + 이상치 방식 재현 검증 |
| `--dry-run` | 학습 없이 절차·규모·누수 검사 |
| `--fold N` | 특정 fold만 실행 |
| `--seed N` | 난수 seed |
| `--resume` | 완료된 산출물 건너뛰기 |
| `--skip-vae` | VAE 조건 제외 |
| `--augmentation {none,vae,class_weight,random_oversampling,smote}` | 증강 조건 (반복 지정 가능) |
| `--models a,b` | 실행할 모델 지정 |

---

## config

| config | 이상치 | latent | 역할 |
| --- | --- | ---: | --- |
| `paper_percentile_latent500.yaml` | percentile 10% | 500 | **primary reported-method reconstruction** (§5.1 본문). ⚠️ 실행 불가 — 발견 #2 |
| `paper_percentile_latent50.yaml` | percentile 10% | 50 | 그림 2 latent 해석 |
| `paper_isoforest_latent500.yaml` | IsolationForest(0.1) | 500 | raw VAE forensic 변형·기존 실패 산출물 |
| `paper_isoforest_latent50.yaml` | IsolationForest(0.1) | 50 | 그림 기준 일관 변형 |
| `paper_isoforest_scaled_latent500.yaml` | IsolationForest(0.1) | 500 | **감사 후 교정한 기본 A5** |
| `leakage_controlled_non_nested.yaml` | (고정) | 500 | 실험 B |
| `nested_subject_independent.yaml` | (inner 선택) | (inner 선택) | 실험 C |

> `paper_percentile_latent500.yaml`은 사용자 지시("본문의 구체적 실험 설명을 우선")에 따라
> primary로 유지한다. 이 config가 실행되지 않는다는 사실 자체가 결과다.
> **어느 config도 "원 저자 코드의 확정 사양"이 아니다.**

---

## 실험 요약

| | 실험 A | 실험 B | 실험 C |
| --- | --- | --- | --- |
| 이름 | `paper_reported_reconstruction` | `leakage_controlled_non_nested` | `nested_subject_independent` |
| 분할 | 행 단위 8:1:1 | 피험자 `subject_stratified` 3-fold | `subject_stratified` outer 3 × inner 3 |
| 이상치 | Isolation Forest(0.1) 기준 | **A와 같은** Isolation Forest(0.1) | inner 후보에서 선택 |
| 전처리 fit | 전체 데이터 (누수) | train fold만 | train fold만 |
| VAE 입력 | **scaled** | **scaled** | **scaled** |
| VAE fit | train Dem + fit 감사 기록 | train fold Dem + 강제 감사 | train fold Dem + 강제 감사 |
| 하이퍼파라미터 | 논문 고정 | 논문 고정 | **inner CV에서 선택** |
| 감사 모드 | `observe` (측정) | `enforce` | `enforce` |
| 주 평가단위 | 기록 (논문과 동일) | **피험자** | **피험자** |

---

## Colab Pro+ 실행 (base.ipynb 기준)

### 한 번에 실행 — Cell 2만 고치고 위에서 아래로 실행

```python
USER_FOLDER = "SangHyo"
RUN_FILE    = "Reproduction/reproduction_lee_lee_2025_vae/run.py"
```

> ⚠️ **`Reproduction/` 한 단계를 빠뜨리면 Cell 3에서 `FileNotFoundError`가 난다.**
> `RUN_FILE`은 `USER_FOLDER`(=`SangHyo`) 기준 상대경로다.
> `"reproduction_lee_lee_2025_vae/run.py"` (✗) → `"Reproduction/reproduction_lee_lee_2025_vae/run.py"` (✓)

이대로 Cell 1 → 2 → 3 → 4 → 5를 순서대로 실행하면 **전체 파이프라인이 한 번에 돈다**.

| 단계 | 내용 |
| ---: | --- |
| 1/10 | 데이터 점검 (`--inspect-data`) |
| 2/10 | 이상치 방식 검증 (`--audit-only`) — Isolation Forest vs 상·하위 10% |
| 3/10 | 실험 A primary(A1) 실행 가능성 확인 — **실패가 예상되며 그 자체가 결과다** |
| 4~6/10 | 실험 A·B·C dry-run (fold 구성·누수 검사) |
| 7/10 | 실험 A5 실행 (증거 기반 교정 재구성; 원저자 설정 아님) |
| 8/10 | 실험 B 실행 (누수 통제 non-nested) |
| 9/10 | 실험 C 실행 (Nested Group CV) |
| 10/10 | 교차 실험 비교표 생성 → `outputs/COMPARISON/` |

각 단계는 독립적으로 실패를 흡수한다. 3단계(A1)는 논문 §5.1 본문 방식이 Dem을 6행만 남겨
8:1:1 분할이 불가능하므로 **반드시 실패하며, 그 실패 기록이 본 재현의 핵심 발견 중 하나**다
(`report_inconsistencies.md` I-1 증거 F). 나머지 단계는 그대로 진행된다.
마지막에 단계별 성공/실패 요약과 소요 시간이 출력되고 `outputs/RUN_ALL_SUMMARY.json`에 저장된다.

**동작 원리**: base.ipynb Cell 5는 `runpy.run_path(..., run_name="__main__")`로 실행하는데,
그 경로에서 `sys.argv`는 노트북 인자가 아니라 **Jupyter 커널 자신의 인자**(`-f kernel-xxxx.json`)다.
그대로 argparse에 넘기면 `unrecognized arguments: -f ...`로 즉사하므로,
`run.py`가 이를 감지해 커널 argv를 무시하고 `run_all()`을 돌린다.
Cell 1이 주입한 `PROJECT_ROOT`/`DATA_ROOT`도 자동으로 쓴다.

### 일부 단계만 돌리고 싶을 때

Cell 5 **앞에** 새 셀을 하나 넣어 환경변수로 인자를 지정한다.

```python
import os
os.environ["VAE2025_ARGS"] = "--config configs/leakage_controlled_non_nested.yaml --dry-run"
```

지우면 다시 전체 실행으로 돌아온다.

```python
import os
os.environ.pop("VAE2025_ARGS", None)
```

또는 셀에서 함수를 직접 부른다 (`argv`를 명시하면 커널 argv를 완전히 무시한다).

```python
import sys
sys.path.insert(0, str(PROJECT_ROOT / "SangHyo/Reproduction/reproduction_lee_lee_2025_vae"))
from run import run_pipeline
run_pipeline(namespace=globals(), argv=["--inspect-data"])
```

### 산출물은 Google Drive(MyDrive)에 저장된다

base.ipynb는 저장소를 `/content/Google-Ajou-AICapstone`에 clone하는데
**`/content`는 런타임이 끊기면 통째로 사라진다.** 실험 C만 몇 시간이 걸리므로
결과를 거기 두면 안 된다. 그래서 산출물 기본 위치를 Drive로 잡는다.

```
내 드라이브/
  GoogleAI_Capstone_Results/
    reproduction_lee_lee_2025_vae/
      LATEST.txt                        ← 가장 최근 실행 폴더 이름
      20260803_102655_full/             ← 실행할 때마다 새 폴더 (이전 결과를 덮어쓰지 않는다)
        inspection/
        A_A5_isoforest_scaled_latent500/
        B_B_leakage_controlled/
        C_C_nested/
        COMPARISON/                     ← 교차 실험 비교표
        RUN_ALL_SUMMARY.json            ← 단계별 성공/실패·소요 시간
```

전체 실행(`run_all`)은 **모든 단계가 하나의 타임스탬프 폴더**를 공유한다.
실행 시작 시 저장 위치가 콘솔 맨 위에 출력되고, 끝날 때 다시 안내된다.

경로 결정 우선순위:

| 순위 | 방법 |
| ---: | --- |
| 1 | `--out-root /경로` |
| 2 | `VAE2025_OUT_ROOT` 환경변수 |
| 3 | config `output.root`가 절대경로인 경우 |
| 4 | **Drive가 마운트되어 있으면 MyDrive 안의 위 폴더** (Colab 기본) |
| 5 | 저장소 `outputs/` (로컬 실행) |

다른 곳에 저장하려면 Cell 5 앞에:

```python
import os
os.environ["VAE2025_OUT_ROOT"] = "/content/drive/MyDrive/내가_원하는_폴더"
```

Drive가 마운트되지 않았거나 쓰기가 안 되면 경고를 출력하고 저장소 `outputs/`로 폴백한다
(이 경우 런타임 종료 시 사라지므로 경고를 반드시 확인할 것).

### 의존성

`run.py`가 스스로 설치한다(`requirements_colab.txt`). torch는 Colab이 CUDA와 맞물려
미리 설치해 둔 것을 건드리지 않고, 없을 때만 설치한다.
base.ipynb Cell 4(개인 폴더 `requirements.txt` 자동 설치)는 `SangHyo/requirements.txt`만
보고 이 폴더처럼 중첩된 경로는 찾지 못하므로 이 프로젝트에는 적용되지 않는다 — 그래서
설치를 스크립트가 책임지도록 만들었다.

### 순수 셸에서 직접 실행 (base.ipynb를 아예 안 쓸 경우)

```bash
cd /content/drive/MyDrive/Google-AJOU-AI-Capstone/SangHyo/Reproduction/reproduction_lee_lee_2025_vae
pip install -r requirements_colab.txt
python run.py --config configs/leakage_controlled_non_nested.yaml --dry-run
python run.py --config configs/leakage_controlled_non_nested.yaml
```

`!python run.py ...`는 서브프로세스로 실행되므로 `sys.argv` 문제가 없다.
데이터 경로가 다르면 `--data-root /content/drive/.../Data` 또는
`SANGHYO_DATA_ROOT` 환경변수로 지정한다.

### GPU / 런타임

**런타임 유형 > T4 GPU (표준)면 충분하다. A100/L4는 이 프로젝트 규모에 낭비다.**

| 근거 | 값 |
| --- | --- |
| 실험 A 최대 학습 행 수 | 9,746행 × 46 feature |
| 실험 B·C fold당 학습 행 수 | ≤ 8,125행 |
| 최대 은닉층 크기 | 512 (VAE encoder 첫 층) |

- `XGBoost`는 `tree_method="hist"`로 **CPU에서 돈다** — GPU 이득이 없다.
- 기본 실행 목록(`run.models`)의 `DNN`·`TabNet`·`Wide & Deep`과 `augmentation.method: vae`는
  torch/pytorch-tabnet 기반이라 GPU가 있으면 유의미하게 빨라진다. 이 셋은 기본값에 포함되므로
  GPU 없이도 동작은 하지만(코드가 `torch.cuda.is_available()`로 자동 폴백) 체감 속도 차이가 크다.
- 고용량 RAM 옵션은 불필요하다. 전체 데이터가 12,183행 × 46 float64 ≈ 4MB다.

런타임이 GPU로 잡혔는지 먼저 확인:

```python
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

**계산량 주의**: 실험 C는 outer 3 × 후보 24 × inner 3 + outer 3 = **219회 모델 적합**이다.
T4에서도 순차 실행 시 시간이 걸리므로 `search.max_evals`를 줄이거나 `--fold N`으로
outer fold를 나눠 여러 세션(Pro+ 세션 한도 24시간)에 걸쳐 실행하는 것을 권장한다.

### 권장 순서

1. `--inspect-data` — 데이터가 논문 표 3과 일치하는지
2. `--dry-run` 3종 — fold 구성과 예상 합성행 수
3. `--audit-only` — 이상치 방식 검증
4. `configs/paper_isoforest_scaled_latent500.yaml` (교정 기본 실험 A5)
5. `configs/leakage_controlled_non_nested.yaml` (실험 B)
6. `configs/nested_subject_independent.yaml` (실험 C, 가장 오래 걸림)

---

## 결과 해석 시 반드시 지킬 것

> 본 실험의 Dem 클래스는 **독립 피험자 12명**에서 유래한다.
> 합성 Dem 행 N개는 해당 fold의 실제 train Dem 피험자 기록 분포에서 생성된 것이며
> 새로운 피험자를 의미하지 않는다.
> 피험자 단위 metric의 분모는 항상 실제 피험자 수다.

fold별 train/eval Dem 피험자 수는 산출물의 fold 구성과 `n_dem_subjects` 열에서 확인한다.
문서나 비교 코드에 특정 수(예: “각 fold 8명”)를 고정해 쓰지 않는다.
자세한 금지 표현과 진단 절차는 [synthetic_data_risk.md](synthetic_data_risk.md) 참조.

---

## 재현 수준

**method-level reconstruction**이며 **exact reproduction은 원리적으로 불가능**하다.
근거 10가지는 [reproduction_spec.md](reproduction_spec.md) §9 참조 (요약: 이상치 기법 충돌,
latent 500/50 충돌, 증강 전후 평가셋 불일치, seed·epoch·batch size 전면 미보고,
VAE·scaler fit 범위 미보고, 임계값 선택 자료 미보고, Wide 입력 미보고, KL 가중치 미보고).

---

## 디렉터리

```
run.py                     단일 실행 진입점
configs/                   base + A1~A4 + B + C
src/
  data/       loader·schema·paper_reference·inspect
  preprocessing/  outliers·pipeline (fold 범위 fit)
  augmentation/   vae·generators·provenance
  models/     base·classifiers·registry
  splits/     row_level(실험 A)·group_cv(실험 B·C)
  evaluation/ metrics·aggregate·bootstrap·tables
  diagnostics/ synthetic_quality·projection·tstr
  audit/      leakage·checks
  experiments/ paper_reconstruction·leakage_controlled·nested_cv
  utils/      config·seeding·io
tests/                     누수 통제 unit test 6종
outputs/                   실행 산출물 (git 미추적)
```
