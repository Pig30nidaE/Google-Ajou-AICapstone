#!/usr/bin/env python3
"""단일 실행형 노트북 ``run_reproduction_ABC.ipynb`` 생성기.

Hong 2024의 notebook/ 처럼 **노트북 하나만 Colab에 올리면 도는** 형태를 만든다.
파이프라인 소스(run.py + src/ + configs/ + requirements*)를 tar.gz → base64로
노트북 셀에 내장하므로 저장소 clone이 필요 없다. 손으로 소스를 셀에 옮겨 적으면
감사로 잡은 버그가 재발할 수 있어, 검증된 소스를 바이트 그대로 내장한다.

소스나 셀 구성이 바뀌면 이 스크립트를 다시 실행해 노트북을 재생성한다::

    python make_standalone_notebook.py
"""
from __future__ import annotations

import base64
import io
import json
import subprocess
import tarfile
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "run_reproduction_ABC.ipynb"
#: 노트북 markdown에서 참조하는 문서들의 GitHub 위치 (단일 노트북에는 문서가 없으므로)
DOC = ("https://github.com/Pig30nidaE/Google-Ajou-AICapstone/blob/main/"
       "SangHyo/Reproduction/reproduction_lee_lee_2025_vae")


def build_archive() -> tuple[str, str, list[str]]:
    """파이프라인 소스를 tar.gz로 묶어 base64 문자열로 돌려준다."""
    files = [HERE / "run.py", HERE / "requirements.txt", HERE / "requirements_colab.txt"]
    files += sorted((HERE / "configs").glob("*.yaml"))
    files += sorted(p for p in (HERE / "src").rglob("*.py") if "__pycache__" not in p.parts)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for f in files:
            tf.add(f, arcname=str(f.relative_to(HERE)))
    b64 = base64.b64encode(buf.getvalue()).decode()
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
        capture_output=True, text=True,
    ).stdout.strip() or "unknown"
    return "\n".join(textwrap.wrap(b64, 200)), commit, [str(f.relative_to(HERE)) for f in files]


B64, COMMIT, MANIFEST = build_archive()

cells: list[dict] = []


def md(source: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.replace("@DOC@", DOC)})


def code(source: str) -> None:
    cells.append({
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": source,
    })


# ══════════════════════════════════════════════════════════════════════════════
md("""\
# 이민지·이석훈 (2025) 「VAE 기반 데이터 불균형 개선을 통한 치매 조기 탐지 기법」 재현 — 실험 A·B·C

*Journal of KIIT* 23(7), pp.1–12 · [DOI 10.14801/jkiit.2025.23.7.1](https://doi.org/10.14801/jkiit.2025.23.7.1)

이 노트북은 세 가지 검증 설계를 **같은 데이터·같은 파이프라인·같은 seed**로 실행하고,
마지막 단계에서 결과를 시각화한다. 목표는 최고 성능이 아니라 **재현성과, 검증 설계에 따른
성능 변화의 정량화**다.

| 실험 | 내용 | 분할 | 하이퍼파라미터 | 주 평가 단위 |
| --- | --- | --- | --- | --- |
| **A** | 원본 논문 재현 — 행 단위 8:1:1, 전처리 전체-fit 누수까지 논문 절차 그대로 | 행 단위 | 논문 고정 | 기록 (피험자 집계 병행) |
| **B** | 다른 조건은 A와 동일, **분할만 피험자 단위**로 변경 | `subject_stratified` 3-fold | 논문 고정 | **피험자** |
| **C** | 피험자 단위 분할 위에서 **Nested CV** — 전처리·증강·모델을 inner CV가 선택 | outer 3 × inner 3 | **inner CV 선택** | **피험자** |

## "정확한 재현"의 한계 — 먼저 읽을 것

논문은 seed·epoch·batch size·VAE fit 범위·scaler fit 범위·KL 가중치·Wide 컴포넌트 입력을
보고하지 않았고, 본문 안에서 서로 충돌하는 서술이 있다
(이상치: §4.2 Isolation Forest ↔ §5.1 "상·하위 10%", latent 차원: 본문 500 ↔ 그림 2의 50).
따라서 실험 A는 **method-level 재구성**이며, 미보고 항목은
[assumptions.md](@DOC@/assumptions.md)에 기록된 가정을 쓴다. 특히:

- §5.1 본문의 percentile 해석은 Dem을 **6행**만 남겨 8:1:1 분할이 산술적으로 불가능하다.
  아래 A1 셀은 그 오류를 그대로 보여준다 — **이 실패 자체가 재현 결과다**
  ([report_inconsistencies.md](@DOC@/report_inconsistencies.md) I-1).
- 실제 학습은 행 수가 논문 §5.1 보고값(10,964행)과 정합하는
  **Isolation Forest(contamination=0.1) + 표준화 공간 VAE**(config `A5`)로 진행한다.

참고 문서: [reproduction_spec.md](@DOC@/reproduction_spec.md) ·
[report_inconsistencies.md](@DOC@/report_inconsistencies.md) ·
[assumptions.md](@DOC@/assumptions.md) · [leakage_audit.md](@DOC@/leakage_audit.md) ·
[synthetic_data_risk.md](@DOC@/synthetic_data_risk.md)

## 실행 방법 — 이 노트북 하나면 된다

1. Colab에서 이 노트북을 열고 **런타임 유형을 T4 GPU(표준)** 로 바꾼다.
2. **런타임 → 모두 실행.** 그게 전부다:
   - **데이터**: `내 드라이브/Data/` (= `MyDrive/Data`, 안에 `1.Training`·`2.Validation`)
     — Drive 마운트 후 바로 읽는다.
   - **코드**: 감사된 재현 파이프라인 소스가 **노트북 안에 내장**되어 있다
     (저장소 clone 불필요). 첫 셀이 런타임 디스크에 풀어 import한다.
   - **의존성**: `run.py`가 자동 설치한다.
3. 예상 소요(T4): 실험 A ≈ 5분 · 실험 B ≈ 5–10분 · **실험 C는 수 시간**
   (outer 3 × inner 3 × 후보 24 ≈ 219회 모델 적합 — `--fold N`으로 나눠 실행 가능).
4. 산출물은 `내 드라이브/reproduction_lee_lee_2025_vae/<타임스탬프>_nb/`에 저장되므로
   **런타임이 끊겨도 유지된다.** 셀을 나눠 실행해도 한 커널 세션 안에서는 같은 폴더에
   누적된다.

> 내장 소스의 원본은 저장소 `SangHyo/Reproduction/reproduction_lee_lee_2025_vae`
> (unit test 105개)이며, 저장소 코드가 바뀌면 `python make_standalone_notebook.py`로
> 이 노트북을 재생성한다. 노트북을 저장소 안에서 로컬로 열면 내장본 대신
> 저장소 소스를 그대로 쓴다.
""")

# ══════════════════════════════════════════════════════════════════════════════
blob = (
    "# ── 내장 파이프라인 소스 (자동 생성 — 편집하지 말 것) ─────────────────────────\n"
    "# 원본: SangHyo/Reproduction/reproduction_lee_lee_2025_vae"
    f" (커밋 {COMMIT}, 파일 {len(MANIFEST)}개)\n"
    "# 재생성: 저장소에서 python make_standalone_notebook.py\n"
    f'PIPELINE_COMMIT = "{COMMIT}"\n'
    'PIPELINE_B64 = """\\\n' + B64 + '"""\n'
    'print("내장 파이프라인 소스:", format(len(PIPELINE_B64), ","), "chars | 커밋", PIPELINE_COMMIT)\n'
)
code(blob)

md("""\
## 0단계 — 환경 준비

이 셀 하나가 실행 환경 전체를 만든다.

- **Colab**: Drive 마운트 → `내 드라이브/Data`(= `MyDrive/Data`) 확인 →
  위 셀에 내장된 파이프라인 소스를 런타임 디스크에 풀어 `run.py` import →
  세션 산출물 폴더 생성.
- **로컬**: 저장소 안에서 열었으면 저장소 소스와 `Data/`를 알아서 찾는다.

데이터 폴더 이름이 `Data`가 아니면 아래 `MYDRIVE_DATA_DIR`만 바꾸면 된다.
""")

code("""\
# ── 0단계: 환경 준비 (Colab 단독 실행 / 로컬 겸용) ────────────────────────────
import base64
import hashlib
import io as _io
import os
import shlex
import sys
import tarfile
import textwrap
import time
from pathlib import Path

MYDRIVE_DATA_DIR = "Data"  # 내 드라이브 안의 데이터 폴더명 (mydrive/Data)
REPRO_SUBPATH = Path("SangHyo/Reproduction/reproduction_lee_lee_2025_vae")

try:
    import google.colab  # type: ignore  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


def _unpack_pipeline(target: Path) -> Path:
    \"\"\"위 셀에 내장된 파이프라인 소스(tar.gz)를 target에 푼다.\"\"\"
    raw = base64.b64decode(PIPELINE_B64)
    print(f"내장 파이프라인 전개: {len(raw):,} bytes → {target}")
    print(f"  sha256 {hashlib.sha256(raw).hexdigest()[:16]}… | 원본 커밋 {PIPELINE_COMMIT}")
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=_io.BytesIO(raw), mode="r:gz") as tf:
        try:
            tf.extractall(target, filter="data")
        except TypeError:  # Python < 3.12는 filter 인자가 없다
            tf.extractall(target)
    return target


if IN_COLAB:
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive")

    # ① 데이터: 내 드라이브의 Data/ (1.Training·2.Validation 포함)
    DATA_ROOT = Path("/content/drive/MyDrive") / MYDRIVE_DATA_DIR
    if not (DATA_ROOT / "1.Training").exists():
        raise FileNotFoundError(
            f"{DATA_ROOT} 아래에 1.Training이 없다. 내 드라이브에 "
            f"'{MYDRIVE_DATA_DIR}/1.Training', '{MYDRIVE_DATA_DIR}/2.Validation'이 "
            "있는지 확인하고, 폴더명이 다르면 MYDRIVE_DATA_DIR을 고쳐라."
        )
    os.environ["SANGHYO_DATA_ROOT"] = str(DATA_ROOT)  # run.py가 이 경로를 쓴다

    # ② 코드: 노트북에 내장된 소스를 런타임 디스크에 푼다 (저장소 불필요)
    REPRO_DIR = _unpack_pipeline(Path("/content/repro_lee_lee_2025_vae"))
else:
    # 로컬: 저장소 안에서 열었으면 저장소 소스를 그대로 쓴다 (개발 편의)
    _here = Path.cwd().resolve()
    _root = next((p for p in (_here, *_here.parents)
                  if (p / REPRO_SUBPATH / "run.py").exists()), None)
    if _root is not None:
        REPRO_DIR = _root / REPRO_SUBPATH
        print(f"로컬 저장소 소스 사용: {REPRO_DIR}")
    else:
        REPRO_DIR = _unpack_pipeline(_here / "repro_lee_lee_2025_vae_standalone")

os.chdir(REPRO_DIR)
if str(REPRO_DIR) not in sys.path:
    sys.path.insert(0, str(REPRO_DIR))

import run  # 재현 파이프라인 진입점 (run.py)

if not IN_COLAB:
    # 로컬: 저장소 Data/ 등 기존 후보 경로에서 찾는다 (SANGHYO_DATA_ROOT로 지정 가능)
    DATA_ROOT = run._resolve_data_root(globals(), None)

# 산출물: Colab+Drive면 MyDrive, 아니면 outputs/ 아래에 세션별 타임스탬프 폴더
OUT_BASE = run._resolve_output_root(globals(), None, None)
if "SESSION_DIR" not in globals():
    SESSION_DIR = run._make_session_dir(OUT_BASE, tag="nb")

print("REPRO_DIR    :", REPRO_DIR)
print("DATA_ROOT    :", DATA_ROOT)
print("산출물 폴더  :", SESSION_DIR)
try:
    import torch
    print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
except ImportError:
    print("torch 미설치 — 학습 셀을 처음 실행할 때 run.py가 자동 설치한다")

RESULTS = {}  # 이 커널 세션에서 실행한 실험 결과 dict — 5단계 비교표 생성에 쓴다


def repro(argstr, *, collect=None, expect_failure=False):
    \"\"\"run.py를 이 세션의 산출물 폴더를 향해 실행한다.

    expect_failure=True면 예외를 '예상된 결과'로 출력하고 삼킨다
    (실험 A1의 InfeasibleSplitError가 그 경우다).
    collect="A"|"B"|"C"를 주면 결과를 RESULTS에 모아 비교표 생성에 쓴다.
    \"\"\"
    argv = shlex.split(argstr) + ["--out-root", str(SESSION_DIR)]
    t0 = time.monotonic()
    try:
        result = run.run_pipeline(namespace=globals(), argv=argv)
    except Exception as error:
        elapsed = time.monotonic() - t0
        if not expect_failure:
            raise
        print(f"\\n[예상된 실패 — 이것이 재현 결과다] {type(error).__name__} ({elapsed:.0f}s)")
        print(textwrap.indent(str(error), "    "))
        return None
    elapsed = time.monotonic() - t0
    if expect_failure:
        print("\\n⚠️ 실패가 예상됐는데 성공했다 — report_inconsistencies.md I-1과 대조하라.")
    if collect and isinstance(result, dict):
        RESULTS[collect] = result
    print(f"\\n[완료] {argstr} — {elapsed / 60:.1f}분, 산출물: {SESSION_DIR}")
    return result
""")

# ══════════════════════════════════════════════════════════════════════════════
md("""\
## 1단계 — 데이터 점검 (학습 없음)

코호트가 논문 표 3과 일치하는지 확인한다: **피험자 174명(CN 111 / MCI 51 / Dem 12),
기록 12,183행(7,737 / 3,661 / 785)**. 논문 표 1·2의 46개 변수가 실제 컬럼과 문자 단위로
일치하는지, 그리고 §5.1의 percentile 해석이 논문 행 수를 재현하는지(못 한다 — I-1)도
함께 스캔한다.
""")

code("""\
repro("--inspect-data")
""")

# ══════════════════════════════════════════════════════════════════════════════
md("""\
## 2단계 — 실험 A: 원본 논문 재현

논문 절차를 그대로 따른다. **논문이 보고한 값은 그대로, 미보고 항목은
`assumptions.md`의 가정값**을 쓴다.

| 항목 | 설정 | 출처 |
| --- | --- | --- |
| 변수 | 활동 22 + 수면 24 = 46개 | 논문 표 1·2 (실제 컬럼과 100% 일치) |
| 이상치 | Isolation Forest, contamination 0.1 | §4.2·그림 1 (행 수가 §5.1 보고값과 정합) |
| 분할 | **행 단위** 8:1:1 층화, 피험자 미고려 | §5.1 표 5와 정합 (누수의 원인) |
| VAE | 46 → 512 → 256 → latent 500, BN + dropout 0.3, Adam 1e-4 | §5.1 (latent 50은 그림 2 변형으로 별도 제공) |
| 합성 Dem | **4,000행** (train Dem 412 → 4,412) | 표 5에서 산술 유도 |
| 분류기 | XGBoost(softmax·depth 6·lr 0.1) / DNN(512-256-128-64-32, dropout 0.5) / TabNet(n_d=n_a=64, 5 steps) / Wide & Deep(deep 256-128-64, dropout 0.3) | §5.1 |
| 전처리 fit | **전체 데이터** (논문 절차의 누수를 의도적으로 재현) | §5.1 서술 순서 |

실험 A의 누수(전처리 전체-fit, train·test 피험자 중복)는 고치지 않고 **observe 모드로
기록**한다 — 그것이 재현 대상이기 때문이다. 통제는 실험 B·C가 맡는다.

### 2-1. 논문 §5.1 본문 방식(A1)의 실행 가능성 — 실패가 예상된다

"각 특성값의 상·하위 10%를 벗어나는 행 제외"를 46개 변수에 적용하면 잔존율 3.05%,
Dem 6행이 되어 8:1:1 분할이 불가능하다. 아래 셀은 그 오류를 그대로 보여준다.
""")

code("""\
# A1: §5.1 본문(percentile) 방식 — InfeasibleSplitError가 나는 것이 정상이다
repro("--config configs/paper_percentile_latent500.yaml --dry-run", expect_failure=True)
""")

md("""\
### 2-2. 실험 A 본 실행 (config `A5`)

행 수가 논문과 정합하는 Isolation Forest + 표준화 공간 VAE 구성이다.
dry-run으로 분할 규모·예상 합성행(4,000)·표 5 대조를 먼저 확인한 뒤 학습한다.
""")

code("""\
repro("--config configs/paper_isoforest_scaled_latent500.yaml --dry-run")
""")

code("""\
# 실험 A 학습·평가 (T4 기준 약 5분)
repro("--config configs/paper_isoforest_scaled_latent500.yaml", collect="A")
""")

# ══════════════════════════════════════════════════════════════════════════════
md("""\
## 3단계 — 실험 B: 피험자 단위 분할 (그 외 조건은 논문과 동일)

**바꾸는 것은 분할 하나다.** 피험자 174명을 클래스별로 층화해 3-fold로 나누고
(`subject_stratified`), 같은 피험자의 기록이 train과 eval에 동시에 등장하지 못하게 한다.
모든 fold의 train·eval에 CN·MCI·Dem 피험자가 최소 1명씩 존재해야 하며, 아니면
`SplitError`로 중단된다.

| 항목 | 실험 A | 실험 B |
| --- | --- | --- |
| 분할 | 행 단위 8:1:1 | **피험자 단위 3-fold** |
| 전처리(imputer·scaler) fit | 전체 데이터 | **train fold만** |
| VAE fit | train Dem | **train fold의 실제 Dem만** (감사기가 강제) |
| 합성 Dem 규모 | 4,000행 고정 | 실제 train Dem의 **9.71배** (표 5와 같은 비율) |
| 하이퍼파라미터 | 논문 고정 | **논문 고정 (재탐색 없음)** |
| 평가 단위 | 기록 | **피험자** (일별 확률 산술평균 → argmax) |
| 누수 감사 | observe (기록만) | **enforce (위반 시 중단)** |

합성행에는 피험자 ID를 부여하지 않으며 피험자 단위 집계에 절대 들어가지 않는다.
""")

code("""\
repro("--config configs/leakage_controlled_non_nested.yaml --dry-run")
""")

code("""\
# 실험 B 학습·평가 (T4 기준 약 5–10분)
repro("--config configs/leakage_controlled_non_nested.yaml", collect="B")
""")

# ══════════════════════════════════════════════════════════════════════════════
md("""\
## 4단계 — 실험 C: Nested Group CV

피험자 단위 outer 3-fold의 **각 train fold 안에서** inner 3-fold CV가
이상치 방식(contamination)·증강 방법과 강도(none / VAE 배수 / class weight /
oversampling / SMOTE)·VAE latent 차원·분류기를 선택한다. outer test는 선택 과정에
한 번도 쓰이지 않는다. 탐색 예산은 분류기 × 증강법 조합에 균형 배분된다
(특정 조합이 "여러 번 시도해 이기는" 편향 방지).

⚠️ **가장 오래 걸린다** — outer 3 × (후보 × inner 3) + 최종 3회 ≈ 219회 모델 적합.
Colab 세션이 끊길 것 같으면 아래처럼 fold 하나씩 나눠 실행할 수 있다
(같은 세션 폴더에 누적된다). 단, **비교표·시각화는 세 fold가 모두 끝난 뒤에** 만들어라.

```python
repro("--config configs/nested_subject_independent.yaml --fold 0", collect="C")
repro("--config configs/nested_subject_independent.yaml --fold 1", collect="C")
repro("--config configs/nested_subject_independent.yaml --fold 2", collect="C")
```
""")

code("""\
repro("--config configs/nested_subject_independent.yaml --dry-run")
""")

code("""\
# 실험 C 학습·평가 (T4 기준 수 시간)
repro("--config configs/nested_subject_independent.yaml", collect="C")
""")

# ══════════════════════════════════════════════════════════════════════════════
md("""\
## 5단계 — 교차 실험 비교표

이 커널 세션에서 실행한 실험(RESULTS에 수집된 것)을 모아
`COMPARISON/` 폴더에 논문 보고값과 나란히 놓은 비교표를 만든다.
일부 실험만 실행했어도 그 부분만으로 표를 만든다.
""")

code("""\
if RESULTS:
    from src.experiments.compare import CrossExperimentResults, assemble_comparison

    _collected = CrossExperimentResults()
    _adders = {"A": _collected.add_experiment_a,
               "B": _collected.add_experiment_b,
               "C": _collected.add_experiment_c}
    for _kind, _result in RESULTS.items():
        _adders[_kind](_result)
    _summary = assemble_comparison(_collected, out_root=str(SESSION_DIR))
    print("비교표 저장 →", SESSION_DIR / "COMPARISON")
    print("포함된 실험:", _summary.get("experiments_present"))
    _main_md = SESSION_DIR / "COMPARISON" / "main_comparison_subject_level.md"
    if _main_md.exists():
        print()
        print(_main_md.read_text(encoding="utf-8"))
else:
    print("이 커널 세션에서 실행한 실험이 없다 — 비교표 생성을 건너뛴다.")
    print("(아래 시각화는 기존 산출물 폴더를 자동으로 찾아 그린다.)")
""")

# ══════════════════════════════════════════════════════════════════════════════
md("""\
## 6단계 — 시각화

각 그림이 답하는 질문:

1. **그림 1** — 논문 절차를 따르면 논문 수치가 나오는가? (기록 단위 F1: 논문 그림 3·표 6 vs 실험 A)
2. **그림 2** — 검증 설계를 바꾸면 성능이 어떻게 변하는가? (피험자 단위 macro-F1: A → B → C)
3. **그림 3** — Dem 피험자를 실제로 몇 명 찾는가? (조기 탐지 관점의 핵심 지표)
4. **그림 4** — fold·파이프라인 간 분산 — Dem 피험자 12명에서 오는 불안정성

위 실험 셀을 이번 세션에 돌리지 않았어도, 이 세션 폴더 → 최근 세션 →
저장소에 보관된 과거 실행(`reproduction_lee_lee_2025_vae_result/`) 순으로
산출물을 자동 탐색해 그린다. 특정 실행을 그리려면 `RESULT_DIR`에 경로를 지정하라.
""")

code("""\
# ── 결과 로딩 ────────────────────────────────────────────────────────────────
RESULT_DIR = None  # 예: Path("/content/drive/MyDrive/reproduction_lee_lee_2025_vae/20260803_063643_full")

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _has_results(p):
    return (any(p.glob("A_*/record_level_metrics.csv"))
            or any(p.glob("B_*/fold_metrics.csv"))
            or any(p.glob("C_*/outer_fold_metrics.csv")))


def discover_result_dir():
    cands = []
    if RESULT_DIR:
        cands.append(Path(RESULT_DIR))
    if "SESSION_DIR" in globals():
        cands.append(Path(SESSION_DIR))
    base = Path(OUT_BASE)
    if base.exists():
        cands += sorted((p for p in base.iterdir() if p.is_dir()),
                        key=lambda p: p.name, reverse=True)
        cands.append(base)  # 세션 폴더 없이 바로 쓴 옛 layout
    archived = REPRO_DIR / "reproduction_lee_lee_2025_vae_result"
    if archived.exists():
        cands += sorted((p for p in archived.iterdir() if p.is_dir()),
                        key=lambda p: p.name, reverse=True)
    for c in cands:
        if c.exists() and _has_results(c):
            return c
    raise FileNotFoundError(
        "실험 산출물을 찾지 못했다. 위 실험 셀을 먼저 실행하거나 RESULT_DIR을 지정하라.")


RES = discover_result_dir()
print("시각화 대상 산출물:", RES)


def _first(pattern, prefer="A5"):
    hits = sorted(RES.glob(pattern))
    pref = [h for h in hits if prefer in str(h)]
    return (pref or hits)[-1] if hits else None


def _read_csv(pattern):
    p = _first(pattern)
    return pd.read_csv(p) if p else None


def _read_json(pattern):
    p = _first(pattern)
    return json.loads(p.read_text(encoding="utf-8")) if p else None


A_df = _read_csv("A_*/record_level_metrics.csv")
B_fold = _read_csv("B_*/fold_metrics.csv")
B_pooled = _read_json("B_*/pooled_subject_metrics.json")
C_fold = _read_csv("C_*/outer_fold_metrics.csv")
C_pooled = _read_json("C_*/pooled_subject_metrics.json")
C_sel = _read_csv("C_*/selected_pipelines.csv")

for _name, _obj in [("A record_level_metrics", A_df), ("B fold_metrics", B_fold),
                    ("B pooled_subject_metrics", B_pooled),
                    ("C outer_fold_metrics", C_fold),
                    ("C pooled_subject_metrics", C_pooled)]:
    _state = "✅" if _obj is not None else "— 없음 (해당 실험 미실행)"
    _n = f" ({len(_obj)}행)" if isinstance(_obj, pd.DataFrame) else ""
    print(f"  {_name:26s}: {_state}{_n}")

# ── 스타일: 검증 통과한 categorical 팔레트 (dataviz 기준 인스턴스, light) ────
C_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue/orange/aqua/yellow
C_PAPER = "#52514e"   # 논문 보고값·비모델 집계용 회색 잉크
C_TEXT = "#0b0b0b"
C_TEXT2 = "#52514e"
C_GRID = "#e4e3df"
C_SURFACE = "#fcfcfb"

MODELS = ["xgboost", "dnn", "tabnet", "wide_deep"]
MODEL_LABEL = {"xgboost": "XGBoost", "dnn": "DNN",
               "tabnet": "TabNet", "wide_deep": "Wide & Deep"}
MODEL_COLOR = dict(zip(MODELS, C_SERIES))  # 모델 색은 모든 그림에서 고정

plt.rcParams.update({
    "figure.facecolor": C_SURFACE, "axes.facecolor": C_SURFACE,
    "axes.edgecolor": C_GRID, "axes.labelcolor": C_TEXT,
    "text.color": C_TEXT, "xtick.color": C_TEXT2, "ytick.color": C_TEXT2,
    "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "figure.dpi": 110,
})

# ── 논문 보고값 (그림 3 + 표 6 정본, 기록 단위·VAE 증강) ─────────────────────
# 주의(report_inconsistencies.md):
#  - XGBoost MCI는 그림 3 라벨(0.7501)과 본문(0.7581)이 다르다.
#    평균 0.8103과 정합하는 본문 값을 쓴다.
#  - Wide & Deep은 본문이 Dem F1과 평균을 뒤바꿔 적었다. 자기정합하는 표 6을 쓴다.
PAPER_F1_VAE = {
    "xgboost":   {"CN": 0.8914, "MCI": 0.7581, "Dem": 0.7816, "Avg": 0.8103},
    "dnn":       {"CN": 0.8958, "MCI": 0.7770, "Dem": 0.7527, "Avg": 0.8085},
    "tabnet":    {"CN": 0.8762, "MCI": 0.7485, "Dem": 0.7391, "Avg": 0.7879},
    "wide_deep": {"CN": 0.8897, "MCI": 0.8022, "Dem": 0.8750, "Avg": 0.8556},
}
PAPER_WD_TABLE6 = {  # 표 6: Wide & Deep 증강 전/후 (기록 단위)
    "none": {"CN": 0.9165, "MCI": 0.8385, "Dem": 0.8298, "Avg": 0.8616, "Dem_recall": 0.7647},
    "vae":  {"CN": 0.8897, "MCI": 0.8022, "Dem": 0.8750, "Avg": 0.8556, "Dem_recall": 0.8235},
}
""")

code("""\
# ── 그림 1 — 논문 보고값 vs 실험 A (기록 단위 F1, VAE 증강) ──────────────────
if A_df is None:
    print("실험 A 산출물이 없어 그림 1을 건너뛴다.")
else:
    CLASSES = ["CN", "MCI", "Dem", "Avg"]
    _a_vae = A_df[A_df["augmentation"] == "vae"].set_index("model")

    def _ours_f1(m, cls):
        if m not in _a_vae.index:
            return None
        row = _a_vae.loc[m]
        return float(row["record_macro_f1"] if cls == "Avg" else row[f"record_{cls}_f1"])

    fig, axes = plt.subplots(1, len(CLASSES), figsize=(13.5, 3.6), sharex=True, sharey=True)
    for ax, cls in zip(axes, CLASSES):
        for i, m in enumerate(MODELS):
            y = len(MODELS) - 1 - i
            pv, ov = PAPER_F1_VAE[m][cls], _ours_f1(m, cls)
            if ov is not None:
                ax.plot([pv, ov], [y, y], color=C_GRID, lw=2, zorder=1)
                ax.scatter([ov], [y], s=52, color=MODEL_COLOR[m], zorder=3)
                ax.annotate(f"{ov:.3f}", (ov, y), xytext=(0, 8), textcoords="offset points",
                            ha="center", fontsize=8, color=MODEL_COLOR[m])
            ax.scatter([pv], [y], s=52, facecolor=C_SURFACE, edgecolor=C_PAPER,
                       linewidth=1.6, zorder=2)
            ax.annotate(f"{pv:.3f}", (pv, y), xytext=(0, -15), textcoords="offset points",
                        ha="center", fontsize=8, color=C_PAPER)
        ax.set_title({"Avg": "Macro avg"}.get(cls, cls), fontsize=11)
        ax.set_ylim(-0.7, len(MODELS) - 0.3)
        ax.set_xlim(0.3, 1.02)
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([MODEL_LABEL[m] for m in MODELS[::-1]])
        ax.grid(axis="x")
        ax.grid(False, axis="y")
    _handles = [
        plt.Line2D([], [], marker="o", ls="", mfc=C_SURFACE, mec=C_PAPER,
                   label="Paper (Fig. 3 / Table 6)"),
        plt.Line2D([], [], marker="o", ls="", color=C_TEXT2,
                   label="Reproduction A (record-level, filled = model color)"),
    ]
    fig.legend(handles=_handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.12))
    fig.suptitle("Record-level F1 (VAE-augmented): paper vs reproduction A", y=1.2, fontsize=13)
    fig.tight_layout()
    plt.show()
    print("읽는 법: 회색 테두리 = 논문 보고값, 채워진 점 = 이 재현(실험 A). 연결선이 짧을수록")
    print("논문 수치에 가깝다. 논문 값 중 XGBoost MCI·Wide & Deep Dem/평균은 본문·그림이 서로")
    print("달라 자기정합하는 쪽(본문 0.7581, 표 6)을 썼다 — report_inconsistencies.md 참조.")
""")

code("""\
# ── 그림 2 — 검증 설계에 따른 피험자 단위 macro-F1 (A → B → C) ───────────────
if A_df is None and B_pooled is None and C_pooled is None:
    print("그릴 산출물이 없다.")
else:
    AUGS = ["none", "vae"]
    _xpos = {"A": 0, "B": 1, "C": 2}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, aug in zip(axes, AUGS):
        _endlabels = []
        for m in MODELS:
            xs, ys = [], []
            if A_df is not None:
                row = A_df[(A_df["model"] == m) & (A_df["augmentation"] == aug)]
                if len(row):
                    xs.append(_xpos["A"])
                    ys.append(float(row["subject_macro_f1"].iloc[0]))
            if B_pooled is not None and f"{m}|{aug}" in B_pooled:
                xs.append(_xpos["B"])
                ys.append(float(B_pooled[f"{m}|{aug}"]["macro_f1"]))
            if xs:
                ax.plot(xs, ys, marker="o", ms=6, lw=2, color=MODEL_COLOR[m],
                        label=MODEL_LABEL[m])
                _endlabels.append((xs[-1], ys[-1], f"{ys[-1]:.2f}", MODEL_COLOR[m]))
        # 끝점 값 레이블 — 겹치면 세로 간격을 강제로 벌린다
        _endlabels.sort(key=lambda t: t[1], reverse=True)
        _prev_y = None
        for _lx, _ly, _txt, _lc in _endlabels:
            _y = _ly if _prev_y is None else min(_ly, _prev_y - 0.045)
            ax.text(_lx + 0.08, _y, _txt, va="center", fontsize=8, color=_lc)
            _prev_y = _y
        if C_pooled is not None:
            ax.scatter([_xpos["C"]], [C_pooled["macro_f1"]], marker="D", s=64,
                       color=C_PAPER, zorder=3)
            ax.annotate(f"{C_pooled['macro_f1']:.2f}",
                        (_xpos["C"], C_pooled["macro_f1"]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8, color=C_PAPER)
        ax.set_xticks(list(_xpos.values()))
        ax.set_xticklabels(["A\\nrow split", "B\\nsubject split", "C\\nnested CV"])
        ax.set_xlim(-0.4, 2.5)
        ax.set_ylim(0, 1)
        ax.set_title(f"augmentation: {aug}", fontsize=11)
        ax.grid(axis="y")
        ax.grid(False, axis="x")
    axes[0].set_ylabel("Subject-level macro-F1")
    axes[1].legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("Validation design vs subject-level macro-F1", fontsize=13)
    fig.tight_layout()
    plt.show()
    print("A조차 피험자 단위로 '집계'만 했을 뿐 분할은 행 단위라서, 같은 피험자가 train과")
    print("test에 모두 등장한다 — A의 높은 값은 그 누수를 포함한 수치다. C(Nested CV)는")
    print("fold마다 inner CV가 파이프라인을 새로 고르므로 모델별 곡선 대신 합동(pooled)")
    print("한 점(회색 마름모)으로 표시한다.")
""")

code("""\
# ── 그림 3 — Dem 피험자 탐지 (피험자 단위 recall, 분모는 실제 피험자 수) ─────
if A_df is None and B_pooled is None:
    print("그릴 산출물이 없다.")
else:
    AUGS = ["none", "vae"]
    _w = 0.38
    _x = np.arange(len(MODELS))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, aug in zip(axes, AUGS):
        for j, m in enumerate(MODELS):
            if A_df is not None:
                row = A_df[(A_df["model"] == m) & (A_df["augmentation"] == aug)]
                if len(row):
                    r = row.iloc[0]
                    v = float(r["subject_Dem_recall"])
                    ax.bar(j - _w / 2, v, _w * 0.92, color=MODEL_COLOR[m], zorder=2)
                    ax.annotate(f"{int(r['subject_n_Dem_correct'])}/{int(r['subject_n_Dem'])}",
                                (j - _w / 2, v), xytext=(0, 3), textcoords="offset points",
                                ha="center", fontsize=8)
            if B_pooled is not None and f"{m}|{aug}" in B_pooled:
                d = B_pooled[f"{m}|{aug}"]
                ax.bar(j + _w / 2, d["Dem_recall"], _w * 0.92, facecolor=C_SURFACE,
                       edgecolor=MODEL_COLOR[m], hatch="//", linewidth=1.2, zorder=2)
                ax.annotate(f"{int(d['n_Dem_correct'])}/{int(d['n_Dem'])}",
                            (j + _w / 2, d["Dem_recall"]), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=8)
        if C_pooled is not None:
            ax.axhline(C_pooled["Dem_recall"], color=C_PAPER, lw=1.4, ls="--", zorder=1)
        ax.set_xticks(_x)
        ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS], fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_title(f"augmentation: {aug}", fontsize=11)
        ax.grid(axis="y")
        ax.grid(False, axis="x")
    axes[0].set_ylabel("Subject-level Dem recall")
    import matplotlib.patches as mpatches
    _handles = [
        mpatches.Patch(facecolor=C_TEXT2, label="A (row split, test Dem subjects)"),
        mpatches.Patch(facecolor=C_SURFACE, edgecolor=C_TEXT2, hatch="//",
                       label="B (subject split, all 12 Dem subjects)"),
    ]
    if C_pooled is not None:
        _handles.append(plt.Line2D([], [], color=C_PAPER, ls="--",
                                   label=f"C nested, pooled: "
                                         f"{int(C_pooled['n_Dem_correct'])}/{int(C_pooled['n_Dem'])}"))
    fig.legend(handles=_handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.09), fontsize=9)
    fig.suptitle("How many real Dem subjects are detected?", y=1.16, fontsize=13)
    fig.tight_layout()
    plt.show()
    print("막대 위 숫자 = 탐지한 Dem 피험자 수 / 평가에 포함된 실제 Dem 피험자 수.")
    print("주의: A는 행 단위 test에 등장한 Dem 피험자만 분모가 되고(보통 12명 중 일부),")
    print("B·C는 3-fold를 합쳐 12명 전원이 정확히 한 번씩 평가된다 — 분모가 다르다.")
""")

code("""\
# ── 그림 4 — fold·파이프라인 간 분산 (피험자 단위 macro-F1) ──────────────────
_cols = []  # (라벨, [(값, 색, 채움, 마름모)…])
if A_df is not None:
    _cols.append(("A\\nrow split\\n(pipelines)", [
        (float(r["subject_macro_f1"]), MODEL_COLOR.get(r["model"], C_PAPER),
         r["augmentation"] == "vae", False)
        for _, r in A_df.iterrows()]))
if B_fold is not None:
    _cols.append(("B\\nsubject split\\n(folds x pipelines)", [
        (float(r["subject_macro_f1"]), MODEL_COLOR.get(r["model"], C_PAPER),
         r["augmentation"] == "vae", False)
        for _, r in B_fold.iterrows()]))
if C_fold is not None:
    _cols.append(("C\\nnested CV\\n(outer folds)", [
        (float(r["subject_macro_f1"]), C_PAPER, True, True)
        for _, r in C_fold.iterrows()]))

if not _cols:
    print("그릴 산출물이 없다.")
else:
    _rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for xc, (_label, _pts) in enumerate(_cols):
        for v, c, filled, diamond in _pts:
            ax.scatter([xc + _rng.uniform(-0.14, 0.14)], [v],
                       s=52 if diamond else 40, marker="D" if diamond else "o",
                       facecolor=c if filled else C_SURFACE, edgecolor=c,
                       linewidth=1.2, zorder=2, alpha=0.9)
        _median = float(np.median([v for v, *_ in _pts]))
        ax.hlines(_median, xc - 0.26, xc + 0.26, color=C_TEXT, lw=2, zorder=3)
        ax.annotate(f"median {_median:.2f}", (xc + 0.3, _median), va="center",
                    fontsize=9, color=C_TEXT)
    ax.set_xticks(range(len(_cols)))
    ax.set_xticklabels([lab for lab, _ in _cols], fontsize=9)
    ax.set_xlim(-0.5, len(_cols) - 0.5 + 0.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Subject-level macro-F1")
    ax.grid(axis="y")
    ax.grid(False, axis="x")
    _handles = [plt.Line2D([], [], marker="o", ls="", color=MODEL_COLOR[m],
                           label=MODEL_LABEL[m]) for m in MODELS]
    _handles += [
        plt.Line2D([], [], marker="o", ls="", color=C_TEXT2, label="filled = VAE aug"),
        plt.Line2D([], [], marker="o", ls="", mfc=C_SURFACE, mec=C_TEXT2, label="open = no aug"),
        plt.Line2D([], [], marker="D", ls="", color=C_PAPER, label="C outer fold"),
    ]
    ax.legend(handles=_handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, fontsize=8)
    ax.set_title("Dispersion across folds and pipelines", fontsize=13)
    fig.tight_layout()
    plt.show()
    print("Dem 피험자가 12명뿐이라 fold 구성이 조금만 달라져도 피험자 단위 성능이 크게")
    print("흔들린다. B·C 해석은 반드시 fold 구성(fold_composition.csv)과 함께 읽어라.")
""")

# ══════════════════════════════════════════════════════════════════════════════
md("""\
## 결과 해석 시 반드시 지킬 것

> 본 실험의 Dem 클래스는 **독립 피험자 12명**에서 유래한다.
> 합성 Dem 행 N개는 해당 fold의 실제 train Dem 피험자 기록 분포에서 생성된 것이며
> **새로운 피험자를 의미하지 않는다.** 피험자 단위 metric의 분모는 항상 실제 피험자 수다.

- **A와 B·C의 수치를 같은 지표처럼 비교하지 마라.** A는 행 단위 분할(같은 피험자가
  train·test에 중복)이고, 피험자 집계도 그 누수 위에서 계산된다. B·C만이
  "새 피험자에 대한 일반화"를 측정한다.
- fold별 train/eval Dem 피험자 수는 산출물의 `fold_composition.csv` /
  `outer_fold_composition.csv`와 `n_dem_subjects` 열에서 확인하고, 문서에 특정 수를
  고정해 쓰지 마라.
- Dem 12명 기준의 신뢰구간은 매우 넓다 (`pooled_subject_metrics.json`의
  `bootstrap_ci`). 점추정만 인용하지 마라.
- 과거 실행(2026-08-03, 교정 전 코드)에서는 행 단위 → 피험자 단위 전환만으로
  macro-F1이 0.98 → 0.37로 떨어졌다. **정확한 수치는 반드시 이 노트북의 최신 실행
  산출물에서 인용하라** — 위 시각화가 어느 폴더를 읽었는지 6단계 첫 셀이 출력한다.
- 합성자료 해석 위험과 진단 절차:
  [synthetic_data_risk.md](@DOC@/synthetic_data_risk.md) ·
  누수 통제 설계: [leakage_audit.md](@DOC@/leakage_audit.md)
""")

nb = {
    "cells": [{**c, "id": f"cell-{i:02d}"} for i, c in enumerate(cells)],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"provenance": [], "toc_visible": True},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
size_kb = OUT.stat().st_size / 1024
print(f"wrote {OUT}")
print(f"  cells: {len(nb['cells'])} | size: {size_kb:.0f} KB | 내장 파일 {len(MANIFEST)}개 | 커밋 {COMMIT}")
