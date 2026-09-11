# %%% CELL 0 [markdown]
# Binary_Ceiling_Nested — Experiment 3 (사전등록, 사용자 지시 "최대한 0.70 이상": 상한(ceiling) 실험, exploratory-record)

**과제**: CN(111) vs MCI+Dem(63), 174명. **피험자 단위 분할 + nested CV + MMSE 완전 제외** (변경 불가).
**질문**: 이 코호트에서 **합법적인 양의 레버를 전부 동시에** 걸면 (전체 조인 유효일 측정 + 유일하게 음수가 아니었던 추가 특징 + fold 안에서만 고르는 정직한 후보 선택)
subject-mean merged AUC 가 **0.70 에 도달하는가?** 도달하지 못하면 그 최댓값(B0 포함 4 arm 의 max)이 이 코호트의 **방어 가능한 상한**이다.
**성격**: STRICT3 §10 stop rule(2026-09-04) 이후 **사용자 지시(2026-09-05)** 로 1회 실행하는 상한 측정. §10.4 의 두 항목(새 특징 정의, fold-local pool 스크린)을 **이 노트북 1회에 한해** 예외 처리하며, 결과가 어떻든 후속 특징 실험·라이브러리 확장·진단 arm 승격·정의 수정을 하지 않는다. 프로그램 headline 은 여전히 B0 0.6491 (STRICT3 §10.2).

## 실행 전에 알려진 것 (정직 공시)
| 사실 | 값 |
|---|---|
| 동결 baseline B0 (첫 35일 `low_SD`, `low_MED`) | merged **0.6491** [0.562, 0.731], CN-MCI 0.6054, CN-Dem 0.8348 (run 2; run 3·4 비트 재현) — **재현 assert** |
| A1 = 전체일 측정 (run 4 S1; STRICT3 §8 #18 "더 많은 날로 재면 좋아진다" 기각 항목) | 0.6577 (MCI 0.6123 / Dem 0.8506), Δ +0.008 [−0.034, +0.050] — **재현 assert** (라벨 기반 기지값) |
| 35일 `B0 + longest_w_SD` (run 4 P; §8 #17 T3) | 0.6589 [0.575, 0.740], Δ +0.010, p_Δ 0.46 — **D1 재현 assert** (라벨 기반 기지값) |
| 산술 | merged = (51·A_MCI + 12·A_Dem)/63 → A_Dem 0.835~0.85 에서 0.70 은 CN-vs-MCI ≥ 0.665~0.668 필요 (현재 0.605~0.613). 네 번의 사전등록 추가에서 Δ_MCI 최대 +0.014 |
| 사전 기대 | A2 ≈ 0.66~0.67, A3 ≤ A2 (inner 선택 잡음). SE 0.044 → **P(후보 arm 점추정 ≥ 0.70) ≈ 18%, P(G) < 5%** (PREREGISTRATION §7 산술) |
| 라벨 노출 (후보별) | `longest_w_SD_all`: 구성개념 노출(run 4, 35일판) · `onset_SD_all`: onset 은 run 2/3 결합 특징의 재료였으나 onset SD 자체의 라벨 값은 미계산 · `clsH_SD_all`: class_5min 스트림은 Training-141 대규모 스크린(§8 #8, §9.2)에 변형 존재, 이 정의의 라벨 값 미계산 · `low_lag1_all`: 미계산 · **`steps_SD` 는 run 2 교란 감사에 174명 Dem 축 AUC 0.6219 가 기록돼 있어 라이브러리에서 제외** |
| 라벨을 보는 arm 수 | main 4 + 진단 5 (D1~D5) + 축소 프로토콜 4 = 13. 판정은 후보 3 arm 의 max 하나에만 (Holm 3) |

## Arm (K=4, 헤드라인 경쟁)
| arm | 특징 | 창 |
|---|---|---|
| B0 | `low_SD`, `low_MED` | 첫 35 조인일 (동결, 비트 재현 assert) |
| A1 | `low_SD_all` (ok_all 일, 날짜축 선형 detrend 후 SD), `low_MED_all` (원값 중앙값) | ok_all = 착용률 ≥ 0.8 & MET==0.0 run ≤ 60분 (run 4 S1 정의 그대로) |
| A2 MAXFIX | A1 + `longest_w_SD_all` | 창 유효일 (ok_all & 깨어있는 창 ≥ 240분) |
| A3 MAXSEL | A1 + {∅ 또는 라이브러리 1개} — **inner CV 가 fold 마다 고른다** (라이브러리 4개: `longest_w_SD_all`, `onset_SD_all`, `clsH_SD_all`, `low_lag1_all`) | 위와 동일 |

**판정 통계량(유일)**: BEST = 후보 3 arm 중 subject-mean AUC 최댓값; G = 점추정 ≥ 0.70 **and** (paired Δ CI 하한 > 0 **and** Holm p_Δ(BEST) < 0.05). 0.70 자체는 어떤 tier 에서도 검정되지 않는다. p_Δmax · p_max · p_col · 진단 D 값은 기술용이며 판정·승격에 쓰지 않는다.

# %%% CELL 1 [code]
# ============================================================================
# 셀 1 — 환경 (BLAS 스레드는 numpy import 전에 고정), 시드, 경로, 실행 모드
# ============================================================================
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import re, sys, json, math, time, hashlib, warnings
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:
    pass
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

EXPERIMENT = "Binary_Ceiling_Nested"
SEED = 20260903                     # run-2 / Exp 1 / Exp 2 와 동일 → 동일 fold / bootstrap seed 계약
np.random.seed(SEED)

# 실행 모드 (환경변수) — 코드는 바꾸지 않고 배선 테스트만 축소한다. 실제 run 은 둘 다 꺼져 있어야 한다.
QUICK     = os.environ.get("EXP3_QUICK", "0") == "1"       # 축소 프로토콜: 결과 무효, 배선 테스트 전용
SYNTHETIC = os.environ.get("EXP3_SYNTHETIC", "0") == "1"   # 합성 데이터: 재현 assert 건너뜀
IN_COLAB = "google.colab" in sys.modules
import sklearn
print(f"Colab {IN_COLAB} | numpy {np.__version__} | pandas {pd.__version__} | sklearn {sklearn.__version__}"
      f" | QUICK {QUICK} | SYNTHETIC {SYNTHETIC}")
if QUICK or SYNTHETIC:
    print("⚠⚠ 배선 테스트 모드 — 이 실행의 숫자는 어디에도 보고하지 않는다.")

if IN_COLAB:
    from google.colab import drive
    try:
        drive.mount("/content/drive")
    except Exception as exc:
        print(f"drive.mount 건너뜀: {exc}")

_env_root = os.environ.get("EXP3_DATA_ROOT")
_CAND = ([Path(_env_root)] if _env_root else []) + [
    Path("/content/drive/Shareddrives/GoogleAI_contest/Data"),
    Path("/content/drive/MyDrive/GoogleAI_contest/Data"),
    Path("/content/drive/MyDrive/Data")]
_here = Path.cwd().resolve()
for _p in [_here, *_here.parents]:
    _CAND.append(_p / "Data")
DATA_ROOT = None
for _c in _CAND:
    if (_c / "1.Training" / "SourceData" / "1.Gait" / "train_activity.csv").exists():
        DATA_ROOT = _c.resolve(); break
if DATA_ROOT is None:
    raise FileNotFoundError("Data 폴더를 찾지 못했습니다:\n  " + "\n  ".join(map(str, _CAND)))
print(f"DATA_ROOT : {DATA_ROOT}")

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc") + ("_QUICK" if QUICK else "") + ("_SYN" if SYNTHETIC else "")
if os.environ.get("EXP3_OUT_ROOT"):
    _base = Path(os.environ["EXP3_OUT_ROOT"])
elif Path("/content/drive/MyDrive").exists():
    _base = Path("/content/drive/MyDrive")
elif _here.name == EXPERIMENT:
    _base = _here
elif (DATA_ROOT.parent / "SangHyo" / "Binary" / EXPERIMENT).exists():
    _base = DATA_ROOT.parent / "SangHyo" / "Binary" / EXPERIMENT
else:
    _base = _here
OUT_DIR = _base / f"{EXPERIMENT}_result" / RUN_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"OUT_DIR   : {OUT_DIR}")
N_JOBS = max(1, os.cpu_count() or 1)
# 실험 폴더 (사전등록 문서·노트북 위치): 환경변수 > 로컬 저장소 관례
EXP_DIR = None
for _p in [Path(os.environ.get("EXP3_EXP_DIR")) if os.environ.get("EXP3_EXP_DIR") else None,
           DATA_ROOT.parent / "SangHyo" / "Binary" / EXPERIMENT,
           _here if _here.name == EXPERIMENT else None]:
    if _p is not None and _p.is_dir():
        EXP_DIR = _p; break
print(f"EXP_DIR   : {EXP_DIR}")

_T_START = time.time()
def progress(stage, done, total, **extra):
    """fold 단위 진행 기록 → PROGRESS.json (다른 터미널에서 열어 볼 수 있다)."""
    el = time.time() - _T_START
    rec = {"stage": stage, "done": int(done), "total": int(total),
           "elapsed_min": round(el / 60, 2),
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **extra}
    try:
        (OUT_DIR / "PROGRESS.json").write_text(json.dumps(rec, ensure_ascii=False, default=float))
    except Exception:
        pass
progress("init", 0, 1)

# %%% CELL 2 [code]
# ============================================================================
# 셀 2 — activity ⋈ sleep 조인 + 라벨 계약 + MMSE fail-closed 가드 (run-2 / Exp 1 / Exp 2 와 동일; 읽는 컬럼 1개 추가: class_5min)
# ============================================================================
FORBIDDEN_PATTERN = re.compile(
    r"(?:^|_)mmse|^q\d+(?:_|$)|^total$|diag|doctor|sample_email|email"
    r"|non_?wear|(?:^|_)n_days|(?:^|_)span(?:$|_)"
    r"|(?:^|_)cover(?:age)?(?:$|_)|(?:^|_)gap_|missfrac|valid_days",
    re.IGNORECASE)
_MUST_PASS = ["low_SD", "low_MED", "longest_w_SD", "low_SD_all", "low_MED_all", "longest_w_SD_all",
              "onset_SD_all", "clsH_SD_all", "low_lag1_all", "cf_nvalid_all", "cf_npairs_all", "activity_total"]
_MUST_FAIL = ["mmse_TOTAL", "TOTAL", "Q13_2", "DIAG_NM", "activity_non_wear", "cf_nonwear_mean_all",
              "n_days", "hr_missfrac", "span_days", "coverage", "gap_days", "cf_gap_days_all", "valid_days"]
assert not [n for n in _MUST_PASS if FORBIDDEN_PATTERN.search(n)], "가드 오탐"
assert all(FORBIDDEN_PATTERN.search(n) for n in _MUST_FAIL), "가드 미탐"

def assert_no_forbidden(names, where):
    bad = [n for n in names if FORBIDDEN_PATTERN.search(str(n))]
    if bad:
        raise RuntimeError(f"[fail-closed] {where}에 금지 컬럼: {bad}")

SRC = {
    "act_tr":   DATA_ROOT / "1.Training"   / "SourceData"   / "1.Gait" / "train_activity.csv",
    "act_va":   DATA_ROOT / "2.Validation" / "SourceData"   / "1.Gait" / "val_activity.csv",
    "slp_tr":   DATA_ROOT / "1.Training"   / "SourceData"   / "2.Sleep"/ "train_sleep.csv",
    "slp_va":   DATA_ROOT / "2.Validation" / "SourceData"   / "2.Sleep"/ "val_sleep.csv",
    "lab_tr_g": DATA_ROOT / "1.Training"   / "LabelingData" / "1.Gait" / "training_label.csv",
    "lab_tr_s": DATA_ROOT / "1.Training"   / "LabelingData" / "2.Sleep"/ "training_label.csv",
    "lab_va_g": DATA_ROOT / "2.Validation" / "LabelingData" / "1.Gait" / "val_label.csv",
    "lab_va_s": DATA_ROOT / "2.Validation" / "LabelingData" / "2.Sleep"/ "val_label.csv",
}
for k, p in SRC.items():
    if "3.CognitiveFunction" in str(p):
        raise RuntimeError("[fail-closed] MMSE 경로가 소스 목록에 있습니다.")
    if not p.exists():
        raise FileNotFoundError(f"{k}: {p}")

_t0 = time.time()
def _read_label(pg, ps):
    g = pd.read_csv(pg)[["SAMPLE_EMAIL", "DIAG_NM"]].copy()
    s = pd.read_csv(ps)[["SAMPLE_EMAIL", "DIAG_NM"]].copy()
    for d in (g, s):
        d["SAMPLE_EMAIL"] = d["SAMPLE_EMAIL"].astype(str).str.strip()
    g = g.sort_values("SAMPLE_EMAIL").reset_index(drop=True)
    s = s.sort_values("SAMPLE_EMAIL").reset_index(drop=True)
    if not g.equals(s):
        raise RuntimeError("Gait / Sleep 라벨 사본 불일치")
    return g

lab = pd.concat([_read_label(SRC["lab_tr_g"], SRC["lab_tr_s"]),
                 _read_label(SRC["lab_va_g"], SRC["lab_va_s"])], ignore_index=True)
lab = lab.rename(columns={"SAMPLE_EMAIL": "sid"})
lab["y3"] = lab["DIAG_NM"].map({"CN": 0, "MCI": 1, "Dem": 2}).astype(int)
lab["y"] = (lab["y3"] > 0).astype(int)
_c = lab["DIAG_NM"].value_counts().to_dict()
print(f"피험자 {len(lab)}명  CN {_c.get('CN',0)} / MCI {_c.get('MCI',0)} / Dem {_c.get('Dem',0)}")
assert len(lab) == 174 and _c.get("CN") == 111 and _c.get("MCI") == 51 and _c.get("Dem") == 12

MET_COL = "CONVERT(activity_met_1min USING utf8)"        # 'activity_met_1min' 컬럼은 '...' 플레이스홀더라 못 쓴다
CLS_COL = "CONVERT(activity_class_5min USING utf8)"      # 5분 activity class 스트림 (0 비착용, 1 rest, 2 inactive, 3 low, 4 medium, 5 high)
ACT_COLS = ["EMAIL", "activity_day_start", "activity_low", "activity_medium",
            "activity_non_wear", "activity_total", CLS_COL, MET_COL]
SLP_COLS = ["EMAIL", "sleep_bedtime_start", "sleep_bedtime_end", "sleep_duration", "sleep_total"]
act = pd.concat([pd.read_csv(SRC["act_tr"], usecols=ACT_COLS, low_memory=False),
                 pd.read_csv(SRC["act_va"], usecols=ACT_COLS, low_memory=False)], ignore_index=True)
slp = pd.concat([pd.read_csv(SRC["slp_tr"], usecols=SLP_COLS, low_memory=False),
                 pd.read_csv(SRC["slp_va"], usecols=SLP_COLS, low_memory=False)], ignore_index=True)
act["sid"] = act["EMAIL"].astype(str).str.strip()
slp["sid"] = slp["EMAIL"].astype(str).str.strip()

def _naive(s):   # utc=True 로 바꾸면 04:00 앵커와 시각 특징이 깨진다
    return pd.to_datetime(s.astype(str).str.slice(0, 19), format="%Y-%m-%dT%H:%M:%S", errors="coerce")

act["ts"] = _naive(act["activity_day_start"]); act["date"] = act["ts"].dt.date
slp["ts_start"] = _naive(slp["sleep_bedtime_start"])
slp["ts_end"] = _naive(slp["sleep_bedtime_end"]);  slp["date"] = slp["ts_end"].dt.date
_hh = act["ts"].dt.hour
print(f"day_start 시각 {_hh.value_counts().to_dict()} (전부 4시여야 분 index ↔ 시각 매핑 성립)")
assert (_hh == 4).all(), "activity_day_start 가 04:00 이 아닙니다"

# run-2 와 동일한 중복 처리 (B0 의 day set 을 비트 수준으로 보존)
slp = (slp.sort_values(["sid", "date", "sleep_duration"], ascending=[True, True, False])
          .drop_duplicates(["sid", "date"], keep="first").reset_index(drop=True))
act = (act.sort_values(["sid", "date"]).drop_duplicates(["sid", "date"], keep="first")
          .reset_index(drop=True))

daily = act.merge(slp, on=["sid", "date"], how="inner", suffixes=("", "_s"))
daily = daily[daily["sid"].isin(set(lab["sid"]))].sort_values(["sid", "date"]).reset_index(drop=True)
_n = daily.groupby("sid").size()
print(f"조인 {len(daily):,}행 / {daily['sid'].nunique()}명 / 일수 min {_n.min()} med {int(_n.median())} max {_n.max()}")
assert daily["sid"].nunique() == 174 and _n.min() >= 35

SIDS = sorted(lab["sid"].tolist())
lab = lab.set_index("sid").loc[SIDS].reset_index()
y3 = lab["y3"].to_numpy(int); y = lab["y"].to_numpy(int); ybin = y.copy()
# 결과 파일에는 원본 이메일을 쓰지 않는다 (AGENTS.md §6): SHA-256 해시 ID 16자
HID = {s: hashlib.sha256(s.encode("utf-8")).hexdigest()[:16] for s in SIDS}
assert len(set(HID.values())) == len(SIDS)
print(f"\n로드 완료 {time.time()-_t0:.1f}초")

# %%% CELL 3 [code]
# ============================================================================
# 셀 3 — MET 파싱 (전체 조인일) + non-wear 마스킹 + 0.0 sentinel + class_5min 파싱 + first-35 창 (run-2/4 와 동일 규칙)
# ============================================================================
WINDOW_DAYS = 35
NONWEAR_MET = 0.5          # Oura 는 비착용 분을 MET 0.1 로 채우고 착용 바닥은 0.9 → 포착 99.4% / 오탐 0.17%
WEAR_GATE = 0.8
SED_MET = 1.5              # 정좌 = worn & MET < 1.5 (run 4 와 동일)
ZERO_RUN_MAX = 60          # MET == 0.0 (no-data sentinel) 연속 > 60분인 날은 전체일 프레임에서 제외 (run 4 S1 과 동일)
CLS_LEN = 288
EPOCH_WORN_MIN = 3         # 5분 epoch 이 '착용' 이려면 MET 기준 착용 분 ≥ 3/5
CLS_PARSE_MIN = 0.95       # class_5min 파싱 성공률 하한 (fail-closed; STRICT3 §1: 98.6% 가 288점)

def parse_series(s):
    if not isinstance(s, str):
        return np.empty(0)
    s = s.strip().strip("/")          # ★ 모든 MET 행은 끝에 '/' 가 있다 → 파싱 후 길이로 assert
    if not s:
        return np.empty(0)
    try:
        a = np.fromstring(s, sep="/")
        if a.size:
            return a
    except Exception:
        pass
    try:
        return np.asarray(s.split("/"), dtype=np.float64)
    except Exception:
        return np.empty(0)

def parse_cls(s):
    """class_5min 문자열 → 길이 288 int8 (0..5). 길이 ≠ 288 이거나 파싱 실패면 None (길이 게이트, STRICT3 §9.2)."""
    if not isinstance(s, str):
        return None
    s = s.strip().rstrip("/")
    t = s.split("/")
    if len(t) != CLS_LEN:
        return None
    try:
        a = np.asarray(t, dtype=np.int16)
    except Exception:
        return None
    if a.min() < 0 or a.max() > 5:
        return None
    return a.astype(np.int8)

def _runs(m):
    """bool 벡터에서 True 연속 run 의 길이 배열."""
    p = np.concatenate(([0], m.astype(np.int8), [0])); d = np.diff(p)
    st = np.flatnonzero(d == 1); en = np.flatnonzero(d == -1)
    return en - st

_t0 = time.time()
daily["_rank"] = daily.groupby("sid").cumcount()
_rows = [parse_series(v) for v in daily[MET_COL].to_numpy()]
_len = np.array([r.size for r in _rows])
print(f"MET 파싱 {len(_rows):,}일(전체 조인일) {time.time()-_t0:.1f}초 | 파싱 후 길이 {pd.Series(_len).value_counts().head(3).to_dict()}")
assert (_len >= 1440).all(), "1440점 미만인 날이 있습니다"
MET_ALL = np.stack([r[:1440] for r in _rows]).astype(np.float32); del _rows
WORN_ALL = MET_ALL >= NONWEAR_MET
cov_ALL = WORN_ALL.mean(1)
ZERO_RUN_ALL = np.array([int(_runs(z).max()) if z.any() else 0 for z in (MET_ALL == 0.0)])
OK_ALL = (cov_ALL >= WEAR_GATE) & (ZERO_RUN_ALL <= ZERO_RUN_MAX)     # 전체일 프레임의 유효일 (label-free 위생 규칙, run 4 S1 과 동일)
METW_ALL = np.where(WORN_ALL, MET_ALL, np.nan).astype(np.float32)
EPW_ALL = WORN_ALL.reshape(len(daily), CLS_LEN, 5).sum(2) >= EPOCH_WORN_MIN   # MET 기준 5분 epoch 착용 마스크 (class 스트림과 같은 격자)
print(f"전체 조인일 착용률 median {np.median(cov_ALL):.3f} | 0.0-sentinel run>{ZERO_RUN_MAX}분인 날 {(ZERO_RUN_ALL > ZERO_RUN_MAX).sum()} | 유효일 비율 {OK_ALL.mean():.4f}")

_t0 = time.time()
_cls = [parse_cls(v) for v in daily[CLS_COL].to_numpy()]
CLS_OK_ALL = np.array([c is not None for c in _cls])
CLS_ALL = np.full((len(daily), CLS_LEN), -1, dtype=np.int8)
for i, c in enumerate(_cls):
    if c is not None:
        CLS_ALL[i] = c
del _cls
_dis = np.where(CLS_OK_ALL[:, None], (CLS_ALL >= 1) != EPW_ALL, False).mean(1)     # class-0 vs MET 착용 불일치 epoch 비율 (일별)
print(f"class_5min 파싱 {time.time()-_t0:.1f}초 | 길이 288 인 날 {CLS_OK_ALL.mean():.4f} | class 착용 vs MET 착용 불일치 epoch 비율 median {np.median(_dis[CLS_OK_ALL]):.4f}")
print(f"  상태 분포 {dict(zip(*np.unique(CLS_ALL[CLS_OK_ALL], return_counts=True)))}")
assert CLS_OK_ALL.mean() >= CLS_PARSE_MIN, f"class_5min 파싱률 {CLS_OK_ALL.mean():.3f} < {CLS_PARSE_MIN} — 형식이 가정과 다르다 (fail-closed)"

# first-35 창 (run-2 와 완전히 동일한 규칙: 조인 프레임 피험자별 처음 35행)
W_IDX = np.flatnonzero(daily["_rank"].to_numpy() < WINDOW_DAYS)
W = daily.iloc[W_IDX].reset_index(drop=True)
assert (W.groupby("sid").size() == WINDOW_DAYS).all(), "전원 35일이어야 합니다"
WEAR_OK_ALL = cov_ALL >= WEAR_GATE            # run 4 의 35일 후보용 게이트 (0.0 sentinel 게이트 없음)
print(f"창 적용 {len(W):,}행 = 174 × {WINDOW_DAYS} | MET_ALL {MET_ALL.shape}")

# %%% CELL 4 [code]
# ============================================================================
# 셀 4 — 일별 지표 (전체 조인일): 깨어있는 창 [onset, offset) 의 정좌 bout 구조 (run 4 규칙) + 같은 창 안의 class_5min 전이 엔트로피 (신규)
# ============================================================================
# onset: 착용분만으로 30분 rolling MET 평균 ≥ 1.5 가 처음 성립하는 창의 시작 index (run-2 / Exp 1 / Exp 2 와 동일 규칙)
# offset: 그 조건이 마지막으로 성립하는 창의 끝(배타) index. 창 길이 < 240분이면 그 날은 무효(NaN).
# onset == 0 은 04:00 에 이미 활동 중(좌측 검열)이다 — 창 정의(longest_w, D1 재현)에는 그대로 두고, onset_SD_all 에서만 제외한다(cell 5).
_k = 30
_c = np.cumsum(np.nan_to_num(METW_ALL), 1); _n = np.cumsum(WORN_ALL, 1)
_roll = (_c[:, _k:] - _c[:, :-_k]) / np.maximum(_n[:, _k:] - _n[:, :-_k], 1)
_hit = _roll >= 1.5
_any = _hit.any(1)
onset_i = np.where(_any, np.argmax(_hit, 1), -1)
offset_i = np.where(_any, _hit.shape[1] - 1 - np.argmax(_hit[:, ::-1], 1) + _k, -1)
del _c, _n, _roll, _hit
MIN_WINDOW = 240

SED = WORN_ALL & (MET_ALL < SED_MET)     # 정좌 분: 착용 중이고 MET < 1.5. 비착용 분은 False → run 이 끊긴다.
_out = np.full((len(daily), 5), np.nan)  # win, sed_w, longest_w, meanb_w, ntr_w
for i in range(len(daily)):
    o, f = int(onset_i[i]), int(offset_i[i])
    if o < 0 or f - o < MIN_WINDOW:
        continue
    m = SED[i, o:f]; L = _runs(m)
    _out[i] = (f - o, m.sum(), (L.max() if L.size else 0.0), (L.mean() if L.size else np.nan), L.size)
win, sed_w, longest_w, meanb_w, ntr_w = _out.T
onset = np.where(onset_i >= 0, onset_i.astype(float), np.nan)
del SED

MIN_WORN_EPOCH = 12      # 창 안 유효 epoch(1시간) 미만이면 NaN
MIN_TRANS = 6            # 유효 epoch 사이 전이 미만이면 NaN
def cls_entropy(c, epw, o, f):
    """깨어있는 창 [o, f)(분) 에 해당하는 5분 epoch 들에서, 유효 epoch = (MET 기준 착용 ≥3/5분) and (class ≥ 1) 사이의 전이만 세어
    5상태(1..5) 전이행렬의 점유율 가중 엔트로피(bit)를 돌려준다. 비착용/불일치 epoch 는 run 을 끊는다. 한 상태만 있는 날은 H = 0."""
    e0, e1 = o // 5, int(math.ceil(f / 5))
    cc = c[e0:e1]; ww = epw[e0:e1] & (cc >= 1)
    if ww.sum() < MIN_WORN_EPOCH:
        return np.nan
    a, b = cc[:-1], cc[1:]
    m = ww[:-1] & ww[1:]
    if m.sum() < MIN_TRANS:
        return np.nan
    N = np.zeros((5, 5)); np.add.at(N, (a[m].astype(int) - 1, b[m].astype(int) - 1), 1.0)
    rows = N.sum(1); pi = rows / rows.sum()
    H = 0.0
    for i in range(5):
        if rows[i] <= 0:
            continue
        p = N[i] / rows[i]; p = p[p > 0]
        H += pi[i] * float(-(p * np.log2(p)).sum())
    return H

_t0 = time.time()
clsH = np.full(len(daily), np.nan)
for i in range(len(daily)):
    o, f = int(onset_i[i]), int(offset_i[i])
    if CLS_OK_ALL[i] and o >= 0 and f - o >= MIN_WINDOW:
        clsH[i] = cls_entropy(CLS_ALL[i], EPW_ALL[i], o, f)
print(f"class 전이 엔트로피(깨어있는 창) {time.time()-_t0:.1f}초 | 유한 비율 {np.isfinite(clsH).mean():.4f} | median {np.nanmedian(clsH):.3f}")

DA = pd.DataFrame({
    "sid": daily["sid"].to_numpy(), "date": daily["date"].to_numpy(), "day_idx": daily["_rank"].to_numpy(int),
    "low": daily["activity_low"].to_numpy(float), "medium": daily["activity_medium"].to_numpy(float),
    "nonwear": daily["activity_non_wear"].to_numpy(float),
    "cov": cov_ALL, "wear_ok": WEAR_OK_ALL, "ok_all": OK_ALL, "cls_ok": CLS_OK_ALL, "cls_dis": _dis,
    "onset": onset, "win": win, "sed_w": sed_w, "longest_w": longest_w, "meanb_w": meanb_w, "ntr_w": ntr_w, "clsH": clsH,
})
DA["valid35"] = DA["wear_ok"] & np.isfinite(DA["longest_w"])      # run 4 의 35일 후보 유효일 규칙 (D1 재현용)
DA["valid_all"] = DA["ok_all"] & np.isfinite(DA["longest_w"])     # 전체일 창 유효일 (창 확정·≥240분·착용률≥0.8·sentinel≤60)
DM = DA[DA["day_idx"] < WINDOW_DAYS].reset_index(drop=True)
assert len(DM) == 174 * WINDOW_DAYS
assert (DM["sid"].to_numpy() == W["sid"].to_numpy()).all() and (DM["date"].to_numpy() == W["date"].to_numpy()).all(), "DM 행이 첫 35일 창 W 와 다르다"
print("전체 조인일:")
print(f"  ok_all {DA['ok_all'].mean():.4f} | valid_all {DA['valid_all'].mean():.4f} | 피험자별 valid_all min {int(DA.groupby('sid')['valid_all'].sum().min())} med {int(DA.groupby('sid')['valid_all'].sum().median())}")
print(f"  창 길이 median {np.nanmedian(win):.0f}분 | 최장 정좌 bout median {np.nanmedian(longest_w):.0f}분 | onset median {np.nanmedian(onset):.0f}분(04:00 기준) | onset==0 검열 {float(np.nanmean(onset == 0)):.4f}")

# %%% CELL 5 [code]
# ============================================================================
# 셀 5 — 피험자 특징: 동결 B0 (35일) + 전체일 A1/A2 + 후보 라이브러리 4개 + 교란 변수 (전부 label-free, 피험자 내부에서만 계산)
# ============================================================================
def _fin(x):
    x = np.asarray(x, float); return x[np.isfinite(x)]

def ST_SD(x):
    x = _fin(x); return float(np.std(x, ddof=0)) if x.size >= 3 else np.nan

def ST_MED(x):
    x = _fin(x); return float(np.median(x)) if x.size else np.nan

DEM_BLOCK = ["low_SD", "low_MED"]                                   # B0 (동결, 35일)
RUN4_P_FEATURES = ["low_SD", "low_MED", "longest_w_SD"]             # D1: run 4 P 재현 전용
A1_FEATURES = ["low_SD_all", "low_MED_all"]                         # A1 = run 4 S1 정의 그대로
A2_FEATURES = ["low_SD_all", "low_MED_all", "longest_w_SD_all"]     # A2 MAXFIX
LIBRARY = ["longest_w_SD_all", "onset_SD_all", "clsH_SD_all", "low_lag1_all"]   # A3 후보 라이브러리 (사전등록, 순서 고정; steps_SD 는 라벨 노출로 제외)
MIN_DAYS_DETREND = 8       # A1 detrend 최소 일수 (run 4 S1)
MIN_PAIRS_LAG1 = 10        # lag1 최소 인접일 쌍
NAN_MAX_LIBRARY = 0.10     # 라이브러리 특징 결측 비율 상한 (fail-closed)

def _lag1(dates, x):
    """연속한 달력일 쌍 (t, t+1) 의 Pearson 상관 (일간 규칙성). 쌍 < 10 이거나 분산 0 이면 NaN. (값, 쌍 수) 반환."""
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    x = np.asarray(x, float)
    o = np.argsort(d); d = d[o]; x = x[o]
    gap = (d[1:] - d[:-1]) / np.timedelta64(1, "D")
    m = (gap == 1) & np.isfinite(x[1:]) & np.isfinite(x[:-1])
    n = int(m.sum())
    if n < MIN_PAIRS_LAG1:
        return np.nan, n
    a, b = x[:-1][m], x[1:][m]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan, n
    return float(np.corrcoef(a, b)[0, 1]), n

def build_features(dm, da, sids):
    rows = []
    sid_arr = dm["sid"].to_numpy(); sid_all = da["sid"].to_numpy()
    for s in sids:
        g = dm.iloc[np.flatnonzero(sid_arr == s)]
        r = {}
        # 동결 D블록 (run-2 와 비트 동일: 35일 전부, 게이트 없음)
        r["low_SD"] = ST_SD(g["low"]); r["low_MED"] = ST_MED(g["low"])
        # run 4 P 후보 (35일, valid35) — D1 재현 전용
        v = g[g["valid35"].to_numpy(bool)]
        r["longest_w_SD"] = ST_SD(v["longest_w"])
        # ── 전체 조인일 ────────────────────────────────────────────────
        ga0 = da.iloc[np.flatnonzero(sid_all == s)]                      # 조인일 전부 (게이트 전)
        ga = ga0[ga0["ok_all"].to_numpy(bool)]
        lo = ga["low"].to_numpy(float)
        # A1: run 4 S1 정의 그대로 — 유효일, 날짜축 선형 detrend 후 SD / 원값 중앙값
        if lo.size >= MIN_DAYS_DETREND:
            t = (pd.to_datetime(ga["date"]) - pd.to_datetime(ga["date"]).min()).dt.days.to_numpy(float)
            b = np.polyfit(t, lo, 1); res = lo - np.polyval(b, t)
            r["low_SD_all"] = ST_SD(res); r["low_MED_all"] = ST_MED(lo)
        else:
            res = None
            r["low_SD_all"] = r["low_MED_all"] = np.nan
        # 후보 라이브러리 (전부 across-day 통계)
        gw = ga[ga["valid_all"].to_numpy(bool)]                          # 창 유효일
        r["longest_w_SD_all"] = ST_SD(gw["longest_w"])                   # 최장 정좌 bout 의 일간 SD (run 4 후보, 전체일)
        _on = gw["onset"].to_numpy(float)
        r["onset_SD_all"] = ST_SD(_on[_on > 0])                          # 활동 개시 시각(04:00 기준 분) 의 일간 SD — 좌측 검열(onset==0) 일 제외, wrap 없음
        r["clsH_SD_all"] = ST_SD(gw["clsH"])                             # 깨어있는 창 안 class_5min 전이 엔트로피의 일간 SD (창 유효 & 길이 288 일)
        if res is not None:
            r["low_lag1_all"], _np = _lag1(ga["date"].to_numpy(), res)   # 인접일 자기상관 — A1 과 같은 detrend 잔차에서 (일간 규칙성)
        else:
            r["low_lag1_all"], _np = np.nan, 0
        # 기술·교란 변수 (모델 입력 아님)
        r["cf_ndays_all"] = float(lo.size); r["cf_nvalid_all"] = float(len(gw)); r["cf_npairs_all"] = float(_np)
        r["cf_nclsok_all"] = float(np.isfinite(gw["clsH"]).sum()); r["cf_nonset_pos_all"] = float((_on > 0).sum())
        r["cf_nok_dropped_all"] = float(len(ga0) - len(ga))
        r["cf_nonwear_mean_all"] = float(_fin(ga["nonwear"]).mean()) if lo.size else np.nan
        _d0 = pd.to_datetime(ga0["date"])
        r["cf_gap_days_all"] = float((_d0.max() - _d0.min()).days + 1 - len(ga0))     # 달력 결측일 (게이트 전 조인일 기준)
        r["cf_start_month"] = float(_d0.min().month + 12 * (_d0.min().year - 2020))
        r["cf_onset0_frac_all"] = float(np.mean(_on == 0)) if _on.size else np.nan
        r["cf_clsdis_all"] = float(_fin(ga["cls_dis"][ga["cls_ok"].to_numpy(bool)]).mean()) if ga["cls_ok"].any() else np.nan
        r["dx_volume_all"] = ST_MED(ga["low"] + ga["medium"])
        r["dx_onset_MED_all"] = ST_MED(_on[_on > 0]); r["dx_longest_MED_all"] = ST_MED(gw["longest_w"])
        r["dx_clsH_MED_all"] = ST_MED(gw["clsH"]); r["dx_win_MED_all"] = ST_MED(gw["win"])
        rows.append(r)
    return pd.DataFrame(rows, index=pd.Index(sids, name="sid"))

_t0 = time.time()
FEAT_ALL = build_features(DM, DA, SIDS)
ALL_MODEL_COLS = DEM_BLOCK + RUN4_P_FEATURES[2:] + A1_FEATURES + LIBRARY
print(f"특징 {FEAT_ALL.shape} ({time.time()-_t0:.1f}초)")
assert_no_forbidden(ALL_MODEL_COLS, "모델 입력 후보")
assert FEAT_ALL.shape[0] == 174
_nan = FEAT_ALL[ALL_MODEL_COLS].isna().mean()
print("\n결측 비율:"); print(_nan.round(4).to_string())
assert (_nan[LIBRARY] <= NAN_MAX_LIBRARY).all(), f"라이브러리 특징 결측 > {NAN_MAX_LIBRARY} — 정의가 데이터에 맞지 않는다 (fail-closed)"
print(f"전체일수 cf_ndays_all: min {FEAT_ALL['cf_ndays_all'].min():.0f} med {FEAT_ALL['cf_ndays_all'].median():.0f} | 창 유효일 cf_nvalid_all: min {FEAT_ALL['cf_nvalid_all'].min():.0f} med {FEAT_ALL['cf_nvalid_all'].median():.0f}"
      f" | lag1 쌍 수 min {FEAT_ALL['cf_npairs_all'].min():.0f} med {FEAT_ALL['cf_npairs_all'].median():.0f} | onset>0 일수 min {FEAT_ALL['cf_nonset_pos_all'].min():.0f}")

# ── split-half 신뢰도 (label-free, 게이트 아님, 검정력 해석용) ─────────────
from scipy import stats as sps
def _split_half(cols, mode="parity"):
    if mode == "parity":
        A = build_features(DM[DM["day_idx"] % 2 == 0].reset_index(drop=True), DA[DA["day_idx"] % 2 == 0].reset_index(drop=True), SIDS)
        B = build_features(DM[DM["day_idx"] % 2 == 1].reset_index(drop=True), DA[DA["day_idx"] % 2 == 1].reset_index(drop=True), SIDS)
    else:   # 전반/후반 시간 분할 (인접일 쌍이 필요한 lag1 용; parity 값과 비교 불가)
        _mid = DA.groupby("sid")["day_idx"].transform("max") // 2
        A = build_features(DM, DA[DA["day_idx"] <= _mid].reset_index(drop=True), SIDS)
        B = build_features(DM, DA[DA["day_idx"] > _mid].reset_index(drop=True), SIDS)
    out = {}
    for c in cols:
        a, b = A[c].to_numpy(float), B[c].to_numpy(float); m = np.isfinite(a) & np.isfinite(b)
        rho = float(sps.spearmanr(a[m], b[m])[0]) if m.sum() >= 10 else float("nan")
        sb = 2 * rho / (1 + rho) if np.isfinite(rho) and rho > -1 else float("nan")
        out[c] = {"split_half_rho": round(rho, 3), "spearman_brown": round(sb, 3), "n": int(m.sum()), "mode": mode}
    return out
RELIAB = _split_half(["low_SD", "low_SD_all", "longest_w_SD_all", "onset_SD_all", "clsH_SD_all"], "parity")
RELIAB.update(_split_half(["low_lag1_all"], "halves"))
print("\nsplit-half 신뢰도 (Spearman → Spearman-Brown):")
for k, v in RELIAB.items():
    print(f"  {k:18s} rho {v['split_half_rho']:+.3f}  SB {v['spearman_brown']:+.3f}  (n={v['n']}, {v['mode']})")

# %%% CELL 6 [code]
# ============================================================================
# 셀 6 — 누수 자기검증 (부분집합 재구축 비트 동일) + 특징 지문 계약 + run 4 feature_matrix 와의 계약 (CT-F)
# ============================================================================
_rng = np.random.default_rng(SEED)
_sub = sorted(_rng.choice(SIDS, size=90, replace=False).tolist())
_F = build_features(DM[DM["sid"].isin(_sub)].reset_index(drop=True), DA[DA["sid"].isin(_sub)].reset_index(drop=True), _sub)
_a = _F.to_numpy(float); _b = FEAT_ALL.loc[_sub, _F.columns].to_numpy(float)
_same = ((_a == _b) | (np.isnan(_a) & np.isnan(_b))).all()
print(f"부분집합 재구축 비트 동일성 (90명, {_F.shape[1]}개 컬럼 전부): {_same}")
assert _same, "[누수] 특징이 다른 피험자에 의존합니다"
# 이 가드는 피험자 집계(cell 5)만 검사한다. 일별 규칙(cell 3–4)은 고정 상수(NONWEAR_MET, WEAR_GATE, ZERO_RUN_MAX, MIN_WINDOW, CLS_LEN, EPOCH_WORN_MIN, MIN_WORN_EPOCH, MIN_TRANS)와 행 단위 연산뿐이다.

def fingerprint(df):
    return hashlib.sha256(np.ascontiguousarray(np.round(np.nan_to_num(df.to_numpy(float), nan=-9999.0), 9)).tobytes()
                          + "|".join(df.columns).encode()).hexdigest()[:16]
FP = {"B0": fingerprint(FEAT_ALL[DEM_BLOCK]), "A1": fingerprint(FEAT_ALL[A1_FEATURES]),
      "A2": fingerprint(FEAT_ALL[A2_FEATURES]), "A3lib": fingerprint(FEAT_ALL[A1_FEATURES + LIBRARY])}
print(f"특징 지문 (소수 9자리 반올림, 환경 무관)  {FP}")
RUN2_B0_FP = "3bc37d120081437d"
CT_F = {"b0_fingerprint": FP["B0"], "b0_fingerprint_expected": RUN2_B0_FP, "b0_fingerprint_match": bool(FP["B0"] == RUN2_B0_FP)}
if not SYNTHETIC:
    assert CT_F["b0_fingerprint_match"], "B0 특징 지문 불일치 — 조인/파싱/정의가 바뀐 것"
    print("B0 특징 지문 계약 ✓")

# run 4 feature_matrix.csv 와 공유 컬럼 5개 계약 (CSV 왕복 허용오차 1e-9) — 실데이터에서만. 경로: 환경변수 > 로컬 저장소 관례
_env_fm = os.environ.get("EXP3_RUN4_FM")
_fm_cands = ([Path(_env_fm)] if _env_fm else []) + [
    DATA_ROOT.parent / "SangHyo" / "Binary" / "Binary_SedentaryBout_Nested" / "Binary_SedentaryBout_Nested_result" / "20260904_041546_utc" / "feature_matrix.csv"]
_run4_fm = next((p for p in _fm_cands if p.is_file()), None)
print(f"run 4 feature_matrix: {_run4_fm}")
if SYNTHETIC:
    CT_F.update({"run4_feature_matrix": str(_run4_fm) if _run4_fm else None, "pass": "SKIPPED", "note": "SYNTHETIC 모드 → 비교 생략 (경로 해석만 실행)"})
    print("CT-F 건너뜀 (SYNTHETIC)")
elif _run4_fm is None:
    CT_F.update({"run4_feature_matrix": None, "pass": "SKIPPED", "note": "run 4 feature_matrix.csv 없음 (EXP3_RUN4_FM 미설정) — cell 9/10 의 per-repeat 재현 assert 가 대신 보호"})
    print("⚠⚠ CT-F 건너뜀: run 4 feature_matrix.csv 를 찾지 못함 — 실데이터 로컬 실행이면 EXP3_RUN4_FM 을 지정하라")
else:
    _fm = pd.read_csv(_run4_fm, index_col=0).loc[SIDS]
    _shared = ["low_SD", "low_MED", "longest_w_SD", "low_SD_all", "low_MED_all"]
    _diff = {c: float(np.nanmax(np.abs(_fm[c].to_numpy(float) - FEAT_ALL[c].to_numpy(float)))) for c in _shared}
    _nanm = {c: bool((np.isnan(_fm[c].to_numpy(float)) == np.isnan(FEAT_ALL[c].to_numpy(float))).all()) for c in _shared}
    CT_F.update({"run4_feature_matrix": str(_run4_fm), "max_abs_diff": _diff, "nan_pattern_match": _nanm,
                 "pass": bool(max(_diff.values()) <= 1e-9 and all(_nanm.values()))})
    print(f"CT-F run 4 feature_matrix 계약: max|diff| {_diff} → {'✓' if CT_F['pass'] else '✗'}")
    assert CT_F["pass"] is True, "run 4 와 공유 특징이 일치하지 않는다 — 조인/파싱/정의가 바뀐 것"

# %%% CELL 7 [code]
# ============================================================================
# 셀 7 — fold 생성기 (run-2 와 동일 seed) + 전 repeat 구성 assert + label-free 감사 (수집·배치 + 수준·창 구조)
# ============================================================================
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

N_OUTER, N_REPEATS = 5, (2 if QUICK else 20)
N_INNER, N_INNER_REP = 4, 2
C_GRID = [0.003, 0.01, 0.03, 0.1]
TOL = 0.005

def outer_folds(y3v, n_repeats=N_REPEATS, n_splits=N_OUTER):
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=100 + r)
        for f, (tr, te) in enumerate(skf.split(np.zeros(len(y3v)), y3v)):
            yield r, f, tr, te

_min_dem, _min_mci = 99, 99
for r, f, tr, te in outer_folds(y3, N_REPEATS):
    assert len(np.intersect1d(tr, te)) == 0 and len(tr) + len(te) == len(y3), "outer train/test 겹침"
    bc = np.bincount(y3[te], minlength=3); _min_dem = min(_min_dem, bc[2]); _min_mci = min(_min_mci, bc[1])
print(f"{N_REPEATS} repeat × {N_OUTER} fold 전수 검사: test Dem min {_min_dem}, MCI min {_min_mci}")
assert _min_dem >= 2 and _min_mci >= 8

def sub_aucs(scores, y3v):
    m = roc_auc_score((y3v > 0).astype(int), scores)
    k = np.isin(y3v, [0, 1]); a1 = roc_auc_score((y3v[k] > 0).astype(int), scores[k])
    k = np.isin(y3v, [0, 2]); a2 = roc_auc_score((y3v[k] > 0).astype(int), scores[k])
    n1, n2 = int((y3v == 1).sum()), int((y3v == 2).sum())
    return m, a1, a2, (n1 * a1 + n2 * a2) / (n1 + n2)

def hanley_se(auc, n_pos, n_neg):
    q1 = auc / (2 - auc); q2 = 2 * auc ** 2 / (1 + auc)
    v = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return float(np.sqrt(max(v, 0)))

# ── 독립성 (label-free) ─────────────────────────────────────────────────
_cols = DEM_BLOCK + A1_FEATURES + LIBRARY
print("\n" + "=" * 78); print("특징 상관행렬 (Spearman, label-free)"); print("=" * 78)
CORR = FEAT_ALL[_cols].corr(method="spearman")
print(CORR.round(3).to_string())

# ── 교란 감사 (label-free): 수집량/배치 + 수준·창 구조 ────────────────────
print("\n" + "=" * 78); print("교란 감사 — 특징 vs 수집량·배치 (A) 및 수준·창 구조 (B) (label-free Spearman)"); print("=" * 78)
_cfA = {"ndays_all": FEAT_ALL["cf_ndays_all"].to_numpy(), "gap_days": FEAT_ALL["cf_gap_days_all"].to_numpy(),
        "nok_dropped": FEAT_ALL["cf_nok_dropped_all"].to_numpy(), "start_month": FEAT_ALL["cf_start_month"].to_numpy(),
        "nonwear_mean": FEAT_ALL["cf_nonwear_mean_all"].to_numpy(), "nvalid": FEAT_ALL["cf_nvalid_all"].to_numpy(),
        "nclsok": FEAT_ALL["cf_nclsok_all"].to_numpy(), "npairs": FEAT_ALL["cf_npairs_all"].to_numpy(),
        "onset0_frac": FEAT_ALL["cf_onset0_frac_all"].to_numpy(), "clsdis": FEAT_ALL["cf_clsdis_all"].to_numpy()}
_cfB = {"low_MED_all": FEAT_ALL["low_MED_all"].to_numpy(), "volume": FEAT_ALL["dx_volume_all"].to_numpy(),
        "win_MED": FEAT_ALL["dx_win_MED_all"].to_numpy(), "onset_MED": FEAT_ALL["dx_onset_MED_all"].to_numpy(),
        "longest_MED": FEAT_ALL["dx_longest_MED_all"].to_numpy(), "clsH_MED": FEAT_ALL["dx_clsH_MED_all"].to_numpy()}
def _rho(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or np.std(b[m]) < 1e-12 or np.std(a[m]) < 1e-12:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(sps.spearmanr(a[m], b[m])[0])
rows = []
for f in _cols:
    v = FEAT_ALL[f].to_numpy()
    rows.append({"특징": f, **{k: round(_rho(v, c), 3) for k, c in _cfA.items()}, **{k: round(_rho(v, c), 3) for k, c in _cfB.items()}})
CONF = pd.DataFrame(rows)
print(CONF.to_string(index=False))
_flagA = [r["특징"] for _, r in CONF.iterrows() if max((abs(r[k]) for k in _cfA if np.isfinite(r[k])), default=0.0) > 0.30]
_flagB = [r["특징"] for _, r in CONF.iterrows() if r["특징"] in LIBRARY and max((abs(r[k]) for k in _cfB if np.isfinite(r[k])), default=0.0) > 0.50]
print(f"\n(A) 수집량·배치 |ρ| > 0.30 인 특징: {_flagA if _flagA else '없음'}")
print(f"(B) 후보 중 수준·창 구조 |ρ| > 0.50 인 특징 (수준 프록시 의심, 기록용): {_flagB if _flagB else '없음'}")
print(f"Hanley–McNeil SE — merged(63 vs 111) @0.66 = {hanley_se(0.66,63,111):.4f}, @0.70 = {hanley_se(0.70,63,111):.4f}")

# %%% CELL 8 [code]
# ============================================================================
# 셀 8 — nested CV 엔진 (run-2 / Exp 1 / Exp 2 와 동일 추정기·C 선택 규칙) + A3 용 fold 내부 특징집합 선택 (신규)
# ============================================================================
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

def _probe_mc():   # sklearn ≥1.7 은 multi_class 인자를 제거했다 → 실제 fit 으로 probe
    rs = np.random.RandomState(0)
    try:
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=100,
                           multi_class="multinomial").fit(rs.randn(30, 3), np.array([0, 1, 2] * 10))
        return True
    except Exception:
        return False
USE_MC_KW = _probe_mc()
_LR_KW = {"multi_class": "multinomial"} if USE_MC_KW else {}
print(f"multi_class 인자: {'명시 지정' if USE_MC_KW else '미지원 → lbfgs 기본이 multinomial (동일 결과)'}")

def make_lr(C):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=C, penalty="l2", solver="lbfgs",
                                               max_iter=5000, **_LR_KW))])

def arm_mnl(Xtr, y3tr, Xte, C):
    """3-class multinomial → 1 − P(CN). 모든 전처리는 Pipeline 안에서 train 에만 fit."""
    p = make_lr(C).fit(Xtr, y3tr)
    cls = list(p.named_steps["lr"].classes_)
    return 1.0 - p.predict_proba(Xte)[:, cls.index(0)]

def make_arm(fn, feat_names, Cs):
    """고정 arm 등록: 특징 컬럼명에 금지어 가드를 적용하고 행렬을 만든다."""
    assert_no_forbidden(feat_names, "arm 특징")
    X = FEAT_ALL[list(feat_names)].to_numpy(float)
    return (fn, X, list(Cs), {"features": list(feat_names), "sets": None})

def make_sel_arm(fn, base_names, cand_names, Cs, X=None):
    """선택 arm 등록: 특징집합 = base ∪ {∅ 또는 후보 1개}. 어느 집합·C 를 쓸지는 outer-train 의 inner CV 만 본다."""
    assert_no_forbidden(list(base_names) + list(cand_names), "선택 arm 특징")
    cols = list(base_names) + list(cand_names)
    if X is None:
        X = FEAT_ALL[cols].to_numpy(float)
    nb = len(base_names)
    sets = [tuple(range(nb))] + [tuple(range(nb)) + (nb + j,) for j in range(len(cand_names))]
    set_names = ["base"] + [f"base+{c}" for c in cand_names]
    return (fn, X, list(Cs), {"features": cols, "base": list(base_names), "candidates": list(cand_names),
                              "sets": sets, "set_names": set_names})

def pick_C(fn, Xtr, y3tr, Cs, n_inner, n_inner_rep, seed, tol=TOL):
    if len(Cs) == 1:
        return Cs[0], {Cs[0]: float("nan")}, 0
    yb = (y3tr > 0).astype(int); acc = {C: [] for C in Cs}; dead = 0
    for rr in range(n_inner_rep):
        skf = StratifiedKFold(n_inner, shuffle=True, random_state=seed + rr)
        for tr, va in skf.split(np.zeros(len(y3tr)), y3tr):
            if len(np.unique(yb[va])) < 2:
                continue
            for C in Cs:
                try:
                    acc[C].append(roc_auc_score(yb[va], fn(Xtr[tr], y3tr[tr], Xtr[va], C)))
                except Exception:
                    acc[C].append(0.5); dead += 1
    mean = {C: (float(np.mean(v)) if v else 0.5) for C, v in acc.items()}
    best = max(mean.values())
    for C in sorted(Cs):                      # tolerance 안에서 가장 작은 C (정규화 최강)
        if mean[C] >= best - tol:
            return C, mean, dead
    return max(mean, key=mean.get), mean, dead

def pick_config(fn, Xtr, y3tr, sets, Cs, n_inner, n_inner_rep, seed, tol=TOL):
    """(특징집합, C) 를 inner CV 평균 AUC 로 고른다. best − tol 안의 후보 중 특징 수 최소 → C 최소 → 집합 index 최소 (결정적)."""
    yb = (y3tr > 0).astype(int); acc = {(j, C): [] for j in range(len(sets)) for C in Cs}; dead = 0
    for rr in range(n_inner_rep):
        skf = StratifiedKFold(n_inner, shuffle=True, random_state=seed + rr)
        for tr, va in skf.split(np.zeros(len(y3tr)), y3tr):
            if len(np.unique(yb[va])) < 2:
                continue
            for j, cols in enumerate(sets):
                cols = list(cols)
                for C in Cs:
                    try:
                        acc[(j, C)].append(roc_auc_score(yb[va], fn(Xtr[tr][:, cols], y3tr[tr], Xtr[va][:, cols], C)))
                    except Exception:
                        acc[(j, C)].append(0.5); dead += 1
    mean = {k: (float(np.mean(v)) if v else 0.5) for k, v in acc.items()}
    best = max(mean.values())
    cands = [k for k in mean if mean[k] >= best - tol]
    cands.sort(key=lambda k: (len(sets[k[0]]), k[1], k[0]))
    return cands[0], mean, dead

def run_arms(arms, y3v, n_repeats=N_REPEATS, n_inner=N_INNER,
             n_inner_rep=N_INNER_REP, verbose=True, tag=""):
    """arms: {name: make_arm(...) | make_sel_arm(...)}. 반환: oof[name] (n_repeats × n), chosen, dead, fold_rows, inner_dead."""
    n = len(y3v); yb = (y3v > 0).astype(int)
    oof = {k: np.full((n_repeats, n), np.nan) for k in arms}
    chosen = {k: [] for k in arms}; dead = {k: 0 for k in arms}; inner_dead = {k: 0 for k in arms}
    fold_rows = []
    total = n_repeats * N_OUTER; t0 = time.time(); done = 0
    for r, f, tr, te in outer_folds(y3v, n_repeats):
        msg = []
        for k, (fn, Xm, Cs, meta) in arms.items():
            sets = meta.get("sets"); idead = 0
            try:
                if sets is None:
                    C, _, idead = pick_C(fn, Xm[tr], y3v[tr], Cs, n_inner, n_inner_rep, seed=7000 + 37 * r + f)
                    s = np.asarray(fn(Xm[tr], y3v[tr], Xm[te], C), float); pick = C
                else:
                    (j, C), _, idead = pick_config(fn, Xm[tr], y3v[tr], sets, Cs, n_inner, n_inner_rep, seed=7000 + 37 * r + f)
                    cols = list(sets[j])
                    s = np.asarray(fn(Xm[tr][:, cols], y3v[tr], Xm[te][:, cols], C), float); pick = (meta["set_names"][j], C)
                inner_dead[k] += idead
                if s.shape != (len(te),) or not np.isfinite(s).all():
                    raise ValueError("비정상 점수")
            except Exception as exc:
                dead[k] += 1; s = np.full(len(te), 0.5)
                pick = float("nan") if sets is None else ("failed", float("nan"))
                if dead[k] <= 3:
                    print(f"    [{k}] fold r{r}f{f} 실패 → 0.5 fallback: {type(exc).__name__}: {exc}")
            oof[k][r, te] = s; chosen[k].append(pick)
            try:
                a = roc_auc_score(yb[te], s)
            except Exception:
                a = float("nan")
            fold_rows.append({"arm": k, "repeat": r, "fold": f, "n_test": int(len(te)), "inner_dead": int(idead),
                              "pick": (pick if not isinstance(pick, tuple) else f"{pick[0]}|C={pick[1]}"), "auc": a})
            msg.append(f"{k.split(' ')[0]} auc={a:.3f}")
        done += 1
        if verbose and (done % 5 == 0 or done == total):
            el = time.time() - t0
            print(f"  [{tag}{done}/{total}] " + " | ".join(msg) + f" | {el/60:.1f}분, ETA {el/done*(total-done)/60:.1f}분")
            progress(tag or "run", done, total, eta_min=round(el / done * (total - done) / 60, 2))
    return oof, chosen, dead, fold_rows, inner_dead

def summarize(oof_arm, y3v):
    """threshold-free 지표만 보고한다."""
    yb = (y3v > 0).astype(int)
    per = []
    for r in range(oof_arm.shape[0]):
        try:
            per.append(roc_auc_score(yb, oof_arm[r]))
        except Exception:
            per.append(float("nan"))
    sc = np.nanmean(oof_arm, axis=0)
    m, a1, a2, ident = sub_aucs(sc, y3v)
    return dict(repeat_mean=float(np.nanmean(per)), repeat_sd=float(np.nanstd(per, ddof=0)),
                subject_mean_auc=float(m), auc_cn_vs_mci=float(a1), auc_cn_vs_dem=float(a2),
                identity_gap=float(abs(m - ident)),
                hanley_se=hanley_se(float(m), int(yb.sum()), int((1 - yb).sum())),
                pr_auc=float(average_precision_score(yb, sc)),
                per_repeat=[float(v) for v in per], scores=sc)

N_BOOT = 300 if QUICK else 5000
def boot_ci(scores, yb, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed); idx = np.arange(len(yb)); v = []
    for _ in range(n):
        b = rng.choice(idx, idx.size, replace=True)
        if len(np.unique(yb[b])) < 2:
            continue
        v.append(roc_auc_score(yb[b], scores[b]))
    v = np.asarray(v)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), float(np.mean(v <= 0.70))

def paired_boot(sa, sb, yb, n=N_BOOT, seed=SEED + 1):
    """Δ = AUC(sa) − AUC(sb): (관측 Δ, 2.5%, 97.5%, bootstrap 평균 Δ) — 점추정은 관측 차이, CI 는 percentile."""
    rng = np.random.default_rng(seed); idx = np.arange(len(yb)); d = []
    for _ in range(n):
        b = rng.choice(idx, idx.size, replace=True)
        if len(np.unique(yb[b])) < 2:
            continue
        d.append(roc_auc_score(yb[b], sa[b]) - roc_auc_score(yb[b], sb[b]))
    obs = float(roc_auc_score(yb, sa) - roc_auc_score(yb, sb))
    return obs, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(np.mean(d))

def delta_block(sP, sB, y3v):
    """Δ = P − B (merged / CN-vs-MCI / CN-vs-Dem), paired subject bootstrap."""
    yb = (y3v > 0).astype(int); out = {}
    out["merged"] = paired_boot(sP, sB, yb)
    k = np.isin(y3v, [0, 1]); out["cn_vs_mci"] = paired_boot(sP[k], sB[k], yb[k])
    k = np.isin(y3v, [0, 2]); out["cn_vs_dem"] = paired_boot(sP[k], sB[k], yb[k])
    return out

def winner_bias_boot(cand_scores, base_scores, yb, n=N_BOOT, seed=SEED + 2):
    """max-over-arms 승자 편향: 같은 subject bootstrap 재표본에서 Δ_j^(b) (후보 j 대 B0) 를 전부 계산,
    bias = mean_b[max_j Δ_j^(b)] − max_j Δ_j(관측). 실제 arm 간 의존 구조를 그대로 쓴다."""
    rng = np.random.default_rng(seed); idx = np.arange(len(yb)); keys = list(cand_scores); mx = []
    obs = {k: roc_auc_score(yb, cand_scores[k]) - roc_auc_score(yb, base_scores) for k in keys}
    for _ in range(n):
        b = rng.choice(idx, idx.size, replace=True)
        if len(np.unique(yb[b])) < 2:
            continue
        ab = roc_auc_score(yb[b], base_scores[b])
        mx.append(max(roc_auc_score(yb[b], cand_scores[k][b]) - ab for k in keys))
    return float(np.mean(mx) - max(obs.values())), obs

def holm(pvals):
    """Holm step-down 보정 (dict name → p)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); adj = {}; running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, (m - i) * p); adj[k] = float(min(1.0, running))
    return adj

def clopper_pearson(k, n, alpha=0.05):
    """permutation p 의 Monte-Carlo 95% 구간 (k = null ≥ obs 횟수, n = 순열 수)."""
    lo = float(sps.beta.ppf(alpha / 2, k, n - k + 1)) if k > 0 else 0.0
    hi = float(sps.beta.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0
    return lo, hi

print("엔진 준비 완료.")

# %%% CELL 9 [code]
# ============================================================================
# 셀 9 — MAIN: K=4 (B0, A1, A2, A3) → B0·A1 재현 assert → CI → Δ vs B0 → 승자 편향 → A3 선택 빈도
# ============================================================================
BASELINE = "B0 (동결 D블록, 35일)"
ARM_A1, ARM_A2, ARM_A3 = "A1 B0_all (전체일)", "A2 MAXFIX (B0_all + longest_w_SD_all)", "A3 MAXSEL (B0_all + inner 선택 ≤1)"
MAIN_ARMS = {BASELINE: make_arm(arm_mnl, DEM_BLOCK, C_GRID),
             ARM_A1: make_arm(arm_mnl, A1_FEATURES, C_GRID),
             ARM_A2: make_arm(arm_mnl, A2_FEATURES, C_GRID),
             ARM_A3: make_sel_arm(arm_mnl, A1_FEATURES, LIBRARY, C_GRID)}
CANDIDATE_ARMS = [ARM_A1, ARM_A2, ARM_A3]
_short = {BASELINE: "B0", ARM_A1: "A1", ARM_A2: "A2", ARM_A3: "A3"}

print("=" * 78); print(f"MAIN — K={len(MAIN_ARMS)}, {N_REPEATS} repeats × {N_OUTER} folds"); print("=" * 78)
_t0 = time.time()
OOF, CHOSEN, DEAD, FOLD_ROWS, INNER_DEAD = run_arms(MAIN_ARMS, y3, tag="main ")
print(f"main 완료 {(time.time()-_t0)/60:.1f}분 | 실패 fold {DEAD} | inner 실패 {INNER_DEAD}")
assert all(v == 0 for v in DEAD.values()), f"main arm 에 실패 fold 가 있다 — headline 무효: {DEAD}"

RESULTS = {}
for k in MAIN_ARMS:
    s = summarize(OOF[k], y3)
    s["ci_low"], s["ci_high"], s["p_boot_le_070"] = boot_ci(s["scores"], ybin)
    s["failed_folds"] = DEAD[k]; s["inner_failed"] = INNER_DEAD[k]
    if MAIN_ARMS[k][3]["sets"] is None:
        s["chosen_C"] = {str(c): int(n) for c, n in pd.Series(CHOSEN[k]).value_counts().items()}
    else:
        _ok = [p for p in CHOSEN[k] if isinstance(p, tuple)]
        s["chosen_C"] = {str(c): int(n) for c, n in pd.Series([c for _, c in _ok]).value_counts().items()}
        s["chosen_set"] = {str(nm): int(n) for nm, n in pd.Series([nm for nm, _ in _ok]).value_counts().items()}
    RESULTS[k] = s

main_tbl = pd.DataFrame([{
    "arm": k, "features": len(MAIN_ARMS[k][3]["features"]) if MAIN_ARMS[k][3]["sets"] is None else "2~3 (선택)",
    "repeat 평균": round(v["repeat_mean"], 4), "repeat SD": round(v["repeat_sd"], 4),
    "subject-mean AUC": round(v["subject_mean_auc"], 4),
    "bootstrap 95% CI": f"[{v['ci_low']:.3f}, {v['ci_high']:.3f}]",
    "Hanley SE": round(v["hanley_se"], 4),
    "CN vs MCI": round(v["auc_cn_vs_mci"], 4), "CN vs Dem": round(v["auc_cn_vs_dem"], 4),
    "PR-AUC": round(v["pr_auc"], 4), "failed_folds": v["failed_folds"]} for k, v in RESULTS.items()])
print("\n" + main_tbl.to_string(index=False))

for k, v in RESULTS.items():
    assert v["identity_gap"] < 1e-9, f"Mann-Whitney 분해 항등식 위반: {k}"
print("\n분해 항등식 OK (모든 arm)")

# ── 재현 계약: B0 (run 2 = 20260902_140024_utc B0 arm; run 3·4 에서 차이 0.0) + A1 (run 4 S1 진단, 동일 정의·fold) ──
RUN2_B0 = {
    "subject_mean_auc": 0.6490776490776491, "auc_cn_vs_mci": 0.6053700759583113, "auc_cn_vs_dem": 0.8348348348348348,
    "repeat_mean": 0.6441798941798942,
    "per_repeat": [0.6469326469326468, 0.6663806663806664, 0.6281996281996282, 0.6445016445016445, 0.6578006578006578,
                   0.6383526383526383, 0.6264836264836264, 0.6462176462176462, 0.667953667953668, 0.6555126555126555,
                   0.6382096382096383, 0.6246246246246245, 0.6578006578006579, 0.6369226369226368, 0.6254826254826255,
                   0.6616616616616616, 0.6386386386386387, 0.6290576290576291, 0.6417846417846418, 0.651079651079651]}
RUN4_S1 = {
    "subject_mean_auc": 0.6576576576576576, "auc_cn_vs_mci": 0.6122593181416712, "auc_cn_vs_dem": 0.8506006006006006,
    "per_repeat": [0.655083655083655, 0.667953667953668, 0.6435006435006435, 0.6497926497926497, 0.666094666094666,
                   0.650936650936651, 0.6533676533676532, 0.638209638209638, 0.6662376662376662, 0.660946660946661,
                   0.6292006292006292, 0.6337766337766337, 0.6669526669526669, 0.6563706563706564, 0.6603746603746603,
                   0.6768196768196768, 0.6573716573716574, 0.6535106535106535, 0.6442156442156443, 0.6615186615186615]}
def _contract(res, ref):
    d_rep = float(np.max(np.abs(np.array(res["per_repeat"]) - np.array(ref["per_repeat"]))))
    d = {"max_abs_per_repeat_diff": d_rep, "abs_subject_mean_diff": abs(res["subject_mean_auc"] - ref["subject_mean_auc"]),
         "abs_mci_diff": abs(res["auc_cn_vs_mci"] - ref["auc_cn_vs_mci"]), "abs_dem_diff": abs(res["auc_cn_vs_dem"] - ref["auc_cn_vs_dem"])}
    d["pass"] = bool(all(v < 1e-9 for v in d.values()))
    return d
B_ = RESULTS[BASELINE]
if QUICK or SYNTHETIC:
    B0_REPRO = {"checked": False, "reason": "QUICK/SYNTHETIC 모드"}
    print("\n⚠ B0 / A1 재현 assert 건너뜀 (배선 테스트 모드)")
else:
    B0_REPRO = {"checked": True, "B0_vs_run2": _contract(B_, RUN2_B0), "A1_vs_run4_S1": _contract(RESULTS[ARM_A1], RUN4_S1)}
    B0_REPRO["pass"] = bool(B0_REPRO["B0_vs_run2"]["pass"] and B0_REPRO["A1_vs_run4_S1"]["pass"])
    print(f"\nB0 재현 계약: per-repeat 최대 |차| {B0_REPRO['B0_vs_run2']['max_abs_per_repeat_diff']:.2e} → {'✓' if B0_REPRO['B0_vs_run2']['pass'] else '✗'}"
          f" | A1 vs run 4 S1: per-repeat 최대 |차| {B0_REPRO['A1_vs_run4_S1']['max_abs_per_repeat_diff']:.2e} → {'✓' if B0_REPRO['A1_vs_run4_S1']['pass'] else '✗'}")
    assert B0_REPRO["pass"], "B0/A1 이 이전 run 과 일치하지 않는다 — 계약이 깨졌으므로 Δ 를 해석하지 않는다"

# ── 증분 Δ vs B0 (arm 별, paired subject bootstrap, 동일 fold) ────────────
DELTAS, PER_DELTA = {}, {}
print("\n" + "=" * 78); print("Δ = arm − B0  (동일 fold; 점추정 = 관측 차이, CI = paired subject bootstrap)"); print("=" * 78)
for k in CANDIDATE_ARMS:
    D = delta_block(RESULTS[k]["scores"], B_["scores"], y3); DELTAS[k] = D
    pdlt = np.array(RESULTS[k]["per_repeat"]) - np.array(B_["per_repeat"])
    PER_DELTA[k] = {"mean": float(pdlt.mean()), "sd": float(pdlt.std(ddof=0)), "n_pos": int((pdlt > 0).sum())}
    print(f"  {k}")
    for nm, key in [("merged", "merged"), ("CN vs MCI", "cn_vs_mci"), ("CN vs Dem", "cn_vs_dem")]:
        d = D[key]
        print(f"     Δ {nm:10s} {d[0]:+.4f}  95% CI [{d[1]:+.3f}, {d[2]:+.3f}]  (boot 평균 {d[3]:+.4f})   {'유의' if d[1] > 0 else ('음의 유의' if d[2] < 0 else '0 포함')}")
    print(f"     per-repeat Δ merged 평균 {PER_DELTA[k]['mean']:+.4f} ± {PER_DELTA[k]['sd']:.4f}, Δ>0 repeat {PER_DELTA[k]['n_pos']}/{N_REPEATS}")

# ── 최고 arm (후보 3 중, 판정용) · 상한 arm (B0 포함 4 중, 상한 진술용) · 승자 편향 ──
BEST_ARM = max(CANDIDATE_ARMS, key=lambda k: RESULTS[k]["subject_mean_auc"])
CEIL_ARM = max(MAIN_ARMS, key=lambda k: RESULTS[k]["subject_mean_auc"])
WINNER_BIAS_BOOT, _obsd = winner_bias_boot({k: RESULTS[k]["scores"] for k in CANDIDATE_ARMS}, B_["scores"], ybin)
SEL_FREQ = pd.Series([nm for nm, _ in CHOSEN[ARM_A3] if isinstance(nm, str)]).value_counts()
print("\nA3 inner 선택 빈도 (outer fold 수):"); print(SEL_FREQ.to_string())
print(f"\n최고 후보 arm (판정용): {BEST_ARM} → merged {RESULTS[BEST_ARM]['subject_mean_auc']:.4f} [{RESULTS[BEST_ARM]['ci_low']:.3f}, {RESULTS[BEST_ARM]['ci_high']:.3f}]")
print(f"상한 arm (B0 포함 max): {CEIL_ARM} → {RESULTS[CEIL_ARM]['subject_mean_auc']:.4f}")
print(f"max-over-3 승자 편향 (subject bootstrap, 실제 의존구조): {WINNER_BIAS_BOOT:+.4f} → 편향 보정 최고 Δ {max(_obsd.values()) - WINNER_BIAS_BOOT:+.4f}")
progress("main_done", 1, 1)

# %%% CELL 10 [code]
# ============================================================================
# 셀 10 — 진단 (동일 20 repeats·동일 fold, K 미포함, 승격 금지 — PREREGISTRATION §9): D1 run 4 P 재현 + D2~D5 후보별 고정 arm
# ============================================================================
RUN4_P = {"subject_mean_auc": 0.6589446589446588, "auc_cn_vs_mci": 0.6133192015544957, "auc_cn_vs_dem": 0.8528528528528528,
          "per_repeat": [0.6603746603746604, 0.6557986557986557, 0.6772486772486773, 0.6635206635206635, 0.6605176605176605,
                         0.658086658086658, 0.6590876590876591, 0.6645216645216645, 0.6539396539396539, 0.6658086658086658,
                         0.5864435864435864, 0.6585156585156585, 0.6768196768196768, 0.653939653939654, 0.6386386386386387,
                         0.6874016874016874, 0.637065637065637, 0.6303446303446303, 0.653081653081653, 0.6576576576576576]}
DIAG_ARMS = [("D1 run 4 P (35일 B0 + longest_w_SD) 재현", make_arm(arm_mnl, RUN4_P_FEATURES, C_GRID))]
DIAG_ARMS += [(f"D{i+2} B0_all + {c}", make_arm(arm_mnl, A1_FEATURES + [c], C_GRID)) for i, c in enumerate(LIBRARY)]

print("=" * 78); print(f"진단 ({N_REPEATS} repeats, 동일 fold; 기준 = main B0; 승격 금지)"); print("=" * 78)
DIAG_RES, DIAG = {}, []
for i, (nm, arm) in enumerate(DIAG_ARMS):
    o, ch, dd, fr, idd = run_arms({nm: arm}, y3, verbose=False)
    s = summarize(o[nm], y3); s["failed_folds"] = dd[nm]; s["inner_failed"] = idd[nm]
    s["chosen_C"] = {str(c): int(n) for c, n in pd.Series(ch[nm]).value_counts().items()}
    ok = dd[nm] == 0
    d = delta_block(s["scores"], B_["scores"], y3) if ok else None
    s["delta_vs_b0"] = d; s["promotion"] = "forbidden (PREREGISTRATION §9)"
    DIAG_RES[nm] = s
    row = {"진단 arm": nm, "features": arm[3]["features"], "n": 174, "failed_folds": dd[nm],
           "merged": round(s["subject_mean_auc"], 4) if ok else None,
           "Δ merged vs B0": round(d["merged"][0], 4) if ok else None,
           "Δ merged CI": f"[{d['merged'][1]:+.3f}, {d['merged'][2]:+.3f}]" if ok else "N/A",
           "Δ MCI vs B0": round(d["cn_vs_mci"][0], 4) if ok else None, "Δ Dem vs B0": round(d["cn_vs_dem"][0], 4) if ok else None,
           "promotion": "forbidden"}
    DIAG.append(row)
    print(f"  {nm:46s} " + (f"merged {s['subject_mean_auc']:.4f}  Δ vs B0 {d['merged'][0]:+.4f} [{d['merged'][1]:+.3f},{d['merged'][2]:+.3f}]  ΔMCI {d['cn_vs_mci'][0]:+.4f}  ΔDem {d['cn_vs_dem'][0]:+.4f}"
                            if ok else f"N/A (실패 fold {dd[nm]})"))
    progress("diag", i + 1, len(DIAG_ARMS))
DIAG_DF = pd.DataFrame(DIAG)

_d1 = DIAG_RES[DIAG_ARMS[0][0]]
if QUICK or SYNTHETIC:
    RUN4_REPRO = {"checked": False, "reason": "QUICK/SYNTHETIC 모드"}
else:
    RUN4_REPRO = {"checked": True, **_contract(_d1, RUN4_P)}
    print(f"\nrun 4 P 재현 계약 (D1): per-repeat 최대 |차| {RUN4_REPRO['max_abs_per_repeat_diff']:.2e} → {'✓ 비트 동일' if RUN4_REPRO['pass'] else '✗ 불일치'}")
    assert RUN4_REPRO["pass"], "run 4 P 가 재현되지 않는다 — 창/파싱 규칙이 바뀐 것"

# %%% CELL 11 [code]
# ============================================================================
# 셀 11 — permutation (K=4, 축소 프로토콜, 피험자 단위 3-class 셔플, A3 는 선택까지 재생) + 후보 열 permutation (co-statistic)
# ============================================================================
N_PERM = 20 if QUICK else 2000
PERM_REPEATS, PERM_INNER, PERM_INNER_REP = 2, 4, 1
PERM_C = [0.01, 0.1]
PERM_VALID_MIN = 0.95                 # 실패 fold 가 있는 순열은 버린다; 유효 순열이 95% 미만이면 p 무효
PERM_ARMS = {k: (fn, Xm, PERM_C, meta) for k, (fn, Xm, _, meta) in MAIN_ARMS.items()}

def perm_eval(y3v, arms=PERM_ARMS):
    o, _, dd, _, _ = run_arms(arms, y3v, n_repeats=PERM_REPEATS, n_inner=PERM_INNER,
                              n_inner_rep=PERM_INNER_REP, verbose=False)
    yb = (y3v > 0).astype(int); out = {}
    for k in o:
        try:
            out[k] = float(roc_auc_score(yb, np.nanmean(o[k], axis=0)))
        except Exception:
            out[k] = float("nan")
    out["_dead"] = int(sum(dd.values()))
    return out

OBS_RED = perm_eval(y3)
assert OBS_RED["_dead"] == 0
OBS_MAX = max(OBS_RED[k] for k in MAIN_ARMS)
OBS_DELTA = {k: OBS_RED[k] - OBS_RED[BASELINE] for k in CANDIDATE_ARMS}
OBS_DMAX = max(OBS_DELTA.values())
print("축소 프로토콜 관측치 (permutation null 과 비교 가능):")
for k in MAIN_ARMS:
    print(f"  {k:44s} {OBS_RED[k]:.4f}" + (f"   Δ(축소) {OBS_DELTA[k]:+.4f}   Δ(전체 프로토콜) {DELTAS[k]['merged'][0]:+.4f}" if k in OBS_DELTA else ""))
print(f"  Δmax (arm 중 최대 Δ) {OBS_DMAX:+.4f} | max over K {OBS_MAX:.4f}")
SIGN_AGREE = {k: bool(np.sign(OBS_DELTA[k]) == np.sign(DELTAS[k]["merged"][0])) for k in CANDIDATE_ARMS}
print(f"  축소/전체 Δ 부호 일치: {SIGN_AGREE}")

_SS = np.random.SeedSequence(SEED).spawn(N_PERM)      # 부분 재실행도 재현되도록 시드를 미리 확정
def perm_stat(i):
    rng = np.random.default_rng(_SS[i])
    return perm_eval(y3[rng.permutation(len(y3))])    # 3-class 라벨을 피험자 단위로 셔플

_t = time.time(); _ = [perm_stat(i) for i in range(2)]; _per = (time.time() - _t) / 2
print(f"\npermutation 1회 {_per:.2f}초 → {N_PERM}회 ≈ {_per*N_PERM/60:.0f}분 (순차)")
NULLS_RAW = []; _t0 = time.time()
for i in range(N_PERM):
    NULLS_RAW.append(perm_stat(i))
    if (i + 1) % 100 == 0 or i + 1 == N_PERM:
        el = time.time() - _t0
        print(f"  perm {i+1}/{N_PERM}  {el/60:.1f}분 경과, ETA {el/(i+1)*(N_PERM-i-1)/60:.1f}분")
        progress("perm", i + 1, N_PERM, eta_min=round(el / (i + 1) * (N_PERM - i - 1) / 60, 2))
null_dead_perms = int(sum(1 for d in NULLS_RAW if d["_dead"] > 0))
NULLS = [d for d in NULLS_RAW if d["_dead"] == 0]           # fail-closed: 실패 fold 가 있는 순열은 제외
NULL_VALID_OK = bool(len(NULLS) >= PERM_VALID_MIN * N_PERM)
print(f"유효 순열 {len(NULLS)}/{N_PERM} (실패 fold 포함 순열 {null_dead_perms}개 제외) → {'✓' if NULL_VALID_OK else '⚠⚠ p 무효'}")

null_arm = {k: np.array([d[k] for d in NULLS]) for k in MAIN_ARMS}
null_max = np.max(np.stack([null_arm[k] for k in MAIN_ARMS]), axis=0)
null_delta = {k: null_arm[k] - null_arm[BASELINE] for k in CANDIDATE_ARMS}
null_dmax = np.max(np.stack([null_delta[k] for k in CANDIDATE_ARMS]), axis=0)
_pv = lambda null, obs: float((np.sum(null >= obs) + 1) / (len(null) + 1))
P_MAX = _pv(null_max, OBS_MAX)
P_ARM = {k: _pv(null_arm[k], OBS_RED[k]) for k in MAIN_ARMS}
P_DELTA = {k: _pv(null_delta[k], OBS_DELTA[k]) for k in CANDIDATE_ARMS}
P_DELTA_HOLM = holm(P_DELTA)
P_DELTA_MC = {k: clopper_pearson(int(np.sum(null_delta[k] >= OBS_DELTA[k])), len(NULLS)) for k in CANDIDATE_ARMS}
P_DMAX = _pv(null_dmax, OBS_DMAX)
NULL_P95 = float(np.percentile(null_max, 95)); NULL_DMAX_P95 = float(np.percentile(null_dmax, 95))
WINNER_BIAS_PERM = float(null_dmax.mean() - np.mean([null_delta[k].mean() for k in CANDIDATE_ARMS]))
print("\n" + "=" * 78); print("permutation 결과 (라벨 셔플 = '어떤 신호라도 있는가' null)"); print("=" * 78)
for k in CANDIDATE_ARMS:
    print(f"  Δ null [{k.split(' ')[0]}]  : 평균 {null_delta[k].mean():+.4f} SD {null_delta[k].std(ddof=0):.4f} → 관측 {OBS_DELTA[k]:+.4f}, p_Δ {P_DELTA[k]:.4f} (MC 95% [{P_DELTA_MC[k][0]:.4f}, {P_DELTA_MC[k][1]:.4f}]; Holm {P_DELTA_HOLM[k]:.4f})")
print(f"  → 판정용: Holm p_Δ(BEST={_short[BEST_ARM]}) = {P_DELTA_HOLM[BEST_ARM]:.4f}")
print(f"Δmax null (기술용)  : 평균 {null_dmax.mean():+.4f} SD {null_dmax.std(ddof=0):.4f} 95%ile {NULL_DMAX_P95:+.4f} → 관측 Δmax {OBS_DMAX:+.4f}, p_Δmax = {P_DMAX:.4f} (판정 미사용)")
print(f"max-over-K={len(MAIN_ARMS)} null (기술용): 평균 {null_max.mean():.4f} 95%ile {NULL_P95:.4f} → 관측 {OBS_MAX:.4f}, p_max = {P_MAX:.4f} (B0 포함 max — 0.70 증거 아님)")
print(f"arm 단독 null 평균 : " + ", ".join(f"{_short[k]} {null_arm[k].mean():.4f} (p {P_ARM[k]:.4f})" for k in MAIN_ARMS))
print(f"승자 편향 (permutation null 휴리스틱, 기술용): {WINNER_BIAS_PERM:+.4f}  | bootstrap 추정(판정 보고용): {WINNER_BIAS_BOOT:+.4f}")
NULL_CENTER = {"B0": float(null_arm[BASELINE].mean()), "A3": float(null_arm[ARM_A3].mean())}
NULL_CENTER_OK = bool(0.46 <= NULL_CENTER["B0"] <= 0.54 and 0.46 <= NULL_CENTER["A3"] <= 0.54)
print(("✓ null 이 0.50 부근에 중심 (B0·A3) — CV 구조·선택 재생에 편향 없음" if NULL_CENTER_OK
       else f"⚠⚠ null 평균 B0 {NULL_CENTER['B0']:.3f} / A3 {NULL_CENTER['A3']:.3f} — 선택 누수 의심, p 값 해석 불가"))

# ── co-statistic (판정 미사용): 후보 열 permutation — 라벨·A1 열 고정, 후보 열만 피험자 간 셔플 → '레버가 B0 위에 무엇을 더했는가' null ──
N_COLPERM = 10 if QUICK else 1000
_SS2 = np.random.SeedSequence(SEED + 7).spawn(N_COLPERM)
_XA2 = MAIN_ARMS[ARM_A2][1]; _XA3 = MAIN_ARMS[ARM_A3][1]; _nb = len(A1_FEATURES)
def colperm_stat(i):
    rng = np.random.default_rng(_SS2[i]); perm = rng.permutation(len(y3))
    Xa2 = _XA2.copy(); Xa2[:, _nb:] = Xa2[perm][:, _nb:]                 # A2: longest_w_SD_all 열만 셔플
    Xa3 = _XA3.copy(); Xa3[:, _nb:] = Xa3[perm][:, _nb:]                 # A3: 라이브러리 4열을 같은 순열로 셔플 → 선택 재생
    arms = {"A2col": (arm_mnl, Xa2, PERM_C, {"features": A2_FEATURES, "sets": None}),
            "A3col": make_sel_arm(arm_mnl, A1_FEATURES, LIBRARY, PERM_C, X=Xa3)}
    out = perm_eval(y3, arms)
    return out["A2col"] - OBS_RED[BASELINE], out["A3col"] - OBS_RED[BASELINE], out["_dead"]
_t0 = time.time(); COLNULL = []
for i in range(N_COLPERM):
    COLNULL.append(colperm_stat(i))
    if (i + 1) % 100 == 0 or i + 1 == N_COLPERM:
        el = time.time() - _t0
        print(f"  colperm {i+1}/{N_COLPERM}  {el/60:.1f}분 경과, ETA {el/(i+1)*(N_COLPERM-i-1)/60:.1f}분")
        progress("colperm", i + 1, N_COLPERM)
COLNULL_OK = [c for c in COLNULL if c[2] == 0]
null_col = {ARM_A2: np.array([c[0] for c in COLNULL_OK]), ARM_A3: np.array([c[1] for c in COLNULL_OK])}
P_COL = {k: _pv(null_col[k], OBS_DELTA[k]) for k in (ARM_A2, ARM_A3)}
print(f"\n후보 열 permutation (유효 {len(COLNULL_OK)}/{N_COLPERM}):")
for k in (ARM_A2, ARM_A3):
    print(f"  [{_short[k]}] null Δ 평균 {null_col[k].mean():+.4f} SD {null_col[k].std(ddof=0):.4f} 95%ile {np.percentile(null_col[k],95):+.4f} → 관측 Δ {OBS_DELTA[k]:+.4f}, p_col = {P_COL[k]:.4f} (co-statistic, 판정 미사용)")
print("  A1 은 열 셔플 대응물이 없다 (같은 두 특징의 다른 측정) — p_col 없음")

# %%% CELL 12 [code]
# ============================================================================
# 셀 12 — 사전등록 판정 (G / G− / N+ / N; 단일 판정 통계량) + 상한 진술 + 그림
# ============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

BEST = RESULTS[BEST_ARM]; DB = DELTAS[BEST_ARM]; CEIL = RESULTS[CEIL_ARM]
critA = BEST["subject_mean_auc"] >= 0.70                                           # 점추정 목표 (0.70 자체는 검정되지 않음)
critC = (DB["merged"][1] > 0) and (P_DELTA_HOLM[BEST_ARM] < 0.05)                  # B0 위의 증분이 실재 (CI + Holm permutation p)
if critA and critC:
    TIER = "G 점추정 ≥ 0.70 + B0 대비 증분 실재 (0.70 자체는 미검정)"
elif critA:
    TIER = "G− 점추정 ≥ 0.70 이나 증분 미확정 (0.70 주장 불가)"
elif critC:
    TIER = "N+ 미달 — 상한 < 0.70 이나 B0 대비 증분은 실재"
else:
    TIER = "N 미달 — 이 코호트의 상한"
if not NULL_CENTER_OK or not NULL_VALID_OK:
    TIER += " [⚠ permutation p 해석 불가]"
SHORTFALL_SE = (0.70 - CEIL["subject_mean_auc"]) / CEIL["hanley_se"]
CEIL_CORRECTED = B_["subject_mean_auc"] + max(0.0, max(DELTAS[k]["merged"][0] for k in CANDIDATE_ARMS) - WINNER_BIAS_BOOT)

print("=" * 78); print("최종 결과"); print("=" * 78)
print(main_tbl.to_string(index=False))
print(f"\n최고 후보 arm (판정용): {BEST_ARM}")
print(f"  merged {BEST['subject_mean_auc']:.4f} [{BEST['ci_low']:.3f}, {BEST['ci_high']:.3f}] | MCI {BEST['auc_cn_vs_mci']:.4f} | Dem {BEST['auc_cn_vs_dem']:.4f} | P_boot(AUC ≤ 0.70) {BEST['p_boot_le_070']:.3f}")
print(f"  Δ vs B0 merged {DB['merged'][0]:+.4f} [{DB['merged'][1]:+.3f}, {DB['merged'][2]:+.3f}] | Δ MCI {DB['cn_vs_mci'][0]:+.4f} [{DB['cn_vs_mci'][1]:+.3f}, {DB['cn_vs_mci'][2]:+.3f}] | Δ Dem {DB['cn_vs_dem'][0]:+.4f} [{DB['cn_vs_dem'][1]:+.3f}, {DB['cn_vs_dem'][2]:+.3f}]")
print(f"  판정 p: Holm p_Δ {P_DELTA_HOLM[BEST_ARM]:.4f} (raw {P_DELTA[BEST_ARM]:.4f}, MC 95% [{P_DELTA_MC[BEST_ARM][0]:.4f}, {P_DELTA_MC[BEST_ARM][1]:.4f}]) | 축소/전체 Δ 부호 일치 {SIGN_AGREE[BEST_ARM]}"
      + (f" | p_col {P_COL[BEST_ARM]:.4f}" if BEST_ARM in P_COL else "") + f" | 기술용 p_Δmax {P_DMAX:.4f}, p_max {P_MAX:.4f}")
print(f"상한 arm (B0 포함 max): {CEIL_ARM} → {CEIL['subject_mean_auc']:.4f} [{CEIL['ci_low']:.3f}, {CEIL['ci_high']:.3f}]"
      f" | 0.70 까지 부족분 {0.70 - CEIL['subject_mean_auc']:+.4f} = Hanley SE 의 {SHORTFALL_SE:.2f}배"
      f" | 승자 편향(bootstrap) {WINNER_BIAS_BOOT:+.4f} → 편향 보정 상한 {CEIL_CORRECTED:.4f}")
DEM_COLLAPSE = bool(DB["cn_vs_dem"][2] < 0)
print("\n" + "-" * 78); print("사전등록 판정 (단일 규칙)"); print("-" * 78)
print(f"  A  최고 후보 arm 점추정 ≥ 0.70                 : {BEST['subject_mean_auc']:.4f} → {'통과' if critA else '미달'}")
print(f"  C  Δ CI 하한 > 0 and Holm p_Δ(BEST) < 0.05    : [{DB['merged'][1]:+.3f}], {P_DELTA_HOLM[BEST_ARM]:.4f} → {'통과' if critC else '미달'}")
print(f"  → 판정: {TIER}")
print(f"  Dem 채널 falsifier (Δ_Dem CI 상한 < 0 → 붕괴): {'⚠ 붕괴' if DEM_COLLAPSE else '유지'}")
print(f"  A3 선택: base {int(SEL_FREQ.get('base', 0))}/{N_REPEATS * N_OUTER} fold → {'라이브러리 null (기록)' if SEL_FREQ.get('base', 0) >= 0.5 * N_REPEATS * N_OUTER else '후보가 다수 fold 에서 선택됨 (기록)'}"
      f" | 선택 비용 A1 − A3 = {RESULTS[ARM_A1]['subject_mean_auc'] - RESULTS[ARM_A3]['subject_mean_auc']:+.4f}")
CEILING_STATEMENT = (f"엄격 3조건(피험자 분할·nested CV·MMSE 제외) 아래 174명 코호트에서 합법적 레버를 전부 건 상한은 "
                     f"{CEIL['subject_mean_auc']:.3f} [{CEIL['ci_low']:.3f}, {CEIL['ci_high']:.3f}] ({CEIL_ARM}; 후보 3 arm 의 max-over-3 승자 편향 {WINNER_BIAS_BOOT:+.3f} 보정 시 {CEIL_CORRECTED:.3f}) 이며, "
                     f"B0 0.649 대비 최고 후보 Δ {DB['merged'][0]:+.3f} [{DB['merged'][1]:+.3f}, {DB['merged'][2]:+.3f}] (Holm p_Δ {P_DELTA_HOLM[BEST_ARM]:.3f}). "
                     f"프로그램 headline 은 B0 0.649 로 유지된다 (STRICT3 §10.2).")
print("\n상한 진술: " + CEILING_STATEMENT)
print("=" * 78)

fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
for k, v in RESULTS.items():
    fpr, tpr, _ = roc_curve(ybin, v["scores"])
    ax[0].plot(fpr, tpr, lw=1.6, label=f"{_short[k]} {v['subject_mean_auc']:.3f}")
ax[0].plot([0, 1], [0, 1], "k--", lw=.8); ax[0].legend(fontsize=9)
ax[0].set_xlabel("1 - Specificity"); ax[0].set_ylabel("Sensitivity"); ax[0].set_title("ROC (subject-mean OOF), K=4")
ax[1].hist(null_delta[BEST_ARM], bins=45, alpha=.75, label=f"null delta ({_short[BEST_ARM]} - B0)")
ax[1].axvline(float(np.percentile(null_delta[BEST_ARM], 95)), color="orange", ls="--", label="null p95")
ax[1].axvline(OBS_DELTA[BEST_ARM], color="red", lw=2, label=f"observed {OBS_DELTA[BEST_ARM]:+.3f}")
ax[1].set_xlabel("delta AUC (reduced protocol)"); ax[1].set_title(f"permutation null of delta, best arm (N={len(NULLS)}, Holm p={P_DELTA_HOLM[BEST_ARM]:.3f})")
ax[1].legend(fontsize=8)
_lbl = ["merged", "CN vs MCI", "CN vs Dem"]; _x = np.arange(3); _wd = 0.2
for i, k in enumerate(MAIN_ARMS):
    v = RESULTS[k]
    ax[2].bar(_x + (i - 1.5) * _wd, [v["subject_mean_auc"], v["auc_cn_vs_mci"], v["auc_cn_vs_dem"]], _wd, label=_short[k])
ax[2].axhline(.5, color="k", ls="--", lw=.8); ax[2].axhline(.70, color="green", ls=":", lw=1.2)
ax[2].set_xticks(_x); ax[2].set_xticklabels(_lbl); ax[2].set_ylim(.3, 1.0)
ax[2].set_title("sub-AUC by arm"); ax[2].legend(fontsize=8)
plt.tight_layout()
FIG_PATH = OUT_DIR / "report.png"; plt.savefig(FIG_PATH, dpi=140); plt.close(fig)
print(f"그림 저장: {FIG_PATH}")

_C = pd.Series([c for c in CHOSEN[ARM_A2]]).mode(); _C = float(_C.iloc[0]) if len(_C) else 0.01
_full = make_lr(_C).fit(MAIN_ARMS[ARM_A2][1], y3)
COEF_DF = pd.DataFrame(_full.named_steps["lr"].coef_.T, index=A2_FEATURES, columns=["CN", "MCI", "Dem"])
print(f"\nA2 전체 적합 계수 (C={_C}, 진단용 — 이 값으로 아무것도 고르지 않는다):"); print(COEF_DF.round(3).to_string())

# %%% CELL 13 [code]
# ============================================================================
# 셀 13 — 결정성 확인 (main 4 arm 전체 재실행 → 전체 OOF 행렬 해시) + 코드 해시 계약 + 저장 (해시 ID만)
# ============================================================================
def _hash_arr(a):
    return hashlib.sha256(np.round(np.asarray(a, float), 10).tobytes()).hexdigest()[:16]
_o2, _, _dead2, _, _ = run_arms(MAIN_ARMS, y3, verbose=False)
DETERMINISM = {}
for k in MAIN_ARMS:
    h1, h2 = _hash_arr(OOF[k]), _hash_arr(_o2[k])
    DETERMINISM[k] = {"hash": h1, "recheck": h2, "match": bool(h1 == h2)}
    print(f"결정성 [{k}] 전체 OOF 재실행: {h1} vs {h2} → {'일치' if h1 == h2 else '⚠ 불일치'}")
DETERMINISM["null_max_hash"] = _hash_arr(null_max); DETERMINISM["null_dmax_hash"] = _hash_arr(null_dmax)
assert all(v["match"] for k, v in DETERMINISM.items() if isinstance(v, dict)), "결정성 불일치"

# ── 코드 해시 계약: 노트북(환경변수 > 실험 폴더)의 코드셀 sha256 을 PREREGISTRATION_KO.md 에 기록된 값과 비교 ──
def _code_hash(nbp):
    _nb = json.loads(Path(nbp).read_text(encoding="utf-8"))
    return hashlib.sha256("\n".join("".join(c["source"]) for c in _nb["cells"] if c["cell_type"] == "code").encode("utf-8")).hexdigest()
PREREG_HASH = PREREG_EXPECTED = None
_nbp = os.environ.get("EXP3_NB_PATH") or (str(EXP_DIR / f"{EXPERIMENT}.ipynb") if EXP_DIR else None)
_prp = os.environ.get("EXP3_PREREG") or (str(EXP_DIR / "PREREGISTRATION_KO.md") if EXP_DIR else None)
try:
    if _nbp and Path(_nbp).is_file():
        PREREG_HASH = _code_hash(_nbp)
    if _prp and Path(_prp).is_file():
        _m = re.search(r"sha256 = `([0-9a-f]{64})`", Path(_prp).read_text(encoding="utf-8"))
        PREREG_EXPECTED = _m.group(1) if _m else None
except Exception as exc:
    print(f"코드 해시 계산 실패: {exc}")
PREREG_MATCH = (PREREG_HASH == PREREG_EXPECTED) if (PREREG_HASH and PREREG_EXPECTED) else None
print(f"코드셀 sha256: {PREREG_HASH} | 사전등록 기록값: {PREREG_EXPECTED} → {'✓ 일치' if PREREG_MATCH else ('✗ 불일치 — 코드가 바뀐 것 (재실행이 아니라 불일치로 기록)' if PREREG_MATCH is False else '⚠ 비교 불가 (경로/기록 없음)')}")
if not (QUICK or SYNTHETIC) and PREREG_MATCH is None:
    print("⚠⚠ 실데이터 run 인데 코드 해시 계약을 확인하지 못했다 — EXP3_NB_PATH / EXP3_PREREG 를 지정하라")

def _strip(d):
    return {kk: vv for kk, vv in d.items() if kk != "scores"}
FINAL = {
    "run_id": RUN_ID, "experiment": EXPERIMENT,
    "task": "CN(0) vs MCI+Dem(1), 피험자 단위, MMSE 제외, nested CV",
    "quick_mode": bool(QUICK), "synthetic_data": bool(SYNTHETIC),
    "status": "ceiling experiment — 사용자 지시(2026-09-05)로 STRICT3 §10 stop rule 뒤에 1회 실행; §10.4 두 항목(새 특징 정의·fold-local pool 스크린) 이 노트북 1회 예외; 확인 실험이 아니라 상한 측정",
    "exposure_status": {"record": "exploratory-record",
                        "B0": "reproduction (run 2/3/4, 0.6491 known)", "A1": "reproduction of run-4 S1 (0.6577 known; STRICT3 §8 #18)",
                        "D1": "reproduction of run-4 P (0.6589 known; §8 #17)",
                        "A2": "re-test of two known levers (#17 + #18) on all valid days; label value never computed before this run",
                        "longest_w_SD_all": "construct exposed (run 4, 35-day)", "onset_SD_all": "onset used in run-2/3 coupling features; onset SD label value never computed",
                        "clsH_SD_all": "class_5min stream had variants in Training-141 label-aware screens (§8 #8, §9.2); this definition never computed",
                        "low_lag1_all": "never computed", "steps_SD": "EXCLUDED from library — 174-cohort Dem-axis AUC 0.6219 recorded in run 2 (STRICT3 §6)",
                        "label_aware_arms": "main 4 + diagnostics 5 + reduced-protocol 4 = 13; decision on max of 3 candidate arms only (Holm 3)"},
    "verdict_tier": TIER, "ceiling_statement": CEILING_STATEMENT, "best_candidate_arm": BEST_ARM, "ceiling_arm": CEIL_ARM,
    "baseline_run": "Binary_DaytimeVariety_Nested 20260902_140024_utc (B0 arm); run 3·4 및 본 run 에서 비트 재현",
    "cohort": {"n": len(SIDS), "CN": int((y3 == 0).sum()), "MCI": int((y3 == 1).sum()), "Dem": int((y3 == 2).sum()),
               "window_days_B0": WINDOW_DAYS, "wear_gate": WEAR_GATE, "nonwear_met": NONWEAR_MET, "sed_met": SED_MET,
               "min_window_min": MIN_WINDOW, "zero_run_max_min": ZERO_RUN_MAX, "cls_len_gate": CLS_LEN, "epoch_worn_min": EPOCH_WORN_MIN,
               "min_worn_epoch": MIN_WORN_EPOCH, "min_trans": MIN_TRANS, "min_days_detrend": MIN_DAYS_DETREND, "min_pairs_lag1": MIN_PAIRS_LAG1,
               "cls_parse_min": CLS_PARSE_MIN, "nan_max_library": NAN_MAX_LIBRARY},
    "protocol": {"outer_folds": N_OUTER, "outer_repeats": N_REPEATS, "inner_folds": N_INNER, "inner_repeats": N_INNER_REP,
                 "C_grid": C_GRID, "tolerance": TOL, "seed": SEED, "stratify": "3-class", "K_arms": len(MAIN_ARMS),
                 "n_boot": N_BOOT, "seeds": {"fold": "100+r", "inner": "7000+37r+f+rr", "boot_ci": SEED, "paired_boot": SEED + 1, "winner_bias_boot": SEED + 2,
                                             "perm": f"SeedSequence({SEED}).spawn({N_PERM})", "colperm": f"SeedSequence({SEED + 7}).spawn({N_COLPERM})"},
                 "estimator": "multinomial L2 LR (lbfgs), score = 1 - P(CN)",
                 "A3_selection": "inner CV mean AUC over {base, base+c (c in LIBRARY)} x C_grid; tie within tol → fewer features → smaller C → earlier set",
                 "fold_membership": "not saved; reproducible from sorted(SIDS) + y3 + StratifiedKFold(5, shuffle, random_state=100+r)"},
    "features": {"B0": DEM_BLOCK, "A1": A1_FEATURES, "A2": A2_FEATURES, "A3_base": A1_FEATURES, "A3_library": LIBRARY,
                 "fingerprints": FP, "correlation_spearman": CORR.round(4).to_dict(), "reliability_split_half": RELIAB},
    "contract_tests": {"CT_F_feature_matrix": CT_F, "b0_a1_reproduction": B0_REPRO, "run4_P_reproduction": RUN4_REPRO,
                       "prereg_code_sha256": PREREG_HASH, "prereg_code_sha256_expected": PREREG_EXPECTED, "prereg_code_match": PREREG_MATCH},
    "arms": {k: _strip(v) for k, v in RESULTS.items()},
    "deltas_vs_b0": {k: {"merged": v["merged"], "cn_vs_mci": v["cn_vs_mci"], "cn_vs_dem": v["cn_vs_dem"],
                         "note": "tuple = (observed delta, ci_low, ci_high, bootstrap mean delta)", **PER_DELTA[k]} for k, v in DELTAS.items()},
    "a3_selection_frequency": {str(k): int(v) for k, v in SEL_FREQ.items()},
    "acceptance": {"primary_statistic": "Holm-adjusted permutation p_delta of BEST (argmax subject-mean AUC over A1/A2/A3), family = 3 candidate arms",
                   "A_point_ge_070": {"value": BEST["subject_mean_auc"], "pass": bool(critA)},
                   "C_delta_ci_low_gt_0_and_holm_p_lt_005": {"delta_ci_low": DB["merged"][1], "holm_p": P_DELTA_HOLM[BEST_ARM], "raw_p": P_DELTA[BEST_ARM],
                                                             "mc_interval": P_DELTA_MC[BEST_ARM], "pass": bool(critC)},
                   "tier": TIER, "null_center_ok": NULL_CENTER_OK, "null_valid_ok": NULL_VALID_OK,
                   "sign_agreement_reduced_vs_full": SIGN_AGREE, "dem_channel_collapse": DEM_COLLAPSE,
                   "p_boot_auc_le_070_best": BEST["p_boot_le_070"], "ci_low_vs_070_best": BEST["ci_low"] - 0.70,
                   "ceiling_raw": CEIL["subject_mean_auc"], "ceiling_bias_corrected": CEIL_CORRECTED,
                   "shortfall_to_070": 0.70 - CEIL["subject_mean_auc"], "shortfall_in_SE": SHORTFALL_SE,
                   "winner_bias_bootstrap": WINNER_BIAS_BOOT, "winner_bias_perm_heuristic": WINNER_BIAS_PERM,
                   "selection_cost_A1_minus_A3": RESULTS[ARM_A1]["subject_mean_auc"] - RESULTS[ARM_A3]["subject_mean_auc"]},
    "permutation": {"n_perm": int(N_PERM), "n_valid": int(len(NULLS)), "n_dropped_dead": null_dead_perms, "K": len(PERM_ARMS),
                    "reduced": {"outer_repeats": PERM_REPEATS, "inner_folds": PERM_INNER, "inner_repeats": PERM_INNER_REP, "C_grid": PERM_C},
                    "observed_reduced": {k: OBS_RED[k] for k in MAIN_ARMS}, "observed_delta_reduced": OBS_DELTA, "observed_dmax_reduced": OBS_DMAX,
                    "p_delta": P_DELTA, "p_delta_holm": P_DELTA_HOLM, "p_delta_mc95": P_DELTA_MC, "p_dmax": P_DMAX, "p_arm": P_ARM, "p_max": P_MAX,
                    "null_delta_mean": {k: float(null_delta[k].mean()) for k in CANDIDATE_ARMS}, "null_delta_sd": {k: float(null_delta[k].std(ddof=0)) for k in CANDIDATE_ARMS},
                    "null_dmax_mean": float(null_dmax.mean()), "null_dmax_p95": NULL_DMAX_P95, "null_arm_mean": {k: float(null_arm[k].mean()) for k in MAIN_ARMS},
                    "null_max_p95": NULL_P95, "null_center": NULL_CENTER,
                    "column_permutation": {"n": int(N_COLPERM), "n_valid": int(len(COLNULL_OK)), "p_col": P_COL,
                                           "null_mean": {k: float(null_col[k].mean()) for k in null_col}, "null_sd": {k: float(null_col[k].std(ddof=0)) for k in null_col},
                                           "note": "co-statistic; 판정 미사용; A1 은 대응물 없음"}},
    "diagnostics": {k: _strip(v) for k, v in DIAG_RES.items()},
    "environment": {"colab": bool(IN_COLAB), "sklearn": sklearn.__version__, "numpy": np.__version__,
                    "pandas": pd.__version__, "n_cpu": int(N_JOBS), "python": sys.version.split()[0]},
    "determinism": DETERMINISM,
    "elapsed_min": round((time.time() - _T_START) / 60, 2),
}
(OUT_DIR / "FINAL_REPORT.json").write_text(json.dumps(FINAL, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
main_tbl.to_csv(OUT_DIR / "main_arms.csv", index=False, encoding="utf-8-sig")
pd.DataFrame([{"arm": k, "endpoint": e, "delta": v[e][0], "ci_low": v[e][1], "ci_high": v[e][2], "boot_mean": v[e][3]}
              for k, v in DELTAS.items() for e in ("merged", "cn_vs_mci", "cn_vs_dem")]).to_csv(OUT_DIR / "delta_bootstrap.csv", index=False, encoding="utf-8-sig")
DIAG_DF.to_csv(OUT_DIR / "diagnostic_arms.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(FOLD_ROWS).to_csv(OUT_DIR / "fold_aucs.csv", index=False, encoding="utf-8-sig")
pd.DataFrame([{"arm": k, **v["chosen_C"]} for k, v in RESULTS.items()]).to_csv(OUT_DIR / "chosen_C.csv", index=False, encoding="utf-8-sig")
SEL_FREQ.rename_axis("set").reset_index(name="n_outer_folds").to_csv(OUT_DIR / "a3_selection_frequency.csv", index=False, encoding="utf-8-sig")
CONF.to_csv(OUT_DIR / "confound_audit.csv", index=False, encoding="utf-8-sig")
CORR.to_csv(OUT_DIR / "feature_correlation.csv", encoding="utf-8-sig")
pd.DataFrame(RELIAB).T.to_csv(OUT_DIR / "reliability.csv", encoding="utf-8-sig")
COEF_DF.to_csv(OUT_DIR / "a2_coefficients.csv", encoding="utf-8-sig")
_hid = np.array([HID[s] for s in SIDS])
FEAT_ALL.set_index(pd.Index(_hid, name="hid")).to_csv(OUT_DIR / "feature_matrix_hashed.csv", encoding="utf-8-sig")
(DA.assign(hid=DA["sid"].map(HID), date=DA["date"].astype(str)).drop(columns=["sid"])
   .to_csv(OUT_DIR / "daily_metrics_all_hashed.csv", index=False, encoding="utf-8-sig"))
np.save(OUT_DIR / "null_max.npy", null_max); np.save(OUT_DIR / "null_dmax.npy", null_dmax)
for k in MAIN_ARMS:
    np.save(OUT_DIR / f"null_arm__{_short[k]}.npy", null_arm[k])
    np.save(OUT_DIR / f"oof_matrix__{_short[k]}.npy", OOF[k])           # 20 × 174 OOF 행렬 (비트 재현 anchor)
for k in null_col:
    np.save(OUT_DIR / f"null_colperm__{_short[k]}.npy", null_col[k])
pd.DataFrame({"hid": _hid, "y3": y3, "y": ybin, **{f"score__{_short[k]}": v["scores"] for k, v in RESULTS.items()}}
             ).to_csv(OUT_DIR / "oof_predictions_hashed.csv", index=False, encoding="utf-8-sig")   # 진단 arm 점수는 내보내지 않는다 (승격 금지)
json.dump({"subject_split": "StratifiedKFold(5, shuffle, random_state=100+r) on 3-class label; train∩test=∅ asserted for all repeats",
           "mmse": "3.CognitiveFunction 경로 미접근 + 컬럼명 fail-closed 가드 (모든 arm 특징에 적용)",
           "preprocessing": "SimpleImputer+StandardScaler inside Pipeline, fit on outer-train only; A3 feature-set selection uses inner CV on outer-train only; permutation replays A3 selection",
           "feature_independence": "90-subject subset rebuild bit-identical for all subject-level columns (cell 5). Day-level rules (cells 3-4) are fixed constants + row-wise ops and are NOT covered by the rebuild test",
           "ids": "outputs use SHA-256[:16] hashed ids only (AGENTS.md §6)",
           "contract_tests": FINAL["contract_tests"], "failed_folds_main": DEAD, "determinism": DETERMINISM},
          open(OUT_DIR / "leakage_audit.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
progress("done", 1, 1, tier=TIER)
print(f"\n저장 → {OUT_DIR}")
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name:32s} {f.stat().st_size/1024:8.1f} KB")
print("\n" + "=" * 78)
print(f"판정: {TIER}")
print(f"최고 후보 arm {BEST_ARM}: merged {BEST['subject_mean_auc']:.4f} [{BEST['ci_low']:.3f}, {BEST['ci_high']:.3f}] | MCI {BEST['auc_cn_vs_mci']:.4f} | Dem {BEST['auc_cn_vs_dem']:.4f}")
print(f"Δ vs B0 {DB['merged'][0]:+.4f} [{DB['merged'][1]:+.3f}, {DB['merged'][2]:+.3f}] | Holm p_Δ {P_DELTA_HOLM[BEST_ARM]:.4f} | 상한(B0 포함) {CEIL['subject_mean_auc']:.4f} | 총 {(time.time()-_T_START)/60:.1f}분")
print("=" * 78)
