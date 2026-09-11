# %%% CELL 0 [markdown]
# Binary_DemScreen_NoMMSE_Nested — CN+MCI vs Dementia, 웨어러블 전용

**과제**: CN(111) + MCI(51) = 162 (음성) **vs** Dem(12) (양성), 174명 풀링(train 141 + validation 33).
유병률 **0.0690**.

**4가지 하드 계약** — 셀마다 assert 로 강제된다.

1. **MMSE 완전 배제.** `3.CognitiveFunction` 경로를 열지 않고, 특징 이름에 대한 fail-closed 정규식 가드가
   `_MUST_PASS` / `_MUST_FAIL` 자기검증과 함께 돈다.
2. **피험자 단위 독립 분할.** 특징 생성이 피험자당 35~122일을 정확히 1행으로 접으므로 행 분할이 곧
   피험자 분할이다 — 하루 단위 무작위 K-fold 가 만드는 0.95 인공물이 **구조적으로 불가능**하다.
   그래도 300개 outer fold 전부에 대해 train∩test=∅ 를 실행 전에 검사한다.
3. **Nested CV.** 특징 블록·정규화 세기·학습기 계열·운영 임계값을 **전부 inner fold 안에서만** 고른다.
   outer-test 는 이미 굳은 설정을 1회 점수화하는 데만 쓴다.
4. **CN+MCI vs Dem 이진분류.**

---

## 이 과제에서 지금까지 실측된 값 (전부 저장소에 파일로 남아 있음)

| 근거 | ROC-AUC | 프로토콜 | 4조건 |
|---|---:|---|---|
| `Binary_Google_DemScreen` `wearable_only__full` (174명, 112특징) | **0.7184** ± 0.036 | 20 rep × outer5 × inner4 | ✓ 전부 준수 |
| 같은 실행 `wearable_only__filtered` (172명, Dem 11) | **0.8304** ± 0.040 | 동일 + 라벨-블라인드 QC | ✓ |
| `Binary_Google_DemRankAUC_select1` `nested\|wd_full` (556특징) | 0.7303 ± 0.042 | 동일 | ✓ |
| 같은 실행 `wd_core` 13특징 + xgboost | **0.8644** ± 0.028 | 고정설정 5 rep, **nested 아님**(210행 스크리닝 1위) | ✗ nested |
| `wearable_plus_mmse__filtered` | 0.9225 | — | ✗ **MMSE 사용** |
| `Hyunsoo` `activity_low_std` 단독 (173명) | 0.9087 ± 0.0046 | — | ✗ post-selection, Dem 1명 지표 기반 제외 |

**0.9 이상은 4조건을 동시에 만족한 상태로 나온 적이 한 번도 없다.** 위 표의 0.9 이상 3건은 각각
MMSE 사용 / 전코호트 SHAP 을 읽고 특징을 고른 뒤 AUC 가 오른다는 이유로 양성 1명을 제외 /
소진된 33명 벤치마크다. 특히 `± 0.0046` 은 repeat 간 분할 잡음이며 실제 표집 SE 는 **0.0604 (13.1배)** 다.

## 그럼에도 남아 있는 헤드룸

0.7184(112특징 in-fold 선택)와 0.8644(13특징 고정)의 **+0.146** 차이는 모델이 아니라 **특징 예산** 효과다.
아직 어떤 nested 루프에도 들어간 적 없는 레버가 셋 있다.

- `low_roll7SD` — 일별 저강도 활동시간의 7일 rolling SD 평균. 전코호트 단변량 **0.8513**, 저장소 최고치.
- across-day intraday `__std` — `metactive_frac_SD` **0.8349**, `cls_lowfrac_SD` **0.8338**. 1분 MET /
  5분 class 스트림에서만 나오는데, 선행 112·556·696 특징 블록은 전부 일별 요약만 썼다.
- `wd_core` 13특징 블록 자체.

## 정직한 기대 밴드

**FULL 174명 0.80~0.87 / QC 172명 0.86~0.92.** 양성 12명에서 AUC 는 1944개 쌍 위의 이산량이고
**0.900 은 도달 가능한 값조차 아니다**(인접값 0.899691 / 0.900206). 양성 1명의 순위 이동이 최대
0.0833 을 움직이므로, 이 노트북은 점추정 하나가 아니라 **점추정 + 부트스트랩 CI + LOPO 최소값 +
permutation + 유효 K** 를 한 문장으로 함께 출력한다.

## 설계 결정 (사전 고정)

- **헤드라인은 174명 전원.** 라벨-블라인드 QC 코호트(172명)는 나란히 보고하되 더 좋은 쪽을 헤드라인으로
  고르지 않는다. QC 이득의 거의 전부가 양성 1명 제거에서 나오기 때문이다.
- **사전등록 arm 6개(K=6)가 메인**, 0.9 를 노리는 탐색은 셀 15 로 분리하고 best-of-K null 과
  winner-bias 보정치를 강제 병기한다.
- 60 repeats (시드군 3 × 20) — 첫 시드군 20 repeats 는 선행 두 실행과 직접 비교하려고 따로도 보고한다.

# %%% CELL 1 [code]
# ============================================================================
# 셀 1 — 환경(BLAS 스레드는 numpy import 전에 고정), 시드, 경로, 실행 모드, 진행 로깅
# ============================================================================
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import re, sys, json, math, time, hashlib, warnings, itertools
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:
    pass
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

EXPERIMENT = "Binary_DemScreen_NoMMSE_Nested"
SEED = 20260907
np.random.seed(SEED)

# 실행 모드 — 코드는 바꾸지 않고 예산만 줄인다. 정식 실행은 둘 다 꺼져 있어야 한다.
QUICK     = os.environ.get("DS_QUICK", "0") == "1"        # 배선 검증 전용: 이 실행의 숫자는 어디에도 보고하지 않는다
SYNTHETIC = os.environ.get("DS_SYNTHETIC", "0") == "1"    # 합성 데이터: 코호트 계약 assert 를 완화

IN_COLAB = "google.colab" in sys.modules
import sklearn
print(f"Colab {IN_COLAB} | numpy {np.__version__} | pandas {pd.__version__} | sklearn {sklearn.__version__}"
      f" | QUICK {QUICK} | SYNTHETIC {SYNTHETIC}")
if QUICK or SYNTHETIC:
    print("!! 배선 검증 모드 — 이 실행의 어떤 숫자도 성능으로 보고하지 않는다.")

if IN_COLAB:
    from google.colab import drive
    try:
        drive.mount("/content/drive")
    except Exception as exc:
        print(f"drive.mount 건너뜀: {exc}")

_env_root = os.environ.get("DS_DATA_ROOT")
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

RUN_ID = (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
          + ("_QUICK" if QUICK else "") + ("_SYN" if SYNTHETIC else ""))
if os.environ.get("DS_OUT_ROOT"):
    _base = Path(os.environ["DS_OUT_ROOT"])
elif Path("/content/drive/MyDrive").exists():
    _base = Path("/content/drive/MyDrive")
elif _here.name == EXPERIMENT:
    _base = _here
elif (DATA_ROOT.parent / "SangHyo" / "Binary" / EXPERIMENT).exists():
    _base = DATA_ROOT.parent / "SangHyo" / "Binary" / EXPERIMENT
else:
    _base = _here
OUT_DIR = _base / f"{EXPERIMENT}_result" / RUN_ID
(OUT_DIR / "figs").mkdir(parents=True, exist_ok=True)
print(f"OUT_DIR   : {OUT_DIR}")

# ---------------------------------------------------------------- 프로토콜 상수
SEED_GROUPS   = (100,) if QUICK else (100, 500, 900)   # 시드군 3개 × 20 = 60 repeats
R_PER_GROUP   = 2 if QUICK else 20
OUTER_K, INNER_K, INNER_REPS = 5, 4, 2
C_GRID        = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
TOL           = 0.005            # inner AUC 동률 허용오차 → 더 단순한 후보 선택
N_BOOT        = 300 if QUICK else 5000
N_PERM        = 20  if QUICK else 1000
PERM_REPEATS  = 1   if QUICK else 2        # permutation 축소 프로토콜 (repeats)
LOPO_REPEATS  = 1   if QUICK else 5        # LOPO 실재 재실행 축소 프로토콜
EXPLORE_REPS  = 1   if QUICK else 20       # 탐색 섹션 repeats
N_REPEATS     = len(SEED_GROUPS) * R_PER_GROUP
print(f"프로토콜: {N_REPEATS} repeats × outer {OUTER_K} × inner {INNER_K}×{INNER_REPS}"
      f" | boot {N_BOOT} | perm {N_PERM} | seed {SEED}")

N_JOBS = max(1, os.cpu_count() or 1)
_T_START = time.time()
_PROG = {}
def progress(stage, done, total, **extra):
    """fold 단위 진행 기록 → PROGRESS.json. repeat 단위만 찍으면 멈춘 것처럼 보인다."""
    el = time.time() - _T_START
    rate = (el / done) if done else float("nan")
    _PROG.update({"stage": stage, "done": int(done), "total": int(total),
                  "elapsed_min": round(el / 60, 2),
                  "eta_min": round(rate * (total - done) / 60, 2) if done else None,
                  "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **extra})
    try:
        (OUT_DIR / "PROGRESS.json").write_text(json.dumps(_PROG, ensure_ascii=False, default=float))
    except Exception:
        pass

def tick(stage, done, total, every=10, **extra):
    progress(stage, done, total, **extra)
    if done == 1 or done == total or done % every == 0:
        el = time.time() - _T_START
        eta = (el / done) * (total - done) / 60 if done else 0.0
        print(f"  [{stage}] {done}/{total}  경과 {el/60:.1f}분  남은 예상 {eta:.1f}분", flush=True)

progress("init", 0, 1)
print(f"실행 ID {RUN_ID}")

# %%% CELL 2 [code]
# ============================================================================
# 셀 2 — 적재 + activity ⋈ sleep 조인 + 라벨 계약 + MMSE fail-closed 가드
# ============================================================================
# 금지 토큰: MMSE/진단/식별자 + 순응도·수집량 프록시 + 날짜 파생.
# 부분문자열이 아니라 토큰 경계로 잡는다(과거 'S2b_recovery_slope' 가 'cover' 에 오탐된 사고).
FORBIDDEN_PATTERN = re.compile(
    r"(?:^|_)mmse|^q\d+(?:_|$)|^total$|diag|doctor|sample_email|(?:^|_)email"
    r"|non_?wear|(?:^|_)n_days(?:$|_)|(?:^|_)span(?:$|_)"
    r"|(?:^|_)cover(?:age)?(?:$|_)|(?:^|_)gap_|missfrac|valid_days"
    r"|(?:^|_)month(?:$|_)|(?:^|_)start_date|day_of_year|(?:^|_)doy(?:$|_)",
    re.IGNORECASE)
_MUST_PASS = ["low_SD", "low_MED", "low_roll7SD", "low_IQR", "low_RANGE", "rest_Q25",
              "light_MED", "restless_Q75", "metactive_frac_SD", "cls_lowfrac_SD",
              "hyp_onsetep_SD", "RAabs_SD", "M10_SD", "activity_total",
              "wd_activity_low__std", "wd_sleep_light__mean", "wd_arch_fragmentation__mean",
              "wd_circ_midpoint__circsd_h", "wd_activity_average_met__mean",
              "wd_sleep_duration__std", "wd_activity_steps__mean", "low_roll7SD_detr", "lowwn_SD"]
_MUST_FAIL = ["mmse_TOTAL", "TOTAL", "Q13_2", "Q01", "DIAG_NM", "DIAG_SEQ", "DOCTOR_NM",
              "SAMPLE_EMAIL", "EMAIL", "activity_non_wear", "nonwear_mean", "n_days",
              "hr_missfrac", "span_days", "coverage", "gap_days", "valid_days",
              "start_month", "day_of_year"]
_fp = [n for n in _MUST_PASS if FORBIDDEN_PATTERN.search(n)]
_fn = [n for n in _MUST_FAIL if not FORBIDDEN_PATTERN.search(n)]
assert not _fp, f"가드 오탐(허용돼야 할 이름을 막음): {_fp}"
assert not _fn, f"가드 미탐(막아야 할 이름을 통과시킴): {_fn}"
print(f"MMSE/교란 fail-closed 가드 자기검증 통과 (허용 {len(_MUST_PASS)} / 차단 {len(_MUST_FAIL)})")

def assert_no_forbidden(names, where):
    bad = [n for n in names if FORBIDDEN_PATTERN.search(str(n))]
    if bad:
        raise RuntimeError(f"[fail-closed] {where} 에 금지 컬럼: {bad}")

SRC = {
    "act_tr":   DATA_ROOT / "1.Training"   / "SourceData"   / "1.Gait"  / "train_activity.csv",
    "act_va":   DATA_ROOT / "2.Validation" / "SourceData"   / "1.Gait"  / "val_activity.csv",
    "slp_tr":   DATA_ROOT / "1.Training"   / "SourceData"   / "2.Sleep" / "train_sleep.csv",
    "slp_va":   DATA_ROOT / "2.Validation" / "SourceData"   / "2.Sleep" / "val_sleep.csv",
    "lab_tr_g": DATA_ROOT / "1.Training"   / "LabelingData" / "1.Gait"  / "training_label.csv",
    "lab_tr_s": DATA_ROOT / "1.Training"   / "LabelingData" / "2.Sleep" / "training_label.csv",
    "lab_va_g": DATA_ROOT / "2.Validation" / "LabelingData" / "1.Gait"  / "val_label.csv",
    "lab_va_s": DATA_ROOT / "2.Validation" / "LabelingData" / "2.Sleep" / "val_label.csv",
}
for k, p in SRC.items():
    if "3.CognitiveFunction" in str(p) or "mmse" in p.name.lower():
        raise RuntimeError("[fail-closed] MMSE 경로가 소스 목록에 있습니다.")
    if not p.exists():
        raise FileNotFoundError(f"{k}: {p}")
print("소스 8개 확인 — 3.CognitiveFunction 경로는 목록에 없음 (MMSE 미개봉)")

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

_lab_tr = _read_label(SRC["lab_tr_g"], SRC["lab_tr_s"])
_lab_va = _read_label(SRC["lab_va_g"], SRC["lab_va_s"])
_ovl = set(_lab_tr["SAMPLE_EMAIL"]) & set(_lab_va["SAMPLE_EMAIL"])
assert not _ovl, f"train/val 피험자 중복 {len(_ovl)}건 — 중단"
lab = pd.concat([_lab_tr.assign(split="train"), _lab_va.assign(split="val")], ignore_index=True)
lab = lab.rename(columns={"SAMPLE_EMAIL": "sid"})
lab["y3"] = lab["DIAG_NM"].map({"CN": 0, "MCI": 1, "Dem": 2}).astype(int)
lab["y"] = (lab["y3"] == 2).astype(int)          # ★ 양성 = Dem. 다른 폴더의 y(=MCI+Dem)와 정의가 다르다.
_c = lab["DIAG_NM"].value_counts().to_dict()
print(f"피험자 {len(lab)}명  CN {_c.get('CN',0)} / MCI {_c.get('MCI',0)} / Dem {_c.get('Dem',0)}"
      f"  → 이진 음성 {int((lab['y']==0).sum())} vs 양성 {int(lab['y'].sum())}"
      f"  (유병률 {lab['y'].mean():.4f})")
if not SYNTHETIC:
    assert len(lab) == 174 and _c.get("CN") == 111 and _c.get("MCI") == 51 and _c.get("Dem") == 12, \
        "코호트 계약 위반 (174 = CN 111 / MCI 51 / Dem 12)"

ACT_RICH = ["activity_score", "activity_steps", "activity_rest", "activity_inactive",
            "activity_low", "activity_medium", "activity_high", "activity_daily_movement",
            "activity_average_met", "activity_cal_active"]
ACT_MEAN = ["activity_score_meet_daily_targets", "activity_score_move_every_hour",
            "activity_score_recovery_time", "activity_score_stay_active",
            "activity_score_training_frequency", "activity_score_training_volume",
            "activity_cal_total", "activity_inactivity_alerts", "activity_met_min_high",
            "activity_met_min_medium", "activity_met_min_low", "activity_met_min_inactive",
            "activity_total"]
SLP_RICH = ["sleep_duration", "sleep_efficiency", "sleep_awake", "sleep_deep", "sleep_light",
            "sleep_rem", "sleep_restless", "sleep_onset_latency", "sleep_midpoint_time",
            "sleep_hr_average", "sleep_hr_lowest", "sleep_rmssd", "sleep_breath_average",
            "sleep_score", "sleep_score_deep", "sleep_temperature_deviation"]
SLP_MEAN = ["sleep_score_alignment", "sleep_score_disturbances", "sleep_score_efficiency",
            "sleep_score_latency", "sleep_score_rem", "sleep_score_total", "sleep_total"]
MET_COL, CLS_COL = "CONVERT(activity_met_1min USING utf8)", "CONVERT(activity_class_5min USING utf8)"
HYP_COL, HR_COL, RMS_COL = ("CONVERT(sleep_hypnogram_5min USING utf8)",
                            "CONVERT(sleep_hr_5min USING utf8)",
                            "CONVERT(sleep_rmssd_5min USING utf8)")
# plain 'activity_met_1min' / 'sleep_hypnogram_5min' 등은 값이 '...' 플레이스홀더라 쓸 수 없다.

def _read_src(path, want):
    head = pd.read_csv(path, nrows=0).columns.tolist()
    use = [c for c in want if c in head]
    miss = [c for c in want if c not in head]
    if miss:
        print(f"  주의: {path.name} 에 없는 컬럼 {len(miss)}개 → NaN 처리 {miss[:6]}{'...' if len(miss)>6 else ''}")
    return pd.read_csv(path, usecols=use, low_memory=False)

ACT_WANT = ["EMAIL", "activity_day_start", "activity_non_wear"] + ACT_RICH + ACT_MEAN + [MET_COL, CLS_COL]
SLP_WANT = ["EMAIL", "sleep_bedtime_start", "sleep_bedtime_end"] + SLP_RICH + SLP_MEAN + [HYP_COL, HR_COL, RMS_COL]
act = pd.concat([_read_src(SRC["act_tr"], ACT_WANT), _read_src(SRC["act_va"], ACT_WANT)], ignore_index=True)
slp = pd.concat([_read_src(SRC["slp_tr"], SLP_WANT), _read_src(SRC["slp_va"], SLP_WANT)], ignore_index=True)
act["sid"] = act["EMAIL"].astype(str).str.strip()
slp["sid"] = slp["EMAIL"].astype(str).str.strip()

def _naive(s):   # utc=True 로 바꾸면 04:00 앵커와 시각 특징이 깨진다
    return pd.to_datetime(s.astype(str).str.slice(0, 19), format="%Y-%m-%dT%H:%M:%S", errors="coerce")

act["ts"] = _naive(act["activity_day_start"]); act["date"] = act["ts"].dt.date
slp["ts_start"] = _naive(slp["sleep_bedtime_start"])
slp["ts_end"] = _naive(slp["sleep_bedtime_end"]); slp["date"] = slp["ts_end"].dt.date
_hh = act["ts"].dt.hour.value_counts().to_dict()
print(f"activity_day_start 시각 분포 {_hh} (전부 4시여야 '분 index ↔ 시각' 매핑이 성립)")
assert set(_hh) == {4}, "activity_day_start 가 04:00 이 아닌 행이 있습니다"

slp = (slp.sort_values(["sid", "date", "sleep_duration"], ascending=[True, True, False])
          .drop_duplicates(["sid", "date"], keep="first").reset_index(drop=True))
act = (act.sort_values(["sid", "date"]).drop_duplicates(["sid", "date"], keep="first")
          .reset_index(drop=True))
daily = act.merge(slp.drop(columns=["EMAIL"]), on=["sid", "date"], how="inner")
daily = daily[daily["sid"].isin(set(lab["sid"]))].sort_values(["sid", "date"]).reset_index(drop=True)
_n = daily.groupby("sid").size()
print(f"조인 {len(daily):,}행 / {daily['sid'].nunique()}명 | 관측일수 min {_n.min()} "
      f"median {int(_n.median())} max {_n.max()}")
assert daily["sid"].nunique() == len(lab), "조인 후 사라진 피험자가 있습니다"

SIDS = sorted(lab["sid"].tolist())
lab = lab.set_index("sid").loc[SIDS].reset_index()
y3 = lab["y3"].to_numpy(int)
y  = lab["y"].to_numpy(int)
SID_HASH = {s: hashlib.sha256(s.encode()).hexdigest()[:16] for s in SIDS}   # 원본 이메일은 저장하지 않는다
N_SUBJ, N_POS, N_NEG = len(SIDS), int(y.sum()), int((y == 0).sum())
print(f"\n적재 완료 {time.time()-_t0:.1f}초 | 분석 단위 {N_SUBJ}행 (양성 {N_POS} / 음성 {N_NEG})")
progress("load", 1, 1)

# %%% CELL 3 [code]
# ============================================================================
# 셀 3 — CONVERT(... USING utf8) 시계열 파싱 + 비착용 마스킹 + 야간 계열 지표
# ============================================================================
NONWEAR_MET = 0.5    # Oura 는 비착용 분을 MET 0.1 로 채우고 착용 생리 하한이 0.9 → 포착 0.9939 / 오탐 0.0017
WEAR_GATE   = 0.80   # admitted day 의 착용률 하한
ZERO_RUN_MAX = 60    # MET == 0.0 (no-data sentinel) 연속 > 60분인 날은 제외
ACTIVE_MET  = 1.5    # 활동 = worn & MET >= 1.5
MOD_MET     = 3.0    # 중강도 이상
M10_W, L5_W = 600, 300

def parse_series(s, dtype=np.float64):
    """'/' 구분 문자열 → 배열. 모든 행이 후행 '/' 를 갖고 있으므로 구분자 개수로 세면 안 되고,
    strip 후 **파싱된 길이**로 검사해야 한다."""
    if not isinstance(s, str):
        return np.empty(0, dtype=dtype)
    s = s.strip().strip("/")
    if not s:
        return np.empty(0, dtype=dtype)
    try:
        a = np.fromstring(s, sep="/")
        if a.size:
            return a.astype(dtype)
    except Exception:
        pass
    try:
        return np.asarray(s.split("/"), dtype=dtype)
    except Exception:
        return np.empty(0, dtype=dtype)

def _runs(mask):
    """bool 벡터에서 True 연속 run 길이 배열 (비착용이 run 을 끊는 구현의 기본 블록)."""
    p = np.concatenate(([0], mask.astype(np.int8), [0])); d = np.diff(p)
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)

_t0 = time.time()
_rows = [parse_series(v, np.float32) for v in daily[MET_COL].to_numpy()]
_len = np.array([r.size for r in _rows])
print(f"MET 1분 파싱 {len(_rows):,}일 {time.time()-_t0:.1f}초 | 파싱 길이 {pd.Series(_len).value_counts().head(3).to_dict()}")
assert (_len >= 1440).all(), f"1440점 미만인 날 {(_len < 1440).sum()}개"
MET = np.stack([r[:1440] for r in _rows]).astype(np.float32); del _rows
WORN = MET >= NONWEAR_MET
cov = WORN.mean(1)
ZRUN = np.array([int(_runs(z).max()) if z.any() else 0 for z in (MET == 0.0)])
ADMIT = (cov >= WEAR_GATE) & (ZRUN <= ZERO_RUN_MAX)         # label-free 위생 규칙
METW = np.where(WORN, MET, np.nan).astype(np.float32)
print(f"착용률 median {np.median(cov):.4f} p10 {np.percentile(cov,10):.4f} | "
      f"0.0-sentinel run>{ZERO_RUN_MAX}분 {int((ZRUN>ZERO_RUN_MAX).sum())}일 | "
      f"admitted {ADMIT.mean():.4f} ({int(ADMIT.sum()):,}일)")

_t1 = time.time()
_crows = [parse_series(v, np.float64) for v in daily[CLS_COL].to_numpy()] if CLS_COL in daily else []
if len(_crows):
    _clen = np.array([r.size for r in _crows])
    CLS_OK = _clen == 288                                    # 길이 게이트 필수 (288 인 날이 98.5%)
    CLS = np.stack([(r[:288] if r.size >= 288 else np.full(288, np.nan)) for r in _crows])
    print(f"class_5min 파싱 {time.time()-_t1:.1f}초 | 길이 288 비율 {CLS_OK.mean():.4f} | "
          f"값 {sorted(set(np.unique(CLS[np.isfinite(CLS)]).astype(int).tolist()))}")
else:
    CLS = np.full((len(daily), 288), np.nan); CLS_OK = np.zeros(len(daily), bool)
del _crows

def _night_metrics(hyp_s, hr_s, rms_s):
    """야간 5분 계열 → 하루치 지표. hr/rmssd 의 0 은 미측정 센티널이라 마스킹한다."""
    out = {"hyp_onsetep": np.nan, "hyp_waso_min": np.nan, "hyp_awakenings": np.nan,
           "hyp_deepbout_max": np.nan, "hr_night_min": np.nan, "hr_night_cv": np.nan,
           "rmssd_night_mean": np.nan}
    h = parse_series(hyp_s, np.float64)
    if h.size >= 6:
        h = h[np.isfinite(h)].astype(int)
        asleep = np.isin(h, (1, 2, 3))
        if asleep.any():
            out["hyp_onsetep"] = float(np.argmax(asleep))            # 첫 수면 epoch index (5분 단위)
            awake = h == 4
            out["hyp_waso_min"] = float(awake[int(np.argmax(asleep)):].sum() * 5)
            out["hyp_awakenings"] = float(len(_runs(awake)))
            dr = _runs(h == 1)
            out["hyp_deepbout_max"] = float(dr.max() * 5) if dr.size else 0.0
    for key, raw, pre in (("hr", hr_s, "hr_night"), ("rms", rms_s, "rmssd_night")):
        v = parse_series(raw, np.float64)
        v = v[np.isfinite(v)]
        v = v[v > 0]                                                 # 0 = 미측정
        if v.size >= 6:
            if pre == "hr_night":
                out["hr_night_min"] = float(v.min())
                out["hr_night_cv"] = float(v.std(ddof=0) / v.mean()) if v.mean() > 0 else np.nan
            else:
                out["rmssd_night_mean"] = float(v.mean())
    return out

_t2 = time.time()
_hyp = daily[HYP_COL].to_numpy() if HYP_COL in daily else np.full(len(daily), None)
_hr  = daily[HR_COL].to_numpy()  if HR_COL  in daily else np.full(len(daily), None)
_rms = daily[RMS_COL].to_numpy() if RMS_COL in daily else np.full(len(daily), None)
NIGHT = pd.DataFrame([_night_metrics(a, b, c) for a, b, c in zip(_hyp, _hr, _rms)])
print(f"야간 5분 계열(hypnogram/hr/rmssd) 파싱 {time.time()-_t2:.1f}초 | "
      f"유효 hypnogram {NIGHT['hyp_onsetep'].notna().mean():.3f} | 유효 rmssd {NIGHT['rmssd_night_mean'].notna().mean():.3f}")
progress("parse", 1, 1)
print(f"\n시계열 파싱 총 {time.time()-_t0:.1f}초 | MET {MET.shape} {MET.nbytes/1e6:.0f}MB")

# %%% CELL 4 [code]
# ============================================================================
# 셀 4 — 일별 지표 (1분 MET / 5분 class 스트림에서). 비착용이 run 을 끊는다.
# ============================================================================
def _win_mean_extrema(A, valid, W, chunk=1500):
    """길이 W 슬라이딩 창의 (최대 평균, 최소 평균, 최대창 시작 index, 최소창 시작 index).
    창 안 유효(착용) 분이 80% 미만이면 그 창은 후보에서 제외한다."""
    n, T = A.shape
    mx = np.full(n, np.nan); mn = np.full(n, np.nan)
    ax = np.full(n, np.nan); an = np.full(n, np.nan)
    need = int(0.8 * W)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        a = np.nan_to_num(A[s:e].astype(np.float64), nan=0.0)
        v = valid[s:e].astype(np.float64)
        cs = np.concatenate([np.zeros((e - s, 1)), np.cumsum(a, 1)], 1)
        cv = np.concatenate([np.zeros((e - s, 1)), np.cumsum(v, 1)], 1)
        ssum = cs[:, W:] - cs[:, :-W]
        scnt = cv[:, W:] - cv[:, :-W]
        m = np.where(scnt >= need, ssum / np.maximum(scnt, 1), np.nan)
        ok = np.isfinite(m).any(1)
        if ok.any():
            sub = m[ok]
            mx[s:e][ok] = np.nanmax(sub, 1); mn[s:e][ok] = np.nanmin(sub, 1)
            ax[s:e][ok] = np.nanargmax(np.nan_to_num(sub, nan=-np.inf), 1)
            an[s:e][ok] = np.nanargmin(np.nan_to_num(sub, nan=+np.inf), 1)
    return mx, mn, ax, an

_t0 = time.time()
D = pd.DataFrame({"sid": daily["sid"].to_numpy(), "date": daily["date"].to_numpy(),
                  "admit": ADMIT, "cov": cov})
D["metactive_frac"] = np.where(WORN.sum(1) > 0, ((MET >= ACTIVE_MET) & WORN).sum(1) / np.maximum(WORN.sum(1), 1), np.nan)
D["metmod_min"]     = ((MET >= MOD_MET) & WORN).sum(1).astype(float)
D["metmean"]        = np.nanmean(METW, axis=1)

M10, _, M10_on, _ = _win_mean_extrema(METW, WORN, M10_W)
_, L5, _, L5_on   = _win_mean_extrema(METW, WORN, L5_W)
D["M10"], D["L5"] = M10, L5
D["RAabs"] = M10 - L5                       # ★ 비율 RA 가 아니라 절대 진폭 차 (비율계열은 선행에서 0.400 으로 기각)
D["RAratio"] = (M10 - L5) / np.where((M10 + L5) > 0, M10 + L5, np.nan)   # 탐색 섹션 전용
D["M10_on_h"] = (M10_on + 4 * 60) % 1440 / 60.0                          # 04:00 앵커 → 시각(h)
D["L5_on_h"]  = (L5_on + 4 * 60) % 1440 / 60.0

_cls_valid = np.where(CLS_OK[:, None], CLS, np.nan)
_den = np.nansum(_cls_valid > 0, axis=1).astype(float)                   # class 0 = 비착용 → 분모에서 제외
D["cls_lowfrac"]  = np.where(_den > 0, np.nansum(_cls_valid == 3, axis=1) / np.maximum(_den, 1), np.nan)
D["cls_highfrac"] = np.where(_den > 0, np.nansum(_cls_valid >= 4, axis=1) / np.maximum(_den, 1), np.nan)

# 깨어있는 창 [onset, offset) 과 그 안의 최장 정좌 bout (비착용이 run 을 끊는다)
_z = np.nan_to_num(METW, nan=0.0).astype(np.float64)
_cs = np.concatenate([np.zeros((len(_z), 1)), np.cumsum(_z, 1)], 1)
_roll = ((_cs[:, 30:] - _cs[:, :-30]) / 30.0).astype(np.float32); del _z, _cs
_awake = _roll >= ACTIVE_MET
_longest = np.full(len(D), np.nan); _wake_h = np.full(len(D), np.nan)
for i in range(len(D)):
    idx = np.flatnonzero(_awake[i])
    if idx.size < 2:
        continue
    on, off = int(idx[0]), int(idx[-1]) + 30
    _wake_h[i] = (off - on) / 60.0
    sed = WORN[i, on:off] & (MET[i, on:off] < ACTIVE_MET)
    r = _runs(sed)
    _longest[i] = float(r.max()) if r.size else 0.0
D["longbout_w"] = _longest; D["wake_win_h"] = _wake_h
del _roll, _awake

for c in NIGHT.columns:
    D[c] = NIGHT[c].to_numpy()
for c in (ACT_RICH + ACT_MEAN + SLP_RICH + SLP_MEAN):
    if c in daily.columns:
        D[c] = pd.to_numeric(daily[c], errors="coerce").to_numpy()
for c in ("sleep_light", "sleep_deep", "sleep_rem", "sleep_awake", "sleep_total", "sleep_duration"):
    if c in D.columns:
        D[c + "_min"] = D[c] / 60.0                                       # Oura 는 초 단위 → 분
D["arch_ratio_sleep_deep"] = D.get("sleep_deep", np.nan) / D.get("sleep_total", np.nan).replace(0, np.nan)
D["arch_ratio_sleep_light"] = D.get("sleep_light", np.nan) / D.get("sleep_total", np.nan).replace(0, np.nan)
D["arch_fragmentation"] = D.get("sleep_awake", np.nan) / D.get("sleep_duration", np.nan).replace(0, np.nan)

def _clock_sec(ts):
    return (ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second).astype(float)
D["bedtime_sec"] = _clock_sec(daily["ts_start"]).to_numpy()
D["waketime_sec"] = _clock_sec(daily["ts_end"]).to_numpy()
D["midpoint_sec"] = pd.to_numeric(daily.get("sleep_midpoint_time"), errors="coerce").to_numpy() \
                    if "sleep_midpoint_time" in daily else np.nan
D["nonwear_min"] = (~WORN).sum(1).astype(float)                            # 감사 전용 (특징 금지)
D["day_ord"] = pd.to_datetime(daily["date"]).map(pd.Timestamp.toordinal).to_numpy()

DA = D[D["admit"]].reset_index(drop=True)                                  # admitted day set
_na = DA.groupby("sid").size()
print(f"일별 지표 {time.time()-_t0:.1f}초 | admitted {len(DA):,}일 / {DA['sid'].nunique()}명 | "
      f"일수 min {_na.min()} median {int(_na.median())} max {_na.max()}")
assert DA["sid"].nunique() == N_SUBJ, "admitted 필터로 피험자가 사라졌습니다"
assert _na.min() >= 7, f"admitted 일수 7일 미만인 피험자 존재 (min {_na.min()}) — roll7 계산 불가"
progress("daily", 1, 1)

# %%% CELL 5 [code]
# ============================================================================
# 셀 5 — 피험자 특징 (블록 A~E + wd_core 13 + 감사열). 전부 피험자 내부에서만 계산.
# ============================================================================
def _circ_sd_h(seconds):
    """시계 시각의 원형 표준편차(시간). 23:50 과 00:10 을 12시간 떨어진 값으로 보지 않기 위함."""
    v = np.asarray(seconds, float); v = v[np.isfinite(v)]
    if v.size < 2:
        return np.nan
    th = 2 * np.pi * (v % 86400.0) / 86400.0
    R = float(np.hypot(np.cos(th).mean(), np.sin(th).mean()))
    if R <= 1e-12:
        return np.nan
    return float(np.sqrt(-2.0 * np.log(min(R, 1.0))) * 24.0 / (2 * np.pi))

def _roll7_sd(v):
    v = np.asarray(v, float)
    if np.isfinite(v).sum() < 7:
        return np.nan
    r = pd.Series(v).rolling(7, min_periods=6).std(ddof=0)
    return float(np.nanmean(r.to_numpy()))

def _detr(v):
    v = np.asarray(v, float); m = np.isfinite(v)
    if m.sum() < 7:
        return v
    x = np.arange(v.size, dtype=float)
    b, a = np.polyfit(x[m], v[m], 1)
    return v - (a + b * x)

def _q(v, p):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(np.percentile(v, p)) if v.size else np.nan

def _sd(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(v.std(ddof=0)) if v.size >= 2 else np.nan

def _mn(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else np.nan

def build_features(frame, sids):
    """frame = admitted 일별 표(일부 피험자만 있어도 됨) → 피험자 1행. 다른 피험자에 의존하지 않는다."""
    rows, audit = {}, {}
    for sid, g in frame.groupby("sid", sort=False):
        g = g.sort_values("date")
        low = g["activity_low"].to_numpy(float) if "activity_low" in g else np.full(len(g), np.nan)
        f = {}
        # --- 블록 A: 앵커
        f["low_SD"]  = _sd(low)
        f["low_MED"] = float(np.nanmedian(low)) if np.isfinite(low).any() else np.nan
        # --- 블록 B: 분산축 확장
        f["low_roll7SD"] = _roll7_sd(low)
        f["low_IQR"]     = _q(low, 75) - _q(low, 25)
        f["low_RANGE"]   = (float(np.nanmax(low)) - float(np.nanmin(low))) if np.isfinite(low).any() else np.nan
        # --- 블록 C: 수준축
        f["rest_Q25"]      = _q(g.get("activity_rest", pd.Series(dtype=float)).to_numpy(float), 25)
        f["light_MED"]     = float(np.nanmedian(g["sleep_light_min"])) if "sleep_light_min" in g else np.nan
        f["restless_Q75"]  = _q(g.get("sleep_restless", pd.Series(dtype=float)).to_numpy(float), 75)
        # --- 블록 D: intraday across-day 분산
        f["metactive_frac_SD"] = _sd(g["metactive_frac"].to_numpy(float))
        f["cls_lowfrac_SD"]    = _sd(g["cls_lowfrac"].to_numpy(float))
        f["hyp_onsetep_SD"]    = _sd(g["hyp_onsetep"].to_numpy(float))
        # --- 블록 E: 일주기 (절대 진폭, 비율 아님)
        f["RAabs_SD"] = _sd(g["RAabs"].to_numpy(float))
        f["M10_SD"]   = _sd(g["M10"].to_numpy(float))
        # --- wd_core 13 (DemRankAUC config.py WD_CORE 정의 그대로)
        f["wd_sleep_light__mean"]        = _mn(g.get("sleep_light", pd.Series(dtype=float)))
        f["wd_sleep_restless__mean"]     = _mn(g.get("sleep_restless", pd.Series(dtype=float)))
        f["wd_activity_rest__mean"]      = _mn(g.get("activity_rest", pd.Series(dtype=float)))
        f["wd_activity_low__std"]        = f["low_SD"]
        f["wd_sleep_score_deep__std"]    = _sd(g.get("sleep_score_deep", pd.Series(dtype=float)))
        f["wd_activity_average_met__mean"] = _mn(g.get("activity_average_met", pd.Series(dtype=float)))
        f["wd_sleep_efficiency__mean"]   = _mn(g.get("sleep_efficiency", pd.Series(dtype=float)))
        f["wd_sleep_duration__std"]      = _sd(g.get("sleep_duration", pd.Series(dtype=float)))
        f["wd_activity_steps__mean"]     = _mn(g.get("activity_steps", pd.Series(dtype=float)))
        f["wd_arch_ratio_sleep_deep__mean"] = _mn(g.get("arch_ratio_sleep_deep", pd.Series(dtype=float)))
        f["wd_arch_fragmentation__mean"] = _mn(g.get("arch_fragmentation", pd.Series(dtype=float)))
        f["wd_circ_midpoint__circsd_h"]  = _circ_sd_h(g.get("midpoint_sec", pd.Series(dtype=float)).to_numpy(float))
        f["wd_circ_bedtime__circsd_h"]   = _circ_sd_h(g.get("bedtime_sec", pd.Series(dtype=float)).to_numpy(float))
        # --- 진단/반증 전용
        f["low_roll7SD_detr"] = _roll7_sd(_detr(low))                      # 계절 표류 방어
        wn = low * 1440.0 / np.maximum(g["cov"].to_numpy(float) * 1440.0, 1.0)
        f["lowwn_SD"] = _sd(wn)                                            # 착용시간 정규화
        f["low_CV"]   = f["low_SD"] / abs(f["low_MED"]) if f["low_MED"] not in (0, np.nan) else np.nan
        f["longbout_w_SD"] = _sd(g["longbout_w"].to_numpy(float))
        rows[sid] = f
        # --- 감사 전용 (라벨-프리 교란). 특징 행렬에 절대 들어가지 않는다.
        d = g["day_ord"].to_numpy(float)
        audit[sid] = {"a_n_days": float(len(g)), "a_span_days": float(d.max() - d.min() + 1),
                      "a_gap_days": float(d.max() - d.min() + 1 - len(g)),
                      "a_cov_mean": _mn(g["cov"]), "a_nonwear_mean": _mn(g["nonwear_min"]),
                      "a_start_month": float(pd.Timestamp.fromordinal(int(d.min())).month
                                             + 12 * (pd.Timestamp.fromordinal(int(d.min())).year - 2020)),
                      "a_volume": _mn(low)}
    F = pd.DataFrame.from_dict(rows, orient="index").reindex([s for s in sids if s in rows])
    A = pd.DataFrame.from_dict(audit, orient="index").reindex([s for s in sids if s in rows])
    return F, A

_t0 = time.time()
FEAT, AUDIT = build_features(DA, SIDS)
FEAT = FEAT.loc[SIDS]; AUDIT = AUDIT.loc[SIDS]
assert_no_forbidden(FEAT.columns, "FEAT")
assert not set(FEAT.columns) & set(AUDIT.columns), "감사열이 특징에 섞였습니다"
_nanrate = FEAT.isna().mean().sort_values(ascending=False)
print(f"특징 {FEAT.shape[1]}개 × {FEAT.shape[0]}명 ({time.time()-_t0:.1f}초) | "
      f"결측 상위 {dict(_nanrate.head(3).round(4))}")
assert _nanrate.max() < 0.25, f"결측률 25% 초과 특징: {_nanrate[_nanrate>=0.25].to_dict()}"

BLOCK_A = ("low_SD", "low_MED")
BLOCK_B = BLOCK_A + ("low_roll7SD", "low_IQR", "low_RANGE")
BLOCK_C = BLOCK_B + ("rest_Q25", "light_MED", "restless_Q75")
BLOCK_D = BLOCK_C + ("metactive_frac_SD", "cls_lowfrac_SD", "hyp_onsetep_SD")
BLOCK_E = BLOCK_D + ("RAabs_SD", "M10_SD")
WD_CORE = ("wd_sleep_light__mean", "wd_sleep_restless__mean", "wd_activity_rest__mean",
           "wd_activity_low__std", "wd_sleep_score_deep__std", "wd_activity_average_met__mean",
           "wd_sleep_efficiency__mean", "wd_sleep_duration__std", "wd_activity_steps__mean",
           "wd_arch_ratio_sleep_deep__mean", "wd_arch_fragmentation__mean",
           "wd_circ_midpoint__circsd_h", "wd_circ_bedtime__circsd_h")
BLOCKS = {"A": BLOCK_A, "B": BLOCK_B, "C": BLOCK_C, "D": BLOCK_D, "E": BLOCK_E}
for nm, blk in list(BLOCKS.items()) + [("WD_CORE", WD_CORE)]:
    miss = [c for c in blk if c not in FEAT.columns]
    assert not miss, f"블록 {nm} 결측 컬럼 {miss}"
print("블록 크기: " + " | ".join(f"{k} {len(v)}" for k, v in BLOCKS.items()) + f" | WD_CORE {len(WD_CORE)}")

FEAT_FP = hashlib.sha256(
    (",".join(FEAT.columns) + "|" + np.round(FEAT.to_numpy(float), 8).tobytes().hex()[:4096]).encode()
).hexdigest()[:16]
print(f"특징 지문 {FEAT_FP}")

# 라벨-블라인드 QC 규칙 (DemScreen data.py QUALITY_RULES 와 동일). 라벨을 전혀 읽지 않는다.
_steps = DA.groupby("sid")["activity_steps"].mean().reindex(SIDS) if "activity_steps" in DA else pd.Series(index=SIDS, dtype=float)
_ndays = DA.groupby("sid").size().reindex(SIDS)
QC_KEEP = ((_steps.fillna(0) <= 25000) & (_ndays >= 7)).to_numpy()
_drop = [SID_HASH[s] for s, k in zip(SIDS, QC_KEEP) if not k]
print(f"QC 코호트: {int(QC_KEEP.sum())}명 유지 / 제외 {len(_drop)}명 {_drop} "
      f"(양성 {int(y[QC_KEEP].sum())} / 음성 {int((y[QC_KEEP]==0).sum())})")
progress("features", 1, 1)

# %%% CELL 6 [code]
# ============================================================================
# 셀 6 — 누수 자기검증 + 라벨-프리 교란 감사
# ============================================================================
# 특징이 다른 피험자에 의존한다면(예: 전체 코호트 분위수로 정규화) 부분집합 재구축 결과가 달라진다.
_rng = np.random.default_rng(SEED)
_sub = sorted(_rng.choice(SIDS, size=min(90, N_SUBJ), replace=False).tolist())
_F2, _ = build_features(DA[DA["sid"].isin(_sub)], _sub)
_a = FEAT.loc[_sub].to_numpy(float); _b = _F2.loc[_sub].to_numpy(float)
_same = np.array_equal(np.nan_to_num(_a, nan=-9e99), np.nan_to_num(_b, nan=-9e99))
_maxdiff = float(np.nanmax(np.abs(_a - _b))) if np.isfinite(_a - _b).any() else 0.0
print(f"누수 자기검증: {len(_sub)}명 부분집합 재구축 → 전 컬럼 비트 동일 {_same} (최대 |차| {_maxdiff:.3e})")
assert _same, "특징이 다른 피험자에 의존합니다 — 피험자 단위 독립 위반"

# 라벨-프리 교란 감사 (여기서는 라벨을 보지만, 특징 정의는 이미 동결됐고 아무것도 고르지 않는다)
from scipy import stats as _st
def auc_of(score, target):
    s = np.asarray(score, float); t = np.asarray(target, int)
    m = np.isfinite(s)
    if m.sum() < 4 or len(set(t[m].tolist())) < 2:
        return np.nan
    r = _st.rankdata(s[m]); n1 = int(t[m].sum()); n0 = int(m.sum() - n1)
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((r[t[m] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

UNI = pd.DataFrame({
    "feature": FEAT.columns,
    "auc_signed": [auc_of(FEAT[c], y) for c in FEAT.columns]})
UNI["auc_dirfree"] = UNI["auc_signed"].apply(lambda a: max(a, 1 - a) if np.isfinite(a) else np.nan)
UNI["auc_CN_vs_Dem"] = [auc_of(FEAT[c][y3 != 1], (y3[y3 != 1] == 2).astype(int)) for c in FEAT.columns]
UNI["auc_MCI_vs_Dem"] = [auc_of(FEAT[c][y3 != 0], (y3[y3 != 0] == 2).astype(int)) for c in FEAT.columns]
UNI = UNI.sort_values("auc_dirfree", ascending=False).reset_index(drop=True)
print("\n전코호트 단변량 (기술 통계 — out-of-sample 아님, 모델 성능으로 인용 금지)")
print(UNI.head(12).round(4).to_string(index=False))

CONF = []
for c in FEAT.columns:
    row = {"feature": c}
    for a in AUDIT.columns:
        row[a] = float(_st.spearmanr(FEAT[c], AUDIT[a], nan_policy="omit").statistic)
    CONF.append(row)
CONF = pd.DataFrame(CONF)
print("\n라벨-프리 교란과의 |Spearman rho| 최대치")
_cm = CONF.set_index("feature").abs().max(1).sort_values(ascending=False)
print(_cm.head(8).round(3).to_string())
AUD_AUC = {a: auc_of(AUDIT[a], y) for a in AUDIT.columns}
print("\n교란 변수 자체의 Dem 단변량 AUC (0.5 근처여야 안전): "
      + " | ".join(f"{k} {v:.3f}" for k, v in AUD_AUC.items()))
_enr = pd.crosstab(AUDIT["a_start_month"], pd.Series(y3, index=SIDS).map({0: "CN", 1: "MCI", 2: "Dem"}))
print(f"\n등록월 × 클래스 (모집시기 교란 — 날짜 파생 특징을 전면 금지한 이유)\n{_enr.to_string()}")
progress("audit", 1, 1)

# %%% CELL 7 [markdown]
## EDA — 신호가 어디에 있는지 먼저 눈으로 확인한다

아래 세 그림은 **모델링 전에** 이 코호트에서 무엇이 실제로 다른지를 보여준다.

- **그림 1 (cohort)** — 클래스 구성, 관측일수, **등록월 분포**, 착용률. 등록월 패널이 중요하다:
  Dem 12명이 특정 시기에 몰려 있으면 계절 표류(저강도 활동이 10월→1월 −17%)가 신호를 흉내낼 수 있다.
  그래서 이 노트북은 날짜 파생 특징을 **전면 금지**하고, 대신 `low_roll7SD_detr`(피험자 내 선형추세 제거)를
  반증 진단으로 갖고 있다.
- **그림 2 (signal)** — 일별 저강도 활동시간 궤적. Dem 12명의 선이 CN/MCI 밴드보다 **평평한지**가 핵심이다.
  이 과제의 신호는 "활동량이 적다"가 아니라 "**하루하루가 서로 비슷하다**"(루틴 경직)이다.
  자기정규화(CV = SD/mean)를 쓰면 이 축이 0.845 → 0.509 로 무너지므로 **절대 단위(분)** 를 유지한다.
- **그림 3 (confound)** — 특징과 라벨-프리 교란(관측일수·span·gap·착용률·비착용·등록월·활동량)의 상관.
  `low_SD` 가 착용시간이나 순응도의 대리변수가 아니라는 것을 코드 안에서 다시 확인한다.

표로 출력되는 단변량 AUC 는 **전코호트 기술 통계**이며 out-of-sample 이 아니다. 모델 성능으로 인용하면 안 된다.

# %%% CELL 8 [code]
# ============================================================================
# 셀 8 — EDA 그림 1~3 (축·범례는 전부 ASCII: Colab 한글 글리프 깨짐 회피, 서사는 마크다운이 담당)
# ============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 140, "font.size": 9,
                     "axes.grid": True, "grid.alpha": .25, "axes.axisbelow": True})
CCOL = {"CN": "#2166AC", "MCI": "#F4A582", "Dem": "#B2182B"}      # 색맹 안전 3색
def _box(a, data, labels):
    """matplotlib 3.9+ 는 tick_labels, 그 이전은 labels — Colab 버전 차이를 흡수한다."""
    try:
        return a.boxplot(data, tick_labels=labels)
    except TypeError:
        return a.boxplot(data, labels=labels)
GRP = pd.Series(y3, index=SIDS).map({0: "CN", 1: "MCI", 2: "Dem"})
FIGS = {}
def _save(fig, name):
    p = OUT_DIR / "figs" / name
    fig.tight_layout(); fig.savefig(p); plt.close(fig); FIGS[name] = str(p)
    print(f"  그림 저장 {p}")

# ---- 그림 1: cohort
fig, ax = plt.subplots(2, 3, figsize=(16, 8))
_cnt = GRP.value_counts().reindex(["CN", "MCI", "Dem"]).fillna(0)
ax[0,0].bar(_cnt.index, _cnt.values, color=[CCOL[k] for k in _cnt.index])
for i, v in enumerate(_cnt.values):
    ax[0,0].text(i, v, f"{int(v)}", ha="center", va="bottom")
ax[0,0].set_title(f"Cohort (positive rate {y.mean():.4f})"); ax[0,0].set_ylabel("subjects")
_nd = AUDIT["a_n_days"]
_box(ax[0,1], [_nd[GRP == k].values for k in ["CN", "MCI", "Dem"]], ["CN", "MCI", "Dem"])
for i, k in enumerate(["CN", "MCI", "Dem"]):
    v = _nd[GRP == k].values
    ax[0,1].scatter(np.full(v.size, i + 1) + np.random.uniform(-.08, .08, v.size), v, s=8, alpha=.5, color=CCOL[k])
_mw = _st.mannwhitneyu(_nd[GRP == "Dem"], _nd[GRP != "Dem"]).pvalue
ax[0,1].set_title(f"Admitted days (Dem vs rest MWU p={_mw:.3f})"); ax[0,1].set_ylabel("days")
_ct = pd.crosstab(AUDIT["a_start_month"], GRP).reindex(columns=["CN", "MCI", "Dem"]).fillna(0)
_bot = np.zeros(len(_ct))
for k in ["CN", "MCI", "Dem"]:
    ax[0,2].bar(_ct.index.astype(int).astype(str), _ct[k].values, bottom=_bot, color=CCOL[k], label=k)
    _bot += _ct[k].values
ax[0,2].legend(fontsize=7); ax[0,2].set_title("Enrollment month x class (CONFOUND)")
ax[0,2].set_xlabel("month index (2020-01 = 1)")
ax[1,0].hist(DA["cov"].values, bins=40, color="#777"); ax[1,0].axvline(WEAR_GATE, color="r", ls="--")
ax[1,0].set_title(f"Daily wear fraction (gate {WEAR_GATE})"); ax[1,0].set_yscale("log")
_box(ax[1,1], [AUDIT["a_span_days"][GRP == k].values for k in ["CN", "MCI", "Dem"]], ["CN", "MCI", "Dem"])
ax[1,1].set_title("Calendar span (days)")
_av = list(AUD_AUC.values())
ax[1,2].bar(range(len(AUD_AUC)), _av,
            color=["#B2182B" if abs(v - .5) > .10 else "#999" for v in _av])
for i, v in enumerate(_av):
    ax[1,2].text(i, v, f"{v:.2f}", ha="center", va="bottom" if v >= .5 else "top", fontsize=6.5)
ax[1,2].set_xticks(range(len(AUD_AUC))); ax[1,2].set_xticklabels(list(AUD_AUC), rotation=45, ha="right", fontsize=7)
ax[1,2].axhline(.5, color="k", ls="--"); ax[1,2].set_ylim(0, 1)
ax[1,2].set_title("Dem AUC of label-free nuisances (want ~0.5)\nred = |AUC-0.5| > 0.10, i.e. NOT null", fontsize=8.5)
fig.suptitle(f"Fig 1  Cohort and label-free audit  |  {EXPERIMENT}  |  run {RUN_ID}", fontsize=11)
_save(fig, "fig1_cohort.png")

# ---- 그림 2: signal
fig, ax = plt.subplots(2, 3, figsize=(16, 8))
_piv = DA.pivot_table(index="sid", columns=DA.groupby("sid").cumcount(), values="activity_low")
_piv = _piv.reindex(SIDS).iloc[:, :60]
# 헤드라인 특징이 실제로 재는 양 = 7일 rolling SD. 그 궤적을 그룹별로 그린다.
_rs = _piv.T.rolling(7, min_periods=6).std(ddof=0).T
for k in ["CN", "MCI", "Dem"]:
    sub = _rs[GRP == k]
    if not len(sub):
        continue
    ax[0,0].fill_between(sub.columns, sub.quantile(.25), sub.quantile(.75), alpha=.18, color=CCOL[k])
    ax[0,0].plot(sub.columns, sub.median(), color=CCOL[k], lw=2.0, label=f"{k} median (n={len(sub)})")
ax[0,0].legend(fontsize=7)
ax[0,0].set_title("7-day rolling SD of activity_low (median + IQR band):\n"
                  "Dem sits BELOW = less day-to-day swing = routine rigidity", fontsize=9)
ax[0,0].set_xlabel("day index"); ax[0,0].set_ylabel("rolling 7-day SD (min)")
for j, feat in enumerate(["low_SD", "low_roll7SD"]):
    a = ax[0, 1 + j]
    parts = [FEAT[feat][GRP == k].dropna().values for k in ["CN", "MCI", "Dem"]]
    a.violinplot(parts, showmedians=True)
    for i, k in enumerate(["CN", "MCI", "Dem"]):
        v = FEAT[feat][GRP == k].dropna().values
        a.scatter(np.full(v.size, i + 1) + np.random.uniform(-.07, .07, v.size), v, s=10, alpha=.6, color=CCOL[k])
    a.set_xticks([1, 2, 3]); a.set_xticklabels(["CN", "MCI", "Dem"])
    a.set_title(f"{feat}  (dir-free AUC {UNI.set_index('feature').loc[feat,'auc_dirfree']:.3f})")
for k in ["CN", "MCI", "Dem"]:
    m = (GRP == k).to_numpy()
    ax[1,0].scatter(FEAT["low_MED"][m], FEAT["low_SD"][m], s=22, alpha=.75, color=CCOL[k], label=k,
                    edgecolor="k" if k == "Dem" else "none", linewidth=.5)
ax[1,0].legend(fontsize=7); ax[1,0].set_xlabel("low_MED (level, min/day)"); ax[1,0].set_ylabel("low_SD (dispersion)")
ax[1,0].set_title("Level is NON-monotone (CN<MCI>Dem); dispersion orders all three")
_top = UNI.head(18).iloc[::-1]
ax[1,1].barh(range(len(_top)), _top["auc_dirfree"], color="#4477AA")
ax[1,1].set_yticks(range(len(_top))); ax[1,1].set_yticklabels(_top["feature"], fontsize=6.5)
ax[1,1].axvline(.5, color="k", ls="--"); ax[1,1].set_xlim(.45, 1.0)
ax[1,1].set_title("Top univariate (WHOLE-COHORT, descriptive only)")
_sub = UNI.set_index("feature").loc[[c for c in BLOCK_E]].copy()
for _c in ("auc_CN_vs_Dem", "auc_MCI_vs_Dem"):
    _sub[_c] = _sub[_c].apply(lambda a: max(a, 1 - a) if np.isfinite(a) else np.nan)
ax[1,2].scatter(_sub["auc_MCI_vs_Dem"], _sub["auc_CN_vs_Dem"], s=30, color="#B2182B", zorder=3)
for f_ in [c for c in BLOCK_B if c in _sub.index]:          # 앵커 블록만 라벨 (겹침 방지)
    ax[1,2].annotate(f_, (_sub.loc[f_, "auc_MCI_vs_Dem"], _sub.loc[f_, "auc_CN_vs_Dem"]),
                     fontsize=6, xytext=(3, 3), textcoords="offset points")
ax[1,2].plot([.4, 1], [.4, 1], "k--", lw=.7); ax[1,2].set_xlim(.4, 1.02); ax[1,2].set_ylim(.4, 1.02)
ax[1,2].set_xlabel("AUC MCI vs Dem (dir-free)"); ax[1,2].set_ylabel("AUC CN vs Dem (dir-free)")
ax[1,2].set_title("Where the separation lives (block A-E)\nabove the diagonal = easier against CN than MCI", fontsize=9)
fig.suptitle("Fig 2  Signal structure (descriptive, in-sample)", fontsize=11)
_save(fig, "fig2_signal.png")

# ---- 그림 3: confound
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
_H = CONF.set_index("feature").loc[list(BLOCK_E) + ["low_roll7SD_detr", "lowwn_SD", "low_CV"]].abs()
im = ax[0].imshow(_H.values, cmap="magma_r", vmin=0, vmax=.6, aspect="auto")
ax[0].set_xticks(range(_H.shape[1])); ax[0].set_xticklabels(_H.columns, rotation=45, ha="right", fontsize=7)
ax[0].set_yticks(range(_H.shape[0])); ax[0].set_yticklabels(_H.index, fontsize=7)
for i in range(_H.shape[0]):
    for j in range(_H.shape[1]):
        if _H.values[i, j] > .30:
            ax[0].text(j, i, f"{_H.values[i,j]:.2f}", ha="center", va="center", fontsize=6, color="w")
fig.colorbar(im, ax=ax[0], shrink=.85); ax[0].set_title("|Spearman rho| feature x label-free nuisance")
for k in ["CN", "MCI", "Dem"]:
    m = (GRP == k).to_numpy()
    ax[1].scatter(AUDIT["a_volume"][m], FEAT["low_SD"][m], s=22, alpha=.75, color=CCOL[k], label=k)
_rho = _st.spearmanr(AUDIT["a_volume"], FEAT["low_SD"], nan_policy="omit").statistic
ax[1].legend(fontsize=7); ax[1].set_xlabel("mean activity_low (volume)"); ax[1].set_ylabel("low_SD")
ax[1].set_title(f"Mean-variance coupling rho={_rho:.3f} (partial, not total)")
fig.suptitle("Fig 3  Label-free confound audit", fontsize=11)
_save(fig, "fig3_confound.png")
progress("eda", 1, 1)

# %%% CELL 9 [code]
# ============================================================================
# 셀 9 — fold 생성기 + 전체 outer fold 사전 전수검사 (학습 전에 실행)
# ============================================================================
from sklearn.model_selection import StratifiedKFold

def safe_k(strat, requested, min_per_fold=2):
    """가장 작은 층이 fold 마다 min_per_fold 개를 채울 수 있는 최대 k."""
    m = int(pd.Series(strat).value_counts().min())
    return int(max(2, min(requested, m // max(1, min_per_fold))))

def outer_folds(strat, n_rows):
    """(repeat_id, seed_group, fold, train_idx, test_idx) 를 생성. 층화는 y3(3-class)."""
    k = safe_k(strat, OUTER_K, 2)
    r_id = 0
    for gi, base in enumerate(SEED_GROUPS):
        for r in range(R_PER_GROUP):
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=base + r)
            for f, (tr, te) in enumerate(skf.split(np.zeros(n_rows), strat)):
                yield r_id, base, f, tr, te
            r_id += 1

def inner_splits(strat_tr, r_id, f):
    """outer-train 블록에 **상대적인** 인덱스만 만든다 → outer-test 를 주소로도 가리킬 수 없다."""
    k = safe_k(strat_tr, INNER_K, 1)
    out = []
    for rr in range(INNER_REPS):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=7000 + 37 * r_id + f + 1000 * rr)
        out.append(list(skf.split(np.zeros(len(strat_tr)), strat_tr)))
    return out

_t0 = time.time(); _rows = []
_cover_ok = True
_by_rep = {}
for r_id, base, f, tr, te in outer_folds(y3, N_SUBJ):
    assert len(set(tr) & set(te)) == 0, f"repeat {r_id} fold {f}: train∩test ≠ ∅"
    assert len(tr) + len(te) == N_SUBJ
    _by_rep.setdefault(r_id, []).extend(te.tolist())
    _rows.append({"repeat": r_id, "seed_group": base, "fold": f, "n_test": len(te),
                  "test_pos": int(y[te].sum()), "test_CN": int((y3[te] == 0).sum()),
                  "test_MCI": int((y3[te] == 1).sum()), "train_pos": int(y[tr].sum())})
for r_id, cov_idx in _by_rep.items():
    if sorted(cov_idx) != list(range(N_SUBJ)):
        _cover_ok = False
FOLD_AUDIT = pd.DataFrame(_rows)
print(f"outer fold 전수검사 {len(FOLD_AUDIT)}개 ({time.time()-_t0:.1f}초) | "
      f"repeat {FOLD_AUDIT['repeat'].nunique()} | test 크기 {FOLD_AUDIT['n_test'].min()}~{FOLD_AUDIT['n_test'].max()}")
print(f"  test fold 당 양성 {FOLD_AUDIT['test_pos'].min()}~{FOLD_AUDIT['test_pos'].max()} | "
      f"train 양성 {FOLD_AUDIT['train_pos'].min()}~{FOLD_AUDIT['train_pos'].max()} | "
      f"repeat 별 코호트 정확히 1회 분할 {_cover_ok}")
assert _cover_ok, "어떤 repeat 이 코호트를 정확히 한 번 덮지 않습니다"
assert FOLD_AUDIT["test_pos"].min() >= 1, "양성이 0인 test fold 가 있습니다"
if not QUICK:
    assert FOLD_AUDIT["test_pos"].min() >= 2, "양성 2명 미만 test fold — outer_k 를 낮추세요"
progress("folds", 1, 1)

# %%% CELL 10 [code]
# ============================================================================
# 셀 10 — nested CV 엔진 (전부 인라인). 선택은 오직 inner 에서, outer-test 는 1회 점수화.
# ============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

class Prep:
    """fold-local 전처리: nanmedian 대치 → 1/99 winsorize → 표준화. **train 행에서만** 적합한다.
    전 컬럼에 한 번 적합해두면 후보들은 컬럼만 골라 쓰면 되므로 후보마다 재적합할 필요가 없다."""
    def fit(self, X):
        A = np.asarray(X, float)
        self.med = np.nanmedian(A, 0)
        self.med = np.where(np.isfinite(self.med), self.med, 0.0)
        B = np.where(np.isfinite(A), A, self.med)
        self.lo = np.percentile(B, 1, axis=0); self.hi = np.percentile(B, 99, axis=0)
        B = np.clip(B, self.lo, self.hi)
        self.mu = B.mean(0); self.sd = B.std(0, ddof=0)
        self.sd = np.where(self.sd > 1e-12, self.sd, 1.0)
        self._fp = hashlib.sha256(np.round(np.r_[self.med, self.mu, self.sd], 10).tobytes()).hexdigest()[:12]
        return self
    def transform(self, X):
        A = np.asarray(X, float)
        return (np.clip(np.where(np.isfinite(A), A, self.med), self.lo, self.hi) - self.mu) / self.sd

def rank_norm(v):
    v = np.asarray(v, float); n = v.size
    if n == 0:
        return v
    ok = np.isfinite(v); out = np.full(n, .5)
    if ok.sum() >= 2:
        out[ok] = (_st.rankdata(v[ok]) - .5) / ok.sum()
    return out

def make_cands(cols_index):
    """사전등록 후보 목록. arm 은 이 목록의 부분집합만 본다(중복 적합 방지)."""
    C = {}
    for b, cols in BLOCKS.items():
        for c in C_GRID:
            C[("lr", b, c)] = (cols, "lr", c)
    for c in C_GRID:
        C[("lr", "WD", c)] = (WD_CORE, "lr", c)
    C[("hgb", "WD", None)] = (WD_CORE, "hgb", None)
    C[("fix", "roll7", None)] = (("low_roll7SD",), "fix", None)
    return C

CANDS = make_cands(FEAT.columns)
ARM_CANDS = {
    "A0_fixed_roll7SD": [("fix", "roll7", None)],
    "A1_anchor_A":      [("lr", "A", c) for c in C_GRID],
    "A2_blockselect":   [("lr", b, c) for b in BLOCKS for c in C_GRID],
    "A3_wdcore13":      [("lr", "WD", c) for c in C_GRID] + [("hgb", "WD", None)],
    "A4_intraday_D":    [("lr", "D", c) for c in C_GRID],
}
PREREG_ARMS = list(ARM_CANDS) + ["A5_rankmean"]
EFFECTIVE_K = len(PREREG_ARMS)
print(f"사전등록 arm {EFFECTIVE_K}개 | 고유 후보 {len(CANDS)}개 "
      f"(arm 별 {{{', '.join(f'{k}:{len(v)}' for k,v in ARM_CANDS.items())}}})")

def _fit_score(kind, c, Xtr, ytr, Xte):
    if kind == "fix":
        return -Xte[:, 0]                      # 방향 사전 고정: 분산이 낮을수록 Dem
    if len(set(ytr.tolist())) < 2:
        return np.full(len(Xte), .5)
    if kind == "lr":
        m = LogisticRegression(penalty="l2", C=c, class_weight="balanced",
                               solver="lbfgs", max_iter=5000)
    else:
        m = HistGradientBoostingClassifier(max_depth=2, max_iter=150, learning_rate=0.05,
                                           l2_regularization=1.0, min_samples_leaf=10,
                                           random_state=SEED)
    try:
        if kind == "hgb":
            w = np.where(ytr == 1, (ytr == 0).sum() / max(1, (ytr == 1).sum()), 1.0)
            m.fit(Xtr, ytr, sample_weight=w)
        else:
            m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    except Exception:
        return np.full(len(Xte), .5)

def _youden_apply(inner_score, ytr, test_score):
    """inner OOF 에서 Youden J 로 임계값을 정하고, 그 임계값의 **분위 위치**를 outer-test 점수에 적용한다.
    outer-test 를 보고 임계값을 고르지 않으므로 threshold leakage 가 없다."""
    io = np.asarray(inner_score, float)
    n = len(test_score)
    if len(set(ytr.tolist())) < 2 or np.unique(io).size < 2:
        return np.full(n, np.nan)
    cand = np.unique(io)
    j = [((io >= c)[ytr == 1].mean() - (io >= c)[ytr == 0].mean(), c) for c in cand]
    t_in = max(j)[1]
    q = float((io < t_in).mean())                      # 학습 fold 에서 음성으로 분류되는 비율
    return (rank_norm(test_score) >= q).astype(float)

def _pick(scores, cand_keys):
    """inner AUC 최대 → TOL 이내 동률이면 (특징 수 ↓, C ↓, lr 우선)."""
    best = max((scores.get(k, .0) for k in cand_keys), default=.0)
    tied = [k for k in cand_keys if scores.get(k, .0) >= best - TOL]
    def _rank(k):
        cols, kind, c = CANDS[k]
        return (len(cols), 0 if kind == "fix" else (1 if kind == "lr" else 2), c if c is not None else 9e9)
    return sorted(tied, key=_rank)[0]

def run_nested(Xdf, y3v, arms=None, cands=None, n_groups=None, r_per=None,
               feat_cols=None, tag="main", log_every=25):
    """반복 nested CV. 반환: per-repeat AUC, OOF 점수 행렬, 선택 이력, fold 지표."""
    arms = arms or ARM_CANDS
    cand_keys = sorted({k for v in arms.values() for k in v}) if cands is None else cands
    cols = list(feat_cols or Xdf.columns)
    X = Xdf[cols].to_numpy(float)
    ci = {c: i for i, c in enumerate(cols)}
    yv = (np.asarray(y3v) == 2).astype(int)
    n = len(yv)
    groups = SEED_GROUPS[:n_groups] if n_groups else SEED_GROUPS
    reps = r_per if r_per is not None else R_PER_GROUP
    n_rep = len(groups) * reps
    all_arms = list(arms) + (["A5_rankmean"] if len(arms) > 1 else [])
    OOF = {a: np.full((n_rep, n), np.nan) for a in all_arms}
    THR = {a: np.full((n_rep, n), np.nan) for a in all_arms}
    sel, foldrec = [], []
    k_out = safe_k(y3v, OUTER_K, 2)
    total = n_rep * k_out
    done = 0
    r_id = 0
    for base in groups:
        for r in range(reps):
            skf = StratifiedKFold(n_splits=k_out, shuffle=True, random_state=base + r)
            for f, (tr, te) in enumerate(skf.split(np.zeros(n), y3v)):
                assert not (set(tr) & set(te))
                prep = Prep().fit(X[tr]); Xtr, Xte = prep.transform(X[tr]), prep.transform(X[te])
                ytr = yv[tr]
                isp = inner_splits(np.asarray(y3v)[tr], r_id, f)
                inner_oof = {k: np.zeros(len(tr)) for k in cand_keys}
                for splits in isp:
                    for itr, iva in splits:
                        p2 = Prep().fit(Xtr[itr]); A, B = p2.transform(Xtr[itr]), p2.transform(Xtr[iva])
                        for k in cand_keys:
                            cl, kind, c = CANDS[k]
                            j = [ci[x] for x in cl]
                            inner_oof[k][iva] += rank_norm(_fit_score(kind, c, A[:, j], ytr[itr], B[:, j])) / len(isp)
                iauc = {k: (roc_auc_score(ytr, inner_oof[k]) if len(set(ytr.tolist())) > 1 else .5)
                        for k in cand_keys}
                fold_scores, inner_pick = {}, {}
                for a, keys in arms.items():
                    kbest = _pick(iauc, keys)
                    inner_pick[a] = kbest
                    cl, kind, c = CANDS[kbest]
                    j = [ci[x] for x in cl]
                    s = _fit_score(kind, c, Xtr[:, j], ytr, Xte[:, j])
                    OOF[a][r_id, te] = s
                    fold_scores[a] = s
                    THR[a][r_id, te] = _youden_apply(inner_oof[kbest], ytr, s)
                    sel.append({"tag": tag, "repeat": r_id, "fold": f, "arm": a, "cand": str(kbest),
                                "block": kbest[1], "C": kbest[2], "kind": kbest[0],
                                "n_feat": len(cl), "inner_auc": iauc[kbest], "prep_fp": prep._fp})
                if len(arms) > 1:
                    _mem = [a for a in arms if a != "A0_fixed_roll7SD"]
                    rm = np.mean([rank_norm(fold_scores[a]) for a in _mem], axis=0)
                    OOF["A5_rankmean"][r_id, te] = rm
                    # A5 의 운영 임계값도 inner 에서 온다: 각 arm 이 inner 에서 고른 후보의 inner OOF 를
                    # 같은 방식으로 rank-mean 한 뒤 Youden 을 적용한다 (임계값 누수 방지).
                    io_rm = np.mean([rank_norm(inner_oof[inner_pick[a]]) for a in _mem], axis=0)
                    THR["A5_rankmean"][r_id, te] = _youden_apply(io_rm, ytr, rm)
                for a in all_arms:
                    ss = OOF[a][r_id, te]
                    foldrec.append({"tag": tag, "repeat": r_id, "fold": f, "arm": a,
                                    "auc": roc_auc_score(yv[te], ss) if len(set(yv[te].tolist())) > 1 else np.nan})
                done += 1
                if log_every:
                    tick(f"nested:{tag}", done, total, every=log_every)
            r_id += 1
    per_rep = {a: np.array([roc_auc_score(yv, OOF[a][i]) if np.isfinite(OOF[a][i]).all() else np.nan
                            for i in range(n_rep)]) for a in all_arms}
    subj = {a: np.mean([rank_norm(OOF[a][i]) for i in range(n_rep)], axis=0) for a in all_arms}
    return {"per_repeat": per_rep, "oof": OOF, "subject_score": subj, "thr": THR,
            "selection": pd.DataFrame(sel), "fold": pd.DataFrame(foldrec), "y": yv, "n_rep": n_rep}

# %%% CELL 11 [code]
# ============================================================================
# 셀 11 — MAIN: 사전등록 arm × {FULL 174, QC 172}. 헤드라인은 FULL, QC 는 병기(더 좋은 쪽을 고르지 않는다).
# ============================================================================
COHORTS = {"FULL": np.ones(N_SUBJ, bool), "QC": QC_KEEP}
RES = {}
for cname, mask in COHORTS.items():
    print(f"\n=== 코호트 {cname}: {int(mask.sum())}명 (양성 {int(y[mask].sum())} / 음성 {int((y[mask]==0).sum())}) ===")
    _t = time.time()
    RES[cname] = run_nested(FEAT.loc[np.array(SIDS)[mask]], y3[mask], tag=cname)
    RES[cname]["mask"] = mask
    print(f"  소요 {(time.time()-_t)/60:.1f}분")

def hanley_se(A, n1, n0):
    Q1 = A / (2 - A); Q2 = 2 * A * A / (1 + A)
    return float(np.sqrt(max(A * (1 - A) + (n1 - 1) * (Q1 - A * A) + (n0 - 1) * (Q2 - A * A), 0) / (n1 * n0)))

def logit_ci(A, se, z=1.96):
    A = min(max(A, 1e-6), 1 - 1e-6)
    l = np.log(A / (1 - A)); sl = se / (A * (1 - A))
    lo, hi = l - z * sl, l + z * sl
    return float(1 / (1 + np.exp(-lo))), float(1 / (1 + np.exp(-hi)))

def threshold_metrics(res, arm):
    yv = res["y"]; T = res["thr"][arm]
    rows = []
    for i in range(res["n_rep"]):
        d = T[i]
        if not np.isfinite(d).all():
            continue
        d = d.astype(int)
        tp = int(((d == 1) & (yv == 1)).sum()); fp = int(((d == 1) & (yv == 0)).sum())
        fn = int(((d == 0) & (yv == 1)).sum()); tn = int(((d == 0) & (yv == 0)).sum())
        sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1); npv = tn / max(tn + fn, 1)
        f1 = 2 * ppv * sens / max(ppv + sens, 1e-9)
        den = math.sqrt(max((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn), 1))
        rows.append({"tn": tn, "fp": fp, "fn": fn, "tp": tp, "sensitivity": sens, "specificity": spec,
                     "ppv": ppv, "npv": npv, "f1": f1, "balanced_acc": (sens + spec) / 2,
                     "accuracy": (tp + tn) / len(yv), "mcc": (tp * tn - fp * fn) / den})
    return pd.DataFrame(rows).mean().to_dict() if rows else {}

from sklearn.metrics import average_precision_score
def summarize(res, cname):
    yv = res["y"]; out = []
    y3c = y3[res["mask"]]
    for a in res["per_repeat"]:
        pr = res["per_repeat"][a]; s = res["subject_score"][a]
        A = roc_auc_score(yv, s)
        n1, n0 = int(yv.sum()), int((yv == 0).sum())
        se = hanley_se(A, n1, n0)
        lo_l, hi_l = logit_ci(A, se)
        a_cn = auc_of(s[y3c != 1], (y3c[y3c != 1] == 2).astype(int))
        a_mci = auc_of(s[y3c != 0], (y3c[y3c != 0] == 2).astype(int))
        n_cn, n_mci = int((y3c == 0).sum()), int((y3c == 1).sum())
        ident = abs(A - (n_cn * a_cn + n_mci * a_mci) / max(n_cn + n_mci, 1))
        row = {"cohort": cname, "arm": a, "auc_subjmean": A,
               "auc_repeat_mean": float(np.nanmean(pr)), "auc_repeat_sd": float(np.nanstd(pr, ddof=0)),
               "auc_repeat_median": float(np.nanmedian(pr)),
               "auc_repeat_min": float(np.nanmin(pr)), "auc_repeat_max": float(np.nanmax(pr)),
               "pr_auc": float(average_precision_score(yv, s)), "prevalence": float(yv.mean()),
               "hm_se": se, "wald_lo": max(0, A - 1.96 * se), "wald_hi": min(1, A + 1.96 * se),
               "logit_lo": lo_l, "logit_hi": hi_l,
               "auc_CN_vs_Dem": a_cn, "auc_MCI_vs_Dem": a_mci, "decomp_identity_gap": ident,
               "n_subjects": len(yv), "n_pos": n1}
        row.update(threshold_metrics(res, a))
        out.append(row)
    return pd.DataFrame(out).sort_values("auc_subjmean", ascending=False)

MAIN = pd.concat([summarize(RES[c], c) for c in COHORTS], ignore_index=True)
assert MAIN["decomp_identity_gap"].max() < 1e-9, \
    f"AUC 분해 항등식 위반 {MAIN['decomp_identity_gap'].max():.3e}"
print("\n" + "=" * 100)
print("MAIN — 사전등록 arm (분해 항등식 merged = (n_CN·A_CNvDem + n_MCI·A_MCIvDem)/n_neg 검산 통과)")
print(MAIN[["cohort", "arm", "auc_subjmean", "auc_repeat_mean", "auc_repeat_sd", "pr_auc",
            "auc_CN_vs_Dem", "auc_MCI_vs_Dem", "sensitivity", "specificity", "ppv"]]
      .round(4).to_string(index=False))
print("\n※ auc_repeat_sd 는 CV 분할 잡음이며 표본 오차가 아니다 (표본 SE 는 hm_se 열).")
HEAD_ARM = "A2_blockselect"
HEAD = MAIN[(MAIN.cohort == "FULL") & (MAIN.arm == HEAD_ARM)].iloc[0]
print(f"\n헤드라인 arm = {HEAD_ARM} / FULL 174 : AUC {HEAD.auc_subjmean:.4f} "
      f"(repeat 평균 {HEAD.auc_repeat_mean:.4f} ± {HEAD.auc_repeat_sd:.4f}, HM SE {HEAD.hm_se:.4f})")
PRIOR = {"DemScreen wearable_only__full (174)": 0.7184, "DemScreen wearable_only__filtered (172)": 0.8304,
         "DemRankAUC nested|wd_full (174)": 0.7303}
print("선행 실측 대비 (같은 과제·같은 코호트, 전부 4조건 준수):")
for k, v in PRIOR.items():
    print(f"  {k:45s} {v:.4f}  → Δ {HEAD.auc_subjmean - v:+.4f}")
_qc = MAIN[(MAIN.cohort == "QC") & (MAIN.arm == HEAD_ARM)].iloc[0]
print(f"  QC 172명 병기: {_qc.auc_subjmean:.4f} (Δ vs FULL {_qc.auc_subjmean - HEAD.auc_subjmean:+.4f}) "
      f"— 더 좋은 쪽을 헤드라인으로 고르지 않는다.")
progress("main", 1, 1)

# %%% CELL 12 [code]
# ============================================================================
# 셀 12 — 불확실성: 부트스트랩 · Hanley-McNeil · LOPO(잭나이프 + 실재 재실행) · permutation
# ============================================================================
def boot_ci(score, yv, n_boot, seed, stratified):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(yv == 1), np.flatnonzero(yv == 0)
    vals, npos, ndist = [], [], []
    for _ in range(n_boot):
        if stratified:
            idx = np.concatenate([rng.choice(pos, pos.size, True), rng.choice(neg, neg.size, True)])
        else:
            idx = rng.choice(len(yv), len(yv), True)
        yy = yv[idx]
        npos.append(int(yy.sum())); ndist.append(len(set(idx[yy == 1].tolist())))
        if yy.sum() and (yy == 0).sum():
            vals.append(roc_auc_score(yy, score[idx]))
    v = np.array(vals)
    return {"point": float(roc_auc_score(yv, score)), "lo": float(np.percentile(v, 2.5)),
            "hi": float(np.percentile(v, 97.5)), "width": float(np.percentile(v, 97.5) - np.percentile(v, 2.5)),
            "n_boot": len(v), "npos_mean": float(np.mean(npos)), "npos_sd": float(np.std(npos)),
            "distinct_pos_mean": float(np.mean(ndist))}, v

BOOT, BOOT_DRAWS = {}, {}
for cname in COHORTS:
    for a in PREREG_ARMS:
        for strat in (True, False):
            k = (cname, a, "strat" if strat else "unstrat")
            BOOT[k], d = boot_ci(RES[cname]["subject_score"][a], RES[cname]["y"], N_BOOT, SEED + 1, strat)
            if a == HEAD_ARM:
                BOOT_DRAWS[(cname, "strat" if strat else "unstrat")] = d
_b = BOOT[("FULL", HEAD_ARM, "strat")]
print(f"부트스트랩(층화, {_b['n_boot']}회) {HEAD_ARM}/FULL : {_b['point']:.4f} "
      f"95% CI [{_b['lo']:.4f}, {_b['hi']:.4f}]  **CI 폭 {_b['width']:.4f}**")
print(f"  리샘플당 양성 수 {_b['npos_mean']:.1f} ± {_b['npos_sd']:.2f} | 서로 다른 원 양성 개체 "
      f"{_b['distinct_pos_mean']:.1f} (CI 가장자리를 한 줌의 피험자가 좌우한다)")
_bu = BOOT[("FULL", HEAD_ARM, "unstrat")]
print(f"  비층화 CI [{_bu['lo']:.4f}, {_bu['hi']:.4f}] 폭 {_bu['width']:.4f} | "
      f"HM Wald [{HEAD.wald_lo:.4f}, {HEAD.wald_hi:.4f}] | HM logit [{HEAD.logit_lo:.4f}, {HEAD.logit_hi:.4f}]")

# --- 양성별 기여: 각 양성이 앞서는 음성의 분율. 이 값들의 평균이 곧 AUC.
_s = RES["FULL"]["subject_score"][HEAD_ARM]; _yv = RES["FULL"]["y"]
POSCONTRIB = pd.DataFrame({
    "hid": [SID_HASH[s] for s, k in zip(SIDS, _yv == 1) if k],
    "frac_negs_outranked": [float(((_s[_yv == 0] < v).mean() + .5 * (_s[_yv == 0] == v).mean()))
                            for v in _s[_yv == 1]]}).sort_values("frac_negs_outranked")
print(f"\n양성 {len(POSCONTRIB)}명의 개별 기여 (평균 = AUC {POSCONTRIB['frac_negs_outranked'].mean():.4f}, "
      f"1명당 가중치 {1/len(POSCONTRIB):.4f})")
print(POSCONTRIB.round(3).to_string(index=False))

# --- LOPO (a) 지표 잭나이프: 동결 OOF 점수에서 양성 1명씩 제거
_pos_idx = np.flatnonzero(_yv == 1)
LOPO_JACK = pd.DataFrame([{
    "hid": SID_HASH[np.array(SIDS)[i]],
    "auc_without": roc_auc_score(np.delete(_yv, i), np.delete(_s, i))} for i in _pos_idx])
# --- LOPO (b) 실재 재실행: 그 피험자를 파이프라인 전체에서 빼고 축소 프로토콜로 다시 돈다
_t = time.time(); _rows = []
for j, i in enumerate(_pos_idx):
    keep = np.ones(N_SUBJ, bool); keep[i] = False
    r = run_nested(FEAT.loc[np.array(SIDS)[keep]], y3[keep],
                   arms={"A2_blockselect": ARM_CANDS["A2_blockselect"]},
                   n_groups=1, r_per=LOPO_REPEATS, tag=f"lopo{j}", log_every=0)
    _rows.append({"hid": SID_HASH[np.array(SIDS)[i]],
                  "auc_rerun": roc_auc_score(r["y"], r["subject_score"]["A2_blockselect"])})
    tick("LOPO", j + 1, len(_pos_idx), every=1)
LOPO = LOPO_JACK.merge(pd.DataFrame(_rows), on="hid").sort_values("auc_rerun")
_ref = run_nested(FEAT, y3, arms={"A2_blockselect": ARM_CANDS["A2_blockselect"]},
                  n_groups=1, r_per=LOPO_REPEATS, tag="lopo_ref", log_every=0)
LOPO_REF = roc_auc_score(_ref["y"], _ref["subject_score"]["A2_blockselect"])
print(f"\nLOPO ({time.time()-_t:.0f}초, 축소 프로토콜 {LOPO_REPEATS} repeats, 동일 프로토콜 기준값 {LOPO_REF:.4f})")
print(LOPO.round(4).to_string(index=False))
print(f"  재실행 min {LOPO['auc_rerun'].min():.4f} / max {LOPO['auc_rerun'].max():.4f} "
      f"/ 폭 {LOPO['auc_rerun'].max()-LOPO['auc_rerun'].min():.4f}  → **헤드라인은 min 을 함께 인용한다**")

# --- permutation: 피험자 단위 3-class 라벨 셔플 + 전 선택 파이프라인 재생 (축소 프로토콜)
PERM_ARMS = {"A1_anchor_A":    [("lr", "A", c) for c in (0.03, 0.3)],
             "A2_blockselect": [("lr", b, c) for b in ("A", "C", "E") for c in (0.03, 0.3)],
             "A4_intraday_D":  [("lr", "D", c) for c in (0.03, 0.3)]}
_t = time.time()
_obs = run_nested(FEAT, y3, arms=PERM_ARMS, n_groups=1, r_per=PERM_REPEATS, tag="perm_obs", log_every=0)
OBS_LITE = {a: roc_auc_score(_obs["y"], _obs["subject_score"][a]) for a in PERM_ARMS}
print(f"\npermutation 축소 프로토콜의 관측값(귀무와 같은 조건): "
      + " | ".join(f"{a} {v:.4f}" for a, v in OBS_LITE.items()))
_rng = np.random.default_rng(SEED + 7)
NULL = {a: [] for a in OBS_LITE}
for b in range(N_PERM):
    yp = y3[_rng.permutation(N_SUBJ)]
    rp = run_nested(FEAT, yp, arms=PERM_ARMS, n_groups=1, r_per=PERM_REPEATS,
                    tag=f"perm{b}", log_every=0)
    for a in NULL:
        NULL[a].append(roc_auc_score(rp["y"], rp["subject_score"][a]))
    if (b + 1) % max(1, N_PERM // 20) == 0 or b == 0:
        tick("permutation", b + 1, N_PERM, every=1)
NULL = {a: np.array(v) for a, v in NULL.items()}
NULL_MAX = np.max(np.vstack([NULL[a] for a in PERM_ARMS if a in NULL]), axis=0)
PERM = {"n_perm": N_PERM, "repeats": PERM_REPEATS, "arms": list(PERM_ARMS),
        "null_mean": float(np.mean(NULL["A2_blockselect"])), "null_sd": float(np.std(NULL["A2_blockselect"])),
        "null_p95": float(np.percentile(NULL["A2_blockselect"], 95)),
        "null_p99": float(np.percentile(NULL["A2_blockselect"], 99)),
        "null_max_p95": float(np.percentile(NULL_MAX, 95)),
        "observed_lite": OBS_LITE,
        "p_primary": float((np.sum(NULL["A2_blockselect"] >= OBS_LITE["A2_blockselect"]) + 1) / (N_PERM + 1)),
        "p_max": float((np.sum(NULL_MAX >= max(OBS_LITE.values())) + 1) / (N_PERM + 1))}
print(f"permutation {N_PERM}회 ({(time.time()-_t)/60:.1f}분) | 귀무 평균 {PERM['null_mean']:.4f} "
      f"(무결성 게이트 ~0.50) SD {PERM['null_sd']:.4f} p95 {PERM['null_p95']:.4f} "
      f"| max-over-{len(PERM_ARMS)} p95 {PERM['null_max_p95']:.4f}")
print(f"  p_primary {PERM['p_primary']:.4f} | p_max {PERM['p_max']:.4f}")
assert 0.42 <= PERM["null_mean"] <= 0.58, f"귀무 평균이 0.5 에서 벗어남 {PERM['null_mean']:.4f} — CV 구조 편향 의심"
print("  ※ 이 검정은 '신호가 전혀 없음'만 기각한다. 귀무 p95 가 0.6~0.7 대이므로 참값 0.75 와 0.90 을 구별할 힘은 없다.")
progress("uncertainty", 1, 1)

# %%% CELL 13 [code]
# ============================================================================
# 셀 13 — 진단 arm (승격 금지): non-nested 낙관 · SMOTE · day-level 누수 대조군
# ============================================================================
DIAG = {}
# (a) non-nested 낙관: 같은 outer fold 위에서 inner 없이 '후보별 pooled OOF AUC 최대'
_t = time.time()
_cols = list(FEAT.columns); _X = FEAT[_cols].to_numpy(float); _ci = {c: i for i, c in enumerate(_cols)}
_keys = sorted({k for k in ARM_CANDS["A2_blockselect"]})
_nn = {k: np.full((R_PER_GROUP, N_SUBJ), np.nan) for k in _keys}
for r in range(R_PER_GROUP):
    skf = StratifiedKFold(n_splits=OUTER_K, shuffle=True, random_state=SEED_GROUPS[0] + r)
    for tr, te in skf.split(np.zeros(N_SUBJ), y3):
        p = Prep().fit(_X[tr]); A, B = p.transform(_X[tr]), p.transform(_X[te])
        for k in _keys:
            cl, kind, c = CANDS[k]; j = [_ci[x] for x in cl]
            _nn[k][r, te] = _fit_score(kind, c, A[:, j], y[tr], B[:, j])
_nn_auc = {k: float(np.mean([roc_auc_score(y, _nn[k][r]) for r in range(R_PER_GROUP)])) for k in _keys}
_nested_same = float(np.nanmean(RES["FULL"]["per_repeat"][HEAD_ARM][:R_PER_GROUP]))
DIAG["non_nested_max"] = max(_nn_auc.values())
DIAG["nested_same_budget"] = _nested_same
DIAG["optimism"] = DIAG["non_nested_max"] - _nested_same
print(f"(a) non-nested 최대 {DIAG['non_nested_max']:.4f} vs 동일 예산 nested {_nested_same:.4f} "
      f"→ **낙관 +{DIAG['optimism']:.4f}** (선행 실측 +0.053~+0.085) [{time.time()-_t:.0f}초]")

# (b) SMOTE 진단 — 동일 후보 예산으로 비교. 리샘플러는 학습 fold 에만, 합성 행은 평가에서 제외.
def _smote(Xtr, ytr, rng):
    pos = np.flatnonzero(ytr == 1)
    if pos.size < 2:
        return Xtr, ytr
    k = int(max(1, min(5, pos.size - 1)))
    need = int((ytr == 0).sum() - pos.size)
    if need <= 0:
        return Xtr, ytr
    P = Xtr[pos]; d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf); nn = np.argsort(d, 1)[:, :k]
    i = rng.integers(0, pos.size, need); jj = nn[i, rng.integers(0, k, need)]
    lam = rng.random((need, 1))
    S = P[i] + lam * (P[jj] - P[i])
    return np.vstack([Xtr, S]), np.concatenate([ytr, np.ones(need, int)])

_t = time.time(); _rng = np.random.default_rng(SEED + 3); _sm, _base = [], []
for r in range(R_PER_GROUP):
    skf = StratifiedKFold(n_splits=OUTER_K, shuffle=True, random_state=SEED_GROUPS[0] + r)
    ssm = np.full(N_SUBJ, np.nan); sbs = np.full(N_SUBJ, np.nan)
    for tr, te in skf.split(np.zeros(N_SUBJ), y3):
        j = [_ci[c] for c in BLOCK_A]
        p = Prep().fit(_X[tr]); A, B = p.transform(_X[tr])[:, j], p.transform(_X[te])[:, j]
        Xs, ys = _smote(A, y[tr], _rng)
        m = LogisticRegression(penalty="l2", C=0.1, solver="lbfgs", max_iter=5000).fit(Xs, ys)
        ssm[te] = m.predict_proba(B)[:, 1]
        sbs[te] = _fit_score("lr", 0.1, A, y[tr], B)
    _sm.append(roc_auc_score(y, ssm)); _base.append(roc_auc_score(y, sbs))
DIAG["smote_auc"] = float(np.mean(_sm)); DIAG["classweight_auc"] = float(np.mean(_base))
print(f"(b) 블록 A 에서 SMOTE {DIAG['smote_auc']:.4f} vs class_weight {DIAG['classweight_auc']:.4f} "
      f"→ Δ {DIAG['smote_auc']-DIAG['classweight_auc']:+.4f} [{time.time()-_t:.0f}초]")

# (c) day-level 누수 대조군 — **무효**. 같은 사람의 다른 날이 양쪽 fold 에 들어가면 어떻게 되는지 시연.
from sklearn.model_selection import KFold
_t = time.time()
_dcols = [c for c in ["activity_low", "activity_rest", "activity_steps", "activity_average_met",
                      "sleep_light_min", "sleep_restless", "metactive_frac", "cls_lowfrac",
                      "M10", "L5", "RAabs"] if c in DA.columns]
_Xd = DA[_dcols].to_numpy(float)
_yd = pd.Series(y, index=SIDS).reindex(DA["sid"]).to_numpy(int)
_sd_ = np.full(len(DA), np.nan)
for tr, te in KFold(5, shuffle=True, random_state=SEED).split(_Xd):
    p = Prep().fit(_Xd[tr])
    _sd_[te] = _fit_score("lr", 0.1, p.transform(_Xd[tr]), _yd[tr], p.transform(_Xd[te]))
DIAG["day_level_random_auc_INVALID"] = float(roc_auc_score(_yd, _sd_))
_sg = np.full(len(DA), np.nan)
_gid = pd.factorize(DA["sid"])[0]
for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(N_SUBJ), y3):
    m_tr = np.isin(_gid, tr); m_te = np.isin(_gid, te)
    p = Prep().fit(_Xd[m_tr])
    _sg[m_te] = _fit_score("lr", 0.1, p.transform(_Xd[m_tr]), _yd[m_tr], p.transform(_Xd[m_te]))
DIAG["day_level_subjectsplit_auc"] = float(roc_auc_score(_yd, _sg))
print(f"(c) 하루 단위 무작위 5-fold **INVALID** {DIAG['day_level_random_auc_INVALID']:.4f} vs "
      f"같은 특징·피험자 분할 {DIAG['day_level_subjectsplit_auc']:.4f} "
      f"→ 누수 크기 +{DIAG['day_level_random_auc_INVALID']-DIAG['day_level_subjectsplit_auc']:.4f} [{time.time()-_t:.0f}초]")
print("    (선행 저장소 실측: 0.9526 vs 0.5214. 이 숫자는 성능이 아니라 누수 진단값이다.)")
progress("diagnostics", 1, 1)

# %%% CELL 14 [markdown]
## ⚠️ 여기부터는 사전등록 밖 — 탐색적 분석

위 셀 11~13 의 결과가 **본 실험의 결론**이다. arm 6개(K=6)와 특징 블록 A~E 는 데이터를 보기 전에 고정됐고,
그 안의 모든 선택은 inner fold 에서만 일어났다.

아래 셀 15 는 성격이 다르다. **숫자를 최대로 끌어올리는 것이 목적인 탐색**이며,
넓은 후보 풀에서의 in-fold 선택, 이미 기각된 비율형 일주기 지표(IS / IV / RA — 선행 34특징 블록 단독 0.3999),
추가 학습기 계열을 전부 던져 넣는다. 여기서 나오는 최고값은 **여러 arm 중 최대를 고른 값**이므로
점추정으로서 위쪽으로 편향돼 있다.

그래서 이 섹션은 최고값과 함께 아래 셋을 **강제로** 출력한다.

1. **유효 K** — 이번 실행에서 본 arm 수. 다만 실제 노출은 이보다 훨씬 크다: 앵커 특징
   (`activity_low` 분산축)은 같은 174명 코호트에서 **라벨을 보며** SHAP/EDA 스크리닝(7,920행 표,
   218특징 스윕, 210행 모델 스크리닝)으로 발견됐다. nested CV 는 fold 내부 선택만 보정하고
   이 상속된 노출은 전혀 보정하지 못한다.
2. **best-of-K 귀무 p95** — K 개 arm 의 최대값이 순전히 우연으로 어디까지 가는지.
3. **winner-bias 보정치** — 같은 부트스트랩 재표본 위에서 `mean_b[max_j A_j^(b)] − max_j A_j(obs)`.

논문에 싣는다면 셀 11 의 표가 결과이고, 이 섹션은 "탐색적 분석" 소절이다.

# %%% CELL 15 [code]
# ============================================================================
# 셀 15 — 탐색적 arm (사전등록 밖). 최고값 + best-of-K 귀무 + winner-bias 보정 동시 출력.
# ============================================================================
_t = time.time()
# (1) 비율형 일주기 지표를 포함한 넓은 특징 풀 — 선행에서 기각된 계열도 논문 보고용으로 실측한다.
_hour = np.nanmean(METW.reshape(len(daily), 24, 60), axis=2)          # 일별 24시간 프로파일
_hour_adm = _hour[ADMIT]
_sid_adm = daily["sid"].to_numpy()[ADMIT]
def _is_iv(H):
    """IS(일간 안정성) / IV(일내 변동성) — 원 정의 그대로. 비율 지표이므로 사전 기대는 낮다."""
    if H.shape[0] < 3:
        return np.nan, np.nan
    x = H.reshape(-1); x = x[np.isfinite(x)]
    if x.size < 48:
        return np.nan, np.nan
    xb = x.mean(); denom = float(((x - xb) ** 2).sum())
    if denom <= 0:
        return np.nan, np.nan
    hp = np.nanmean(H, axis=0)
    IS = float(x.size * ((hp - xb) ** 2).sum() / (24 * denom))
    IV = float(x.size * ((np.diff(x)) ** 2).sum() / ((x.size - 1) * denom))
    return IS, IV
_rows = {}
for s in SIDS:
    H = _hour_adm[_sid_adm == s]
    IS, IV = _is_iv(H)
    g = DA[DA["sid"] == s]
    _rows[s] = {"circ_IS_ratio": IS, "circ_IV_ratio": IV,
                "circ_RAratio_mean": _mn(g["RAratio"]), "circ_M10_mean": _mn(g["M10"]),
                "circ_L5_mean": _mn(g["L5"]), "circ_M10on_circsd": _circ_sd_h(g["M10_on_h"] * 3600),
                "circ_L5on_circsd": _circ_sd_h(g["L5_on_h"] * 3600),
                "metmod_min_SD": _sd(g["metmod_min"]), "cls_highfrac_SD": _sd(g["cls_highfrac"]),
                "wake_win_h_SD": _sd(g["wake_win_h"]), "hr_night_min_SD": _sd(g["hr_night_min"]),
                "rmssd_night_SD": _sd(g["rmssd_night_mean"]), "waso_SD": _sd(g["hyp_waso_min"]),
                "awakenings_SD": _sd(g["hyp_awakenings"]), "deepbout_SD": _sd(g["hyp_deepbout_max"])}
EXTRA = pd.DataFrame.from_dict(_rows, orient="index").loc[SIDS]
# 일별 요약 채널의 mean/std 전수 (넓은 풀)
_wide = {}
for c in [c for c in (ACT_RICH + SLP_RICH) if c in DA.columns]:
    gp = DA.groupby("sid")[c]
    _wide[f"x_{c}__mean"] = gp.mean().reindex(SIDS)
    _wide[f"x_{c}__std"] = gp.std(ddof=0).reindex(SIDS)
WIDE = pd.DataFrame(_wide)
EXPL = pd.concat([FEAT, EXTRA, WIDE], axis=1)
EXPL = EXPL.loc[:, ~EXPL.columns.duplicated()]
EXPL = EXPL.loc[:, EXPL.isna().mean() < 0.25]
assert_no_forbidden(EXPL.columns, "EXPL")
print(f"탐색 특징 풀 {EXPL.shape[1]}개 (사전등록 {FEAT.shape[1]} + 일주기 비율 {EXTRA.shape[1]} + 넓은 {WIDE.shape[1]})")
_uni_x = pd.DataFrame({"feature": EXPL.columns,
                       "auc_dirfree": [max(a, 1 - a) if np.isfinite(a) else np.nan
                                       for a in (auc_of(EXPL[c], y) for c in EXPL.columns)]}
                      ).sort_values("auc_dirfree", ascending=False)
print("\n비율형 일주기 지표의 전코호트 단변량 (선행 기각 계열 — 논문 보고용):")
print(_uni_x[_uni_x.feature.str.startswith("circ_")].round(4).to_string(index=False))

# (2) 탐색 arm: 넓은 풀 in-fold top-k 선택 + 추가 학습기
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
_EXPL_COLS = list(EXPL.columns)
def _fit_score_x(kind, c, Xtr, ytr, Xte, topk=None, seed=SEED):
    if topk:
        a = np.array([abs(auc_of(Xtr[:, j], ytr) - .5) for j in range(Xtr.shape[1])])
        j = np.argsort(-np.nan_to_num(a))[:topk]
        Xtr, Xte = Xtr[:, j], Xte[:, j]
    if len(set(ytr.tolist())) < 2:
        return np.full(len(Xte), .5)
    if kind == "lr":
        m = LogisticRegression(penalty="l2", C=c, class_weight="balanced", solver="lbfgs", max_iter=5000)
    elif kind == "rf":
        m = RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=5,
                                   class_weight="balanced_subsample", random_state=seed, n_jobs=1)
    else:
        m = ExtraTreesClassifier(n_estimators=300, max_depth=4, min_samples_leaf=5,
                                 class_weight="balanced", random_state=seed, n_jobs=1)
    try:
        m.fit(Xtr, ytr); return m.predict_proba(Xte)[:, 1]
    except Exception:
        return np.full(len(Xte), .5)

EXPLORE_SPECS = [("wide_lr_top2", "lr", 0.1, 2), ("wide_lr_top5", "lr", 0.1, 5),
                 ("wide_lr_top10", "lr", 0.1, 10), ("wide_lr_all", "lr", 0.03, None),
                 ("wide_rf_all", "rf", None, None), ("wide_et_top10", "et", None, 10),
                 ("circ_ratio_lr", "lr", 0.1, None), ("blockE_rf", "rf", None, None)]
_Xx = EXPL.to_numpy(float)
_circ_j = [i for i, c in enumerate(_EXPL_COLS) if c.startswith("circ_")]
_bE_j = [_EXPL_COLS.index(c) for c in BLOCK_E]
EXP_OOF, EXP_ROWS = {}, []
for si, (nm, kind, c, topk) in enumerate(EXPLORE_SPECS):
    O = np.full((EXPLORE_REPS, N_SUBJ), np.nan)
    cols_j = _circ_j if nm == "circ_ratio_lr" else (_bE_j if nm == "blockE_rf" else list(range(len(_EXPL_COLS))))
    for r in range(EXPLORE_REPS):
        skf = StratifiedKFold(n_splits=OUTER_K, shuffle=True, random_state=SEED_GROUPS[0] + r)
        for tr, te in skf.split(np.zeros(N_SUBJ), y3):
            p = Prep().fit(_Xx[tr][:, cols_j])
            O[r, te] = _fit_score_x(kind, c, p.transform(_Xx[tr][:, cols_j]), y[tr],
                                    p.transform(_Xx[te][:, cols_j]), topk)
    s = np.mean([rank_norm(O[r]) for r in range(EXPLORE_REPS)], axis=0)
    EXP_OOF[nm] = s
    EXP_ROWS.append({"arm": nm, "kind": kind, "topk": topk, "n_pool": len(cols_j),
                     "auc_subjmean": roc_auc_score(y, s),
                     "auc_repeat_mean": float(np.mean([roc_auc_score(y, O[r]) for r in range(EXPLORE_REPS)]))})
    tick("explore", si + 1, len(EXPLORE_SPECS), every=1)
for a in PREREG_ARMS:                       # 사전등록 arm 도 같은 최대화 대상에 포함해야 정직하다
    EXP_OOF[a] = RES["FULL"]["subject_score"][a]
    EXP_ROWS.append({"arm": a + " (prereg)", "kind": "-", "topk": None, "n_pool": None,
                     "auc_subjmean": roc_auc_score(y, EXP_OOF[a]),
                     "auc_repeat_mean": float(np.nanmean(RES["FULL"]["per_repeat"][a]))})
EXPLORE = pd.DataFrame(EXP_ROWS).sort_values("auc_subjmean", ascending=False).reset_index(drop=True)
print(f"\n탐색 결과 ({(time.time()-_t)/60:.1f}분, {EXPLORE_REPS} repeats — 사전등록 60 repeats 와 예산이 다름)")
print(EXPLORE.round(4).to_string(index=False))

K_TOTAL = len(EXP_OOF)
_names = list(EXP_OOF)
_obs_max = max(roc_auc_score(y, EXP_OOF[a]) for a in _names)
_rng = np.random.default_rng(SEED + 11)
_obs_each = {a: roc_auc_score(y, EXP_OOF[a]) for a in _names}
_opt = []
for _ in range(min(N_BOOT, 2000)):
    idx = _rng.choice(N_SUBJ, N_SUBJ, True); yy = y[idx]
    if not (yy.sum() and (yy == 0).sum()):
        continue
    _b = {a: roc_auc_score(yy, EXP_OOF[a][idx]) for a in _names}
    _win = max(_b, key=_b.get)
    _opt.append(_b[_win] - _obs_each[_win])     # 재표본 승자가 재표본에서 얼마나 더 좋아 보이는가
WINNER_BIAS = float(np.mean(_opt))              # 선택 낙관 (양수) → 관측 최대에서 **뺀다**
# best-of-K 귀무: 측정된 단일 arm 귀무(평균·SD)에 rho=0.7 상관을 가정한 K개 최대의 p95
_mu, _sg = PERM["null_mean"], PERM["null_sd"]; _rho = 0.7
_z = _rng.standard_normal((20000, K_TOTAL)) * np.sqrt(1 - _rho) + _rng.standard_normal((20000, 1)) * np.sqrt(_rho)
BESTK_P95 = float(np.percentile((_mu + _sg * _z).max(1), 95))
print(f"\n유효 K = {K_TOTAL} (이번 실행분) | 관측 최대 {_obs_max:.4f}")
print(f"  best-of-K 귀무 p95 ≈ {BESTK_P95:.4f} (단일 arm 귀무 {_mu:.3f}±{_sg:.3f}, rho 0.7 가정)")
print(f"  선택 낙관(winner's curse) {WINNER_BIAS:+.4f} → 편향 보정 최대 ≈ {_obs_max - WINNER_BIAS:.4f}")
print("  ※ 진짜 유효 K 는 이보다 훨씬 크다: 앵커 특징이 같은 코호트의 라벨-인지 SHAP/EDA 스크리닝에서 왔다.")
progress("explore", 1, 1)

# %%% CELL 16 [code]
# ============================================================================
# 셀 16 — 그림 4~7 + 사전등록 판정
# ============================================================================
from sklearn.metrics import roc_curve, precision_recall_curve
_yv = RES["FULL"]["y"]

# ---- 그림 4: performance
fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
for a in PREREG_ARMS:
    fpr, tpr, _ = roc_curve(_yv, RES["FULL"]["subject_score"][a])
    ax[0,0].plot(fpr, tpr, lw=1.4, label=f"{a} {roc_auc_score(_yv, RES['FULL']['subject_score'][a]):.3f}")
ax[0,0].plot([0,1],[0,1],"k--",lw=.7); ax[0,0].legend(fontsize=6.5, loc="lower right")
ax[0,0].set_title("ROC, FULL 174 (subject-mean OOF)"); ax[0,0].set_xlabel("FPR"); ax[0,0].set_ylabel("TPR")
for a in PREREG_ARMS:
    pr, rc, _ = precision_recall_curve(_yv, RES["FULL"]["subject_score"][a])
    ax[0,1].plot(rc, pr, lw=1.3, label=a)
ax[0,1].axhline(_yv.mean(), color="k", ls="--", lw=.8, label=f"prevalence {_yv.mean():.3f}")
ax[0,1].legend(fontsize=6.5); ax[0,1].set_title("Precision-Recall (PPV is structurally low at 6.9%)")
ax[0,1].set_xlabel("recall"); ax[0,1].set_ylabel("precision")
_pr = [RES["FULL"]["per_repeat"][a] for a in PREREG_ARMS]
_box(ax[0,2], _pr, [a.split("_")[0] for a in PREREG_ARMS])
for i, v in enumerate(_pr):
    ax[0,2].scatter(np.full(v.size, i + 1) + np.random.uniform(-.1, .1, v.size), v, s=7, alpha=.5, color="#4477AA")
ax[0,2].axhline(.9, color="g", ls=":", lw=1.2); ax[0,2].axhline(.5, color="k", lw=.6)
ax[0,2].set_title(f"Per-repeat AUC (n={RES['FULL']['n_rep']}) - SPLIT NOISE, not sampling error")
_rows = [(f"{c}/{a}", MAIN[(MAIN.cohort==c)&(MAIN.arm==a)].iloc[0].auc_subjmean,
          BOOT[(c,a,"strat")]["lo"], BOOT[(c,a,"strat")]["hi"]) for c in COHORTS for a in PREREG_ARMS]
for i, (nm, p, lo, hi) in enumerate(_rows):
    ax[1,0].plot([lo, hi], [i, i], color="#666", lw=1.4)
    ax[1,0].plot(p, i, "o", color="#B2182B" if nm.startswith("FULL") else "#2166AC", ms=4)
ax[1,0].set_yticks(range(len(_rows))); ax[1,0].set_yticklabels([r[0] for r in _rows], fontsize=6)
ax[1,0].axvline(.9, color="g", ls=":", lw=1.2); ax[1,0].axvline(.5, color="k", lw=.6); ax[1,0].set_xlim(.3, 1.02)
ax[1,0].set_title("Forest: point + stratified subject bootstrap 95% CI")
_fa = RES["FULL"]["fold"]
ax[1,1].hist(_fa[_fa.arm == HEAD_ARM]["auc"].dropna(), bins=25, color="#888")
ax[1,1].axvline(HEAD.auc_subjmean, color="r"); ax[1,1].set_title(f"Per-FOLD AUC of {HEAD_ARM} (2-3 positives each)")
_p = np.nanmean(RES["FULL"]["oof"][HEAD_ARM], axis=0)
_bins = np.quantile(_p, np.linspace(0, 1, 11)); _bins[-1] += 1e-9
_bi = np.clip(np.digitize(_p, _bins) - 1, 0, 9)
_cal = pd.DataFrame({"b": _bi, "p": _p, "y": _yv}).groupby("b").agg(p=("p","mean"), o=("y","mean"), n=("y","size"))
ax[1,2].plot([0,1],[0,1],"k--",lw=.7); ax[1,2].plot(_cal["p"], _cal["o"], "o-", color="#B2182B")
BRIER = float(np.mean((_p - _yv) ** 2))
ax[1,2].set_title(f"Calibration (10 bins), Brier {BRIER:.4f}\nclass_weight='balanced' shifts the prior: "
                  f"miscalibration is expected, ROC-AUC is unaffected", fontsize=8)
ax[1,2].set_xlabel("mean predicted"); ax[1,2].set_ylabel("observed")
fig.suptitle(f"Fig 4  Performance  |  headline arm {HEAD_ARM} / FULL 174", fontsize=11)
_save(fig, "fig4_performance.png")

# ---- 그림 5: honesty
fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
ax[0,0].hist(NULL["A2_blockselect"], bins=45, alpha=.8, color="#999", label="null (single arm)")
ax[0,0].hist(NULL_MAX, bins=45, alpha=.5, color="#7B3294", label=f"null max over {len(PERM_ARMS)}")
ax[0,0].axvline(PERM["null_p95"], color="orange", ls="--", label=f"p95 {PERM['null_p95']:.3f}")
ax[0,0].axvline(OBS_LITE["A2_blockselect"], color="r", lw=1.6, label=f"observed {OBS_LITE['A2_blockselect']:.3f}")
ax[0,0].legend(fontsize=6.5); ax[0,0].set_title(f"Permutation null (n={N_PERM}), p_primary={PERM['p_primary']:.4f}")
_L = LOPO.sort_values("auc_rerun")
ax[0,1].hlines(range(len(_L)), LOPO_REF, _L["auc_rerun"], color="#999")
ax[0,1].plot(_L["auc_rerun"], range(len(_L)), "o", color="#B2182B", ms=5)
ax[0,1].axvline(LOPO_REF, color="k", ls="--", lw=.8, label=f"all-12 ref {LOPO_REF:.3f}")
ax[0,1].set_yticks(range(len(_L))); ax[0,1].set_yticklabels(_L["hid"], fontsize=6)
ax[0,1].legend(fontsize=6.5); ax[0,1].set_title(f"Leave-one-POSITIVE-out re-runs (min {_L['auc_rerun'].min():.3f})")
_P = POSCONTRIB
ax[0,2].barh(range(len(_P)), _P["frac_negs_outranked"], color="#B2182B")
ax[0,2].axvline(_P["frac_negs_outranked"].mean(), color="k", ls="--",
                label=f"mean = AUC {_P['frac_negs_outranked'].mean():.3f}")
ax[0,2].set_yticks(range(len(_P))); ax[0,2].set_yticklabels(_P["hid"], fontsize=6)
ax[0,2].legend(fontsize=6.5); ax[0,2].set_title(f"Each positive's share (weight 1/{len(_P)} = {1/len(_P):.4f})")
_np2 = N_POS * N_NEG
_lo_k = int(np.floor(.9 * _np2)); _grid = np.arange(_lo_k - 3, _lo_k + 5) / _np2
ax[1,0].vlines(_grid, 0, 1, color="#999", lw=1.2)
ax[1,0].axvline(.9, color="g", lw=2.0)
for gv in _grid[abs(_grid - .9) < 1.5 / _np2]:
    ax[1,0].annotate(f"{gv:.6f}", (gv, .55), rotation=90, fontsize=7, ha="right", va="center")
ax[1,0].annotate("0.900\n(not attainable)", (.9, .12), color="g", fontsize=8, ha="center")
ax[1,0].set_xlim(.9 - 3.2 / _np2, .9 + 3.2 / _np2); ax[1,0].set_ylim(0, 1); ax[1,0].set_yticks([])
ax[1,0].set_title(f"AUC is discrete: {N_POS} x {N_NEG} = {_np2} pairs, step 1/{_np2} = {1/_np2:.6f}\n"
                  f"one positive subject is worth up to {1/N_POS:.4f}", fontsize=8)
_np_ = np.arange(5, 120)
_hw = [1.96 * hanley_se(HEAD.auc_subjmean, k, int(k * N_NEG / max(N_POS, 1))) for k in _np_]
ax[1,1].plot(_np_, _hw, color="#2166AC"); ax[1,1].axvline(N_POS, color="r", ls="--", label=f"this study n_pos={N_POS}")
ax[1,1].axhline(HEAD.auc_subjmean - .8, color="#999", ls=":", label="half-width to clear LB 0.80")
ax[1,1].legend(fontsize=6.5); ax[1,1].set_xlabel("n_positive"); ax[1,1].set_ylabel("95% CI half-width")
ax[1,1].set_title("Precision is sample-size bound, not model bound")
_ks = np.array([1, 2, 4, 8, 16, 32, 64])
_sim = []
for k in _ks:
    z = np.random.default_rng(SEED).standard_normal((8000, int(k))) * np.sqrt(.3) + \
        np.random.default_rng(SEED + 1).standard_normal((8000, 1)) * np.sqrt(.7)
    _sim.append(np.percentile((PERM["null_mean"] + PERM["null_sd"] * z).max(1), 95))
ax[1,2].plot(_ks, _sim, "o-", color="#7B3294"); ax[1,2].set_xscale("log", base=2)
ax[1,2].axhline(HEAD.auc_subjmean, color="r", ls="--", label=f"observed {HEAD.auc_subjmean:.3f}")
ax[1,2].legend(fontsize=6.5); ax[1,2].set_xlabel("K arms looked at"); ax[1,2].set_ylabel("null p95 of max")
ax[1,2].set_title("Best-of-K null (rho=0.7)")
fig.suptitle("Fig 5  Honesty panel - what the number can and cannot support", fontsize=11)
_save(fig, "fig5_honesty.png")

# ---- 그림 6: interpretation
SEL = pd.concat([RES[c]["selection"] for c in COHORTS], ignore_index=True)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
_s2 = SEL[(SEL.tag == "FULL") & (SEL.arm == HEAD_ARM)]
_ct2 = pd.crosstab(_s2["block"], _s2["C"])
im = ax[0].imshow(_ct2.values, cmap="Blues", aspect="auto", vmin=0)
ax[0].set_xticks(range(_ct2.shape[1])); ax[0].set_xticklabels(_ct2.columns, fontsize=7)
ax[0].set_yticks(range(_ct2.shape[0])); ax[0].set_yticklabels(_ct2.index, fontsize=7)
for i in range(_ct2.shape[0]):
    for j in range(_ct2.shape[1]):
        ax[0].text(j, i, int(_ct2.values[i, j]), ha="center", va="center", fontsize=7)
ax[0].set_xlabel("C"); ax[0].set_ylabel("feature block")
ax[0].set_title(f"What inner CV chose over {len(_s2)} outer folds\n(instability itself is the finding)", fontsize=8.5)
_blk = BLOCKS[_ct2.sum(1).idxmax()]
_pp = Prep().fit(FEAT[list(_blk)].to_numpy(float))
_mfit = LogisticRegression(penalty="l2", C=0.1, class_weight="balanced", solver="lbfgs",
                           max_iter=5000).fit(_pp.transform(FEAT[list(_blk)].to_numpy(float)), y)
_co = pd.Series(_mfit.coef_[0], index=_blk).sort_values()
ax[1].barh(range(len(_co)), _co.values, color=["#B2182B" if v > 0 else "#2166AC" for v in _co.values])
ax[1].set_yticks(range(len(_co))); ax[1].set_yticklabels(_co.index, fontsize=7); ax[1].axvline(0, color="k", lw=.7)
ax[1].set_title(f"Full-data coefficients, block {_ct2.sum(1).idxmax()}\nDIAGNOSTIC ONLY - nothing is selected from this", fontsize=8.5)
_tm = MAIN[(MAIN.cohort == "FULL") & (MAIN.arm == HEAD_ARM)].iloc[0]
_cm2 = np.array([[_tm["tn"], _tm["fp"]], [_tm["fn"], _tm["tp"]]])
ax[2].imshow(_cm2, cmap="Oranges", vmin=0)
for i in range(2):
    for j in range(2):
        ax[2].text(j, i, f"{_cm2[i,j]:.1f}", ha="center", va="center", fontsize=11,
                   color="w" if _cm2[i, j] > _cm2.max() * .6 else "k")
ax[2].set_xticks([0,1]); ax[2].set_xticklabels(["pred CN+MCI", "pred Dem"], fontsize=7)
ax[2].set_yticks([0,1]); ax[2].set_yticklabels(["true CN+MCI", "true Dem"], fontsize=7)
ax[2].set_title(f"Mean confusion at nested Youden threshold\nsens {_tm['sensitivity']:.3f} spec {_tm['specificity']:.3f} PPV {_tm['ppv']:.3f}", fontsize=8.5)
fig.suptitle("Fig 6  Interpretation and stability", fontsize=11)
_save(fig, "fig6_interpretation.png")

# ---- 그림 7: leakage
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
_bars = [("day-random KFold\n(INVALID)", DIAG["day_level_random_auc_INVALID"], "#cccccc"),
         ("same features,\nsubject split", DIAG["day_level_subjectsplit_auc"], "#2166AC"),
         (f"nested {HEAD_ARM}\n(this study)", HEAD.auc_subjmean, "#B2182B")]
b = ax[0].bar([x[0] for x in _bars], [x[1] for x in _bars], color=[x[2] for x in _bars])
b[0].set_hatch("//")
for i, x in enumerate(_bars):
    ax[0].text(i, x[1], f"{x[1]:.3f}", ha="center", va="bottom", fontsize=8)
ax[0].axhline(.5, color="k", lw=.7); ax[0].set_ylim(0, 1.05); ax[0].set_ylabel("ROC-AUC")
ax[0].set_title("Day-level leakage demo (repo precedent: 0.9526 vs 0.5214)", fontsize=9)
ax[1].bar(["non-nested\n(max over cands)", "nested\n(same budget)"],
          [DIAG["non_nested_max"], DIAG["nested_same_budget"]], color=["#F4A582", "#2166AC"])
ax[1].text(.5, max(DIAG["non_nested_max"], DIAG["nested_same_budget"]),
           f"optimism +{DIAG['optimism']:.4f}", ha="center", va="bottom", fontsize=9)
ax[1].set_ylim(0, 1.05); ax[1].set_title("Nested vs non-nested optimism (repo precedent +0.053 to +0.085)", fontsize=9)
fig.suptitle("Fig 7  Leakage and optimism controls", fontsize=11)
_save(fig, "fig7_leakage.png")

# ---------------------------------------------------------------- 사전등록 판정
GATE = {
    "point_ge_0.90":      bool(HEAD.auc_subjmean >= 0.90),
    "boot_LB_ge_0.80":    bool(BOOT[("FULL", HEAD_ARM, "strat")]["lo"] >= 0.80),
    "LOPO_min_ge_0.85":   bool(LOPO["auc_rerun"].min() >= 0.85),
    "p_max_lt_0.01":      bool(PERM["p_max"] < 0.01)}
VERDICT = "0.9 DECLARED" if all(GATE.values()) else "0.9 NOT DECLARED"
print("\n" + "=" * 100)
print(f"사전등록 판정 게이트 (헤드라인 {HEAD_ARM} / FULL 174) — {VERDICT}")
for k, v in GATE.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")
_b = BOOT[("FULL", HEAD_ARM, "strat")]
SENTENCE = (f"점추정 {HEAD.auc_subjmean:.4f}, 층화 부트스트랩 95% CI [{_b['lo']:.4f}, {_b['hi']:.4f}] (폭 {_b['width']:.4f}), "
            f"LOPO 재실행 최소 {LOPO['auc_rerun'].min():.4f}, permutation p_primary {PERM['p_primary']:.4f} / "
            f"p_max {PERM['p_max']:.4f}, 이번 실행 유효 K {K_TOTAL} (실제 노출은 더 큼); "
            f"이 데이터는 참값 {_b['lo']:.2f} 와 이 값을 구별하지 못한다.")
print("\n[보고용 한 문장]\n  " + SENTENCE)
_best_pre = MAIN[MAIN.cohort == "FULL"].sort_values("auc_subjmean", ascending=False).iloc[0]
if _best_pre.arm != HEAD_ARM:
    _bb = BOOT[("FULL", _best_pre.arm, "strat")]
    print(f"\n[참고] 사전등록 6개 arm 중 FULL 최고는 {_best_pre.arm} {_best_pre.auc_subjmean:.4f} "
          f"CI [{_bb['lo']:.4f}, {_bb['hi']:.4f}] — 헤드라인보다 {_best_pre.auc_subjmean - HEAD.auc_subjmean:+.4f}. "
          f"헤드라인은 '최대'가 아니라 '사전등록된 추정량'이므로 바꾸지 않는다 (6개 중 최대에는 선택 편향이 남는다).")
print(f"\n[QC 병기] QC 172명 {HEAD_ARM}: "
      f"{MAIN[(MAIN.cohort=='QC')&(MAIN.arm==HEAD_ARM)].iloc[0].auc_subjmean:.4f} "
      f"CI [{BOOT[('QC',HEAD_ARM,'strat')]['lo']:.4f}, {BOOT[('QC',HEAD_ARM,'strat')]['hi']:.4f}] "
      f"— 라벨-블라인드 규칙이지만 양성 1명을 제거하며, 더 좋은 쪽을 헤드라인으로 고르지 않는다.")
progress("figures", 1, 1)

# %%% CELL 17 [code]
# ============================================================================
# 셀 17 — 결정성 재확인 + 산출물 저장
# ============================================================================
_t = time.time()
_re = run_nested(FEAT, y3, arms={HEAD_ARM: ARM_CANDS[HEAD_ARM]}, n_groups=1,
                 r_per=R_PER_GROUP, tag="determinism", log_every=0)
_h1 = hashlib.sha256(np.round(RES["FULL"]["oof"][HEAD_ARM][:R_PER_GROUP], 10).tobytes()).hexdigest()[:16]
_h2 = hashlib.sha256(np.round(_re["oof"][HEAD_ARM], 10).tobytes()).hexdigest()[:16]
DETERMINISTIC = _h1 == _h2
print(f"결정성 재실행 ({time.time()-_t:.0f}초): OOF 행렬 해시 {_h1} vs {_h2} → 동일 {DETERMINISTIC}")
assert DETERMINISTIC, "동일 시드에서 결과가 재현되지 않습니다"

CODE_CELLS = None
try:
    import IPython, json as _json
    _nbp = [p for p in Path.cwd().rglob(f"{EXPERIMENT}.ipynb")]
    if _nbp:
        _nb = _json.loads(_nbp[0].read_text())
        CODE_CELLS = hashlib.sha256("\n".join(
            "".join(c["source"]) for c in _nb["cells"] if c["cell_type"] == "code").encode()).hexdigest()
except Exception:
    pass

for nm, df in [("main_arms.csv", MAIN), ("fold_aucs.csv", pd.concat([RES[c]["fold"] for c in COHORTS])),
               ("selection_frequency.csv", SEL), ("univariate.csv", UNI), ("confound_audit.csv", CONF),
               ("fold_audit.csv", FOLD_AUDIT), ("lopo.csv", LOPO), ("positive_contribution.csv", POSCONTRIB),
               ("explore_arms.csv", EXPLORE), ("explore_univariate.csv", _uni_x),
               ("feature_matrix_hashed.csv", FEAT.rename(index=SID_HASH)),
               ("audit_matrix_hashed.csv", AUDIT.rename(index=SID_HASH)),
               ("bootstrap.csv", pd.DataFrame([{**{"cohort": k[0], "arm": k[1], "kind": k[2]}, **v}
                                               for k, v in BOOT.items()]))]:
    df.to_csv(OUT_DIR / nm, index=(nm.endswith("hashed.csv")))
pd.DataFrame({"hid": [SID_HASH[s] for s in SIDS], "y": y, "y3": y3, "qc_keep": QC_KEEP,
              **{f"score_{a}": RES["FULL"]["subject_score"][a] for a in PREREG_ARMS}}
             ).to_csv(OUT_DIR / "oof_predictions_hashed.csv", index=False)
np.save(OUT_DIR / "null_primary.npy", NULL["A2_blockselect"]); np.save(OUT_DIR / "null_max.npy", NULL_MAX)

REPORT = {
    "experiment": EXPERIMENT, "run_id": RUN_ID, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "task": "CN+MCI (0) vs Dem (1) — dementia screening, wearable-only",
    "contract": {"mmse_excluded": True, "subject_level_split": True, "nested_cv": True,
                 "binary_cn_mci_vs_dem": True, "has_holdout_test_set": False},
    "quick_mode": QUICK, "synthetic": SYNTHETIC, "wall_minutes": round((time.time() - _T_START) / 60, 2),
    "seed": SEED, "feature_fingerprint": FEAT_FP, "code_cells_sha256": CODE_CELLS,
    "deterministic_rerun": DETERMINISTIC, "oof_hash": _h1,
    "protocol": {"seed_groups": list(SEED_GROUPS), "repeats_per_group": R_PER_GROUP,
                 "n_repeats": RES["FULL"]["n_rep"], "outer_k": OUTER_K, "inner_k": INNER_K,
                 "inner_repeats": INNER_REPS, "C_grid": list(C_GRID), "tol": TOL,
                 "n_boot": N_BOOT, "n_perm": N_PERM, "perm_repeats": PERM_REPEATS,
                 "lopo_repeats": LOPO_REPEATS, "explore_repeats": EXPLORE_REPS,
                 "stratify_on": "y3 (CN/MCI/Dem)", "nonwear_met": NONWEAR_MET, "wear_gate": WEAR_GATE},
    "cohort": {"n_subjects": N_SUBJ, "CN": int((y3 == 0).sum()), "MCI": int((y3 == 1).sum()),
               "Dem": int((y3 == 2).sum()), "prevalence": float(y.mean()),
               "qc_kept": int(QC_KEEP.sum()), "qc_dropped_hids": _drop,
               "admitted_days": int(len(DA)), "joined_days": int(len(daily))},
    "headline": {"arm": HEAD_ARM, "cohort": "FULL",
                 "auc_subjmean": float(HEAD.auc_subjmean),
                 "auc_repeat_mean": float(HEAD.auc_repeat_mean),
                 "auc_repeat_sd_SPLIT_NOISE_NOT_SE": float(HEAD.auc_repeat_sd),
                 "hanley_mcneil_se": float(HEAD.hm_se),
                 "bootstrap_strat": BOOT[("FULL", HEAD_ARM, "strat")],
                 "bootstrap_unstrat": BOOT[("FULL", HEAD_ARM, "unstrat")],
                 "wald_ci": [float(HEAD.wald_lo), float(HEAD.wald_hi)],
                 "logit_ci": [float(HEAD.logit_lo), float(HEAD.logit_hi)],
                 "lopo_rerun_min": float(LOPO["auc_rerun"].min()),
                 "lopo_rerun_max": float(LOPO["auc_rerun"].max()),
                 "lopo_reference_same_protocol": float(LOPO_REF),
                 "brier": BRIER, "sentence": SENTENCE},
    "arms": MAIN.to_dict("records"),
    "permutation": PERM, "diagnostics": DIAG,
    "exploratory": {"note": "NOT pre-registered; max over arms is upward biased",
                    "effective_K_this_run": K_TOTAL, "observed_max": float(_obs_max),
                    "selection_optimism": WINNER_BIAS,
                    "bias_corrected_max": float(_obs_max - WINNER_BIAS), "bestK_null_p95": BESTK_P95,
                    "arms": EXPLORE.to_dict("records")},
    "gate": GATE, "verdict": VERDICT, "figures": FIGS,
    "prior_measurements_this_task": {
        "DemScreen_wearable_only_full_174": 0.7184, "DemScreen_wearable_only_filtered_172": 0.8304,
        "DemRankAUC_nested_wd_full": 0.7303, "DemRankAUC_wd_core_xgboost_NOT_NESTED": 0.8644},
    "limitations": [
        "독립 홀드아웃 없음 — train 141 + validation 33 을 풀링해 CV 한다. 33명 Validation 은 수십 개 선행 실험에 이미 노출된 역사적 벤치마크다.",
        "앵커 특징(activity_low 분산축)은 같은 174명에서 라벨을 보며 SHAP/EDA 스크리닝으로 발견됐다. nested CV 는 fold 내부 선택만 보정하고 이 상속된 노출은 보정하지 못한다.",
        f"양성 {N_POS}명 — AUC 는 {N_POS * N_NEG} 개 쌍 위의 이산량이고 0.900 은 도달 가능한 값이 아니다(0.899691 / 0.900206). 양성 1명이 최대 {1/max(N_POS,1):.4f} 를 움직인다.",
        "repeat 간 SD 는 CV 분할 잡음이며 표본 오차가 아니다. 표본 SE 는 Hanley-McNeil 값을 쓴다.",
        "모집 시기 교란: 등록월 × 클래스 분포가 균등하지 않다. 날짜 파생 특징을 전면 금지하고 등록월 표를 병기했다.",
        "인구통계(나이/성별/교육)가 릴리스에 없어 공변량 보정이 불가능하다. 활동 변동성은 연령과 |r|≈0.4 로 알려져 있다.",
        "permutation 은 '신호가 전혀 없음'만 기각한다. 참값 0.75 와 0.90 을 구별할 힘은 없다.",
        "QC 코호트 결과는 라벨-블라인드 규칙의 산물이지만 양성 1명을 제거한다. 헤드라인으로 쓰지 않는다."]}
(OUT_DIR / "FINAL_REPORT.json").write_text(json.dumps(REPORT, ensure_ascii=False, indent=2, default=float))
(OUT_DIR / "RUN_COMPLETE.json").write_text(json.dumps(
    {"status": "complete", "verdict": VERDICT, "headline_auc": float(HEAD.auc_subjmean),
     "wall_minutes": round((time.time() - _T_START) / 60, 2), "quick": QUICK, "synthetic": SYNTHETIC},
    ensure_ascii=False, indent=2))
progress("done", 1, 1)
print(f"\n저장 완료 → {OUT_DIR}")
print(f"총 소요 {(time.time()-_T_START)/60:.1f}분 | 판정 {VERDICT}")
if QUICK or SYNTHETIC:
    print("!! 배선 검증 모드 실행 — 이 숫자는 어디에도 보고하지 않는다.")
