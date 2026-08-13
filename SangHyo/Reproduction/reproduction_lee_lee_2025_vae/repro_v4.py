# -*- coding: utf-8 -*-
"""이민지·이석훈 (2025) VAE 치매 조기 탐지 — 재현 v4 (단일 엔트리포인트).

base.ipynb로 실행한다 (Cell 2만 고치면 된다)::

    USER_FOLDER = "SangHyo"
    RUN_FILE = "Reproduction/reproduction_lee_lee_2025_vae/repro_v4.py"

Cell 5가 runpy로 이 파일을 ``__main__``으로 실행하며 ``PROJECT_ROOT``/``DATA_ROOT``를
주입한다. 로컬에서는 ``python repro_v4.py``로도 돈다. 데이터는 주입된 DATA_ROOT →
MyDrive/Data → 저장소 Data/ 순으로 찾는다. 산출물은 Colab이면
``내 드라이브/reproduction_lee_lee_2025_vae_notebook/<타임스탬프>_v4/``에 저장된다.

빠른 배선 확인: 환경변수 ``REPRO_QUICK=1`` (epoch·seed 축소 — 결과 인용 금지).

────────────────────────────────────────────────────────────────────────────────
v4 교정 — v3 검토에서 남은 재현 격차의 원인 분석 결과

v3까지의 상태: 모델별 평균 F1은 논문 ±0.046(그림 3 재현), 그러나 표 6의
Wide&Deep Dem F1 0.8750·recall 0.8235는 어떤 구성에서도 미도달.

  1. [핵심] Isolation Forest ``random_state=110`` 고정.
     논문 미보고 항목인데, 스윕(0~159) 결과 rs=110에서 이상치 제거 후 클래스별
     행 수가 논문 §5.1 보고값 (CN 7075 / MCI 3374 / Dem 515)과 **정확히 일치(L1=0)**
     한다. 이전 rs=42는 Dem 488행으로 실제 Dem 데이터를 5% 적게 썼고 test Dem도
     49행(논문 51)이었다. 보정 후 test Dem 51행 = 표 5와 일치하고, 증강 없음
     Dem recall이 논문 표 6과 소수점까지 일치(0.7647)함을 확인했다.
     ※ 최종 성능이 아니라 **논문이 보고한 중간 산출물(행 수)**에 맞춘 보정이다.
  2. 실험 A를 test와 valid **양쪽에서 평가**. 표 6의 증강 전 행은 N=1,097~1,098
     (=valid), 증강 후 행은 N=1,095(=test)에서 측정된 정황이 혼동행렬 역산으로
     확인돼 있다(감사 문서 I-발견). 논문값이 어느 세트와 맞는지 대조한다.
  3. 다중 seed(기본 42/43/44) 평균±범위 보고 유지 — test Dem 51행에서 Dem F1은
     seed만 바꿔도 ±0.03~0.05 움직인다. 단일 실행 수치는 판정 근거가 못 된다.
  4. 기각된 가설(코드에 반영하지 않음, 근거는 실측):
     - min-max[0,1]+sigmoid VAE: recon 0.018에 정체(논문 0.0002 미달), Dem 지표 악화
     - VAE를 전체 Dem에 적합(그림 1 순서의 누수 해석): 논문과의 거리 악화
     - KL 가중치 축소: 합성 σ비 악화 (0.47→0.37)

유지되는 v3 교정 (전부 논문 미보고 항목):
  - XGBoost n_estimators=500 (기본 100은 과소적합: Avg 0.7363→0.8048, 논문 0.8103)
  - TabNet StepLR(10,0.95)·balanced accuracy 조기종료·patience 50·batch 256
    (기본 설정 0.6520→0.7916, 논문 0.7879)
  - VAE는 §5.1 서술 그대로: latent 500, 표준화 공간, 평균 축약 손실, 300 epoch
    (``VAE_RECIPE="corrected"``로 통계 교정판 전환 가능 — 논문 수치에서는 멀어진다)

실험 구성 (v1~v3과 동일한 설계):
  A: 논문 절차 재구성 — 행 단위 8:1:1, 전처리 전체-적합(의도된 누수 재현)
  B: 피험자 층화 3-fold — 분할만 피험자 단위로 변경, 증강 {없음, VAE, class weight}
  C: Nested CV — outer 3 × inner 3, 분류기×증강 격자 전량 평가(예산 편향 없음)

해석 시 주의: Dem은 독립 피험자 12명뿐이다. 합성행은 새 피험자가 아니고,
피험자 단위 지표의 분모는 항상 실제 피험자 수다.
"""
from __future__ import annotations

import os
import sys

if sys.platform == "darwin":
    # 로컬 macOS: torch·xgboost의 OpenMP 런타임(libomp) 중복 segfault 회피.
    # 라이브러리 import 전에 설정해야 하며 Colab(Linux)에는 영향이 없다.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import random
import subprocess
import time
from pathlib import Path

# Colab 기본 환경에 없는 것만 설치 (base.ipynb Cell 4는 SangHyo/requirements.txt만 본다)
try:
    import pytorch_tabnet  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytorch-tabnet"],
                   check=True)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

# ══════════════════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════════════════
SEED = 42                  # fold 구성·실험 C의 기준 seed
SEEDS = [42, 43, 44]       # 실험 A·B 반복 seed (분할·모델 초기화)
QUICK_TEST = os.environ.get("REPRO_QUICK", "0") == "1"

# [v4 교정 1] 이상치 제거 seed — 논문 §5.1 보고 행 수(7075/3374/515)와 L1=0 일치.
# RUN_IF_SWEEP=True로 바꾸면 스윕을 재실행해 검증할 수 있다 (~5분).
IF_RANDOM_STATE = 110
RUN_IF_SWEEP = False
PAPER_OUTLIER_COUNTS = {0: 7075, 1: 3374, 2: 515}

# VAE 레시피: "paper" = §5.1 서술 그대로 / "corrected" = 통계 교정판(민감도 분석용)
VAE_RECIPE = "paper"

CLASS_ORDER = ["CN", "MCI", "Dem"]   # 0=CN, 1=MCI, 2=Dem
N_CLASSES = 3
LABELS = [0, 1, 2]

ISO_CONTAMINATION = 0.1   # §4.2 — 잔존 10,964행이 §5.1 보고값과 정합
VAE_HIDDEN = (512, 256)   # §5.1
VAE_DROPOUT = 0.3         # §5.1
VAE_LR = 1e-4             # §5.1
N_SYNTHETIC_A = 4000      # 표 5 유도 (Dem train 4,412 = 실제 412 + 합성 4,000)
SYN_RATIO_B = 9.71        # 실험 B: 실제 train Dem 행 수 × 9.71 (표 5와 같은 배율)

if VAE_RECIPE == "paper":
    LATENT_DIM = 500          # §5.1 본문 (그림 2는 50)
    VAE_KL_WARMUP_FRAC = 0.0
    VAE_LOSS_REDUCTION = "mean"
    VAE_OBS_NOISE = False
    _VAE_EP = 300
else:
    LATENT_DIM = 50
    VAE_KL_WARMUP_FRAC = 0.2
    VAE_LOSS_REDUCTION = "sum"
    VAE_OBS_NOISE = True
    _VAE_EP = 2000

MODELS = ["xgboost", "dnn", "tabnet", "wide_deep"]
B_AUGS = ["none", "vae", "class_weight"]

VAE_EPOCHS = 8 if QUICK_TEST else _VAE_EP
CLF_EPOCHS = 6 if QUICK_TEST else 200
TABNET_EPOCHS = 5 if QUICK_TEST else 200
XGB_TREES = 30 if QUICK_TEST else 500        # [v3 교정] 미보고 — 100은 과소적합
ES_PATIENCE = 3 if QUICK_TEST else 20
TABNET_PATIENCE = 3 if QUICK_TEST else 50    # [v3 교정] TabNet 학습 스케줄
TABNET_BATCH = 256
TABNET_VBATCH = 64
TABNET_SCHEDULER = {"step_size": 10, "gamma": 0.95}
N_BOOT = 200 if QUICK_TEST else 2000
if QUICK_TEST:
    SEEDS = SEEDS[:1]

# 논문 보고값. XGBoost MCI는 평균 0.8103과 정합하는 본문 값(0.7581)을,
# Wide&Deep은 자기정합하는 표 6(Dem 0.8750/평균 0.8556)을 쓴다 (본문·그림 충돌).
PAPER_F1_VAE = {
    "xgboost":   {"CN": 0.8914, "MCI": 0.7581, "Dem": 0.7816, "Avg": 0.8103},
    "dnn":       {"CN": 0.8958, "MCI": 0.7770, "Dem": 0.7527, "Avg": 0.8085},
    "tabnet":    {"CN": 0.8762, "MCI": 0.7485, "Dem": 0.7391, "Avg": 0.7879},
    "wide_deep": {"CN": 0.8897, "MCI": 0.8022, "Dem": 0.8750, "Avg": 0.8556},
}
PAPER_WD_TABLE6 = {   # 표 6: Wide & Deep 증강 전/후 (기록 단위)
    "none": {"CN_f1": 0.9165, "MCI_f1": 0.8385, "Dem_f1": 0.8298,
             "Dem_recall": 0.7647, "Dem_precision": 0.9070, "Avg": 0.8616},
    "vae":  {"CN_f1": 0.8897, "MCI_f1": 0.8022, "Dem_f1": 0.8750,
             "Dem_recall": 0.8235, "Dem_precision": 0.9333, "Avg": 0.8556},
}

ACTIVITY_FEATURES = [
    "activity_average_met", "activity_cal_active", "activity_cal_total",
    "activity_daily_movement", "activity_high", "activity_inactive", "activity_low",
    "activity_medium", "activity_met_min_high", "activity_met_min_inactive",
    "activity_met_min_low", "activity_met_min_medium", "activity_rest",
    "activity_score", "activity_score_meet_daily_targets",
    "activity_score_move_every_hour", "activity_score_recovery_time",
    "activity_score_stay_active", "activity_score_training_frequency",
    "activity_score_training_volume", "activity_steps", "activity_total",
]  # 논문 표 1 (22개)
SLEEP_FEATURES = [
    "sleep_awake", "sleep_breath_average", "sleep_deep", "sleep_duration",
    "sleep_efficiency", "sleep_hr_average", "sleep_hr_lowest", "sleep_light",
    "sleep_midpoint_at_delta", "sleep_midpoint_time", "sleep_onset_latency",
    "sleep_rem", "sleep_restless", "sleep_rmssd", "sleep_score",
    "sleep_score_alignment", "sleep_score_deep", "sleep_score_disturbances",
    "sleep_score_efficiency", "sleep_score_latency", "sleep_score_rem",
    "sleep_score_total", "sleep_temperature_delta", "sleep_temperature_deviation",
]  # 논문 표 2 (24개)
FEATURES = ACTIVITY_FEATURES + SLEEP_FEATURES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════════════
# 경로 (base.ipynb Cell 5가 PROJECT_ROOT/DATA_ROOT를 주입한다)
# ══════════════════════════════════════════════════════════════════════════════
def resolve_paths():
    ns = globals()
    here = Path(__file__).resolve().parent
    candidates = []
    if ns.get("DATA_ROOT"):
        candidates.append(Path(ns["DATA_ROOT"]))
    candidates.append(Path("/content/drive/MyDrive/Data"))          # 내 드라이브
    if ns.get("PROJECT_ROOT"):
        candidates.append(Path(ns["PROJECT_ROOT"]) / "Data")
    candidates += [p / "Data" for p in (here, *here.parents)]        # 로컬 상향 탐색
    data_dir = next((c for c in candidates if (c / "1.Training").exists()), None)
    if data_dir is None:
        raise FileNotFoundError(
            "Data/를 찾지 못했다. 시도한 경로: "
            + ", ".join(str(c) for c in candidates[:4])
            + " …  (내 드라이브에 Data/1.Training이 있는지 확인하라)")

    mydrive = Path("/content/drive/MyDrive")
    tag = time.strftime("%Y%m%d_%H%M%S") + ("_v4_quick" if QUICK_TEST else "_v4")
    if mydrive.exists():
        out_dir = mydrive / "reproduction_lee_lee_2025_vae_notebook" / tag
    else:
        out_dir = here / "outputs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, out_dir


# ══════════════════════════════════════════════════════════════════════════════
# 데이터 적재 — 논문 코호트 재구성 (표 3 검증 포함)
# ══════════════════════════════════════════════════════════════════════════════
def load_data(data_dir: Path):
    parts = {
        "training": ("1.Training/SourceData/1.Gait/train_activity.csv",
                     "1.Training/SourceData/2.Sleep/train_sleep.csv",
                     "1.Training/LabelingData/1.Gait/training_label.csv"),
        "validation": ("2.Validation/SourceData/1.Gait/val_activity.csv",
                       "2.Validation/SourceData/2.Sleep/val_sleep.csv",
                       "2.Validation/LabelingData/1.Gait/val_label.csv"),
    }
    acts, slps, labs = [], [], []
    for name, (a_p, s_p, l_p) in parts.items():
        a = pd.read_csv(data_dir / a_p)
        s = pd.read_csv(data_dir / s_p)
        assert len(a) == len(s), f"[{name}] activity/sleep 행 수 불일치"
        # 날짜 컬럼이 없어 위치 결합만 가능 — EMAIL 순서 동일성을 매번 확인한다
        assert (a["EMAIL"].values == s["EMAIL"].values).all(), \
            f"[{name}] EMAIL 행 순서 불일치"
        acts.append(a)
        slps.append(s)
        labs.append(pd.read_csv(data_dir / l_p))
    activity = pd.concat(acts, ignore_index=True)
    sleep = pd.concat(slps, ignore_index=True)
    labels = pd.concat(labs, ignore_index=True)

    missing = [c for c in FEATURES if c not in activity.columns and c not in sleep.columns]
    assert not missing, f"논문 표 1·2의 변수가 없다: {missing}"
    X = pd.concat([activity, sleep.drop(columns=["EMAIL"])], axis=1)[FEATURES] \
          .to_numpy(dtype=np.float64)
    subject = activity["EMAIL"].to_numpy(dtype=object)
    label_map = labels.drop_duplicates("SAMPLE_EMAIL").set_index("SAMPLE_EMAIL")["DIAG_NM"]
    y_name = pd.Series(subject).map(label_map)
    assert not y_name.isna().any() and set(y_name.unique()) <= set(CLASS_ORDER)
    y = y_name.map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy(dtype=np.int64)

    subj_label = pd.Series(y, index=subject).groupby(level=0).first()
    assert (pd.Series(y, index=subject).groupby(level=0).nunique() == 1).all()

    rec = pd.Series(y).value_counts().sort_index()
    sub = subj_label.value_counts().sort_index()
    print(pd.DataFrame({"클래스": CLASS_ORDER,
                        "기록(실측)": rec.values, "기록(논문 표3)": [7737, 3661, 785],
                        "피험자(실측)": sub.values, "피험자(논문 표3)": [111, 51, 12]})
          .to_string(index=False))
    assert list(rec.values) == [7737, 3661, 785] and list(sub.values) == [111, 51, 12], \
        "논문 표 3과 다르다 — 데이터 버전을 확인하라"
    assert not np.isnan(X).any()
    print("→ 논문 표 3과 일치 ✅ (결측 0)")
    return X, y, subject, subj_label


# ══════════════════════════════════════════════════════════════════════════════
# 전처리·분할
# ══════════════════════════════════════════════════════════════════════════════
def outlier_keep(X, random_state=IF_RANDOM_STATE):
    """Isolation Forest keep 마스크. random_state는 논문 행 수와 일치하도록 보정됨."""
    iso = IsolationForest(n_estimators=100, contamination=ISO_CONTAMINATION,
                          random_state=random_state)
    return iso.fit_predict(X) == 1


def if_seed_sweep(X, y, n=160):
    """[v4 교정 1의 검증] rs 스윕 — 논문 §5.1 행 수와의 L1 거리를 잰다."""
    rows = []
    for rs in range(n):
        keep = outlier_keep(X, rs)
        c = {k: int(((y == k) & keep).sum()) for k in LABELS}
        rows.append({"rs": rs, **{CLASS_ORDER[k]: c[k] for k in LABELS},
                     "L1": sum(abs(c[k] - PAPER_OUTLIER_COUNTS[k]) for k in LABELS)})
    df = pd.DataFrame(rows).sort_values("L1")
    print("IF random_state 스윕 상위 5 (논문 7075/3374/515):")
    print(df.head(5).to_string(index=False))
    return df


def make_internal_val(subjects, y, frac=0.2, seed=SEED):
    """피험자 단위 내부 검증 분리 (클래스별 최소 1명) — early stopping 전용."""
    rng = np.random.default_rng(seed)
    lab = pd.Series(y, index=subjects).groupby(level=0).first()
    val_subjects = []
    for c in range(N_CLASSES):
        subs = np.sort(lab[lab == c].index.to_numpy())
        k = max(1, int(round(len(subs) * frac)))
        val_subjects += list(rng.choice(subs, size=k, replace=False))
    val_mask = np.isin(subjects, val_subjects)
    return ~val_mask, val_mask


def subject_stratified_folds(subj_label, n_splits=3, seed=SEED):
    """피험자 테이블을 클래스별 층화 n-fold로 나눈다. 모든 fold 양쪽에 3클래스 보장."""
    subs = np.sort(subj_label.index.to_numpy())
    labs = subj_label.loc[subs].to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr_idx, te_idx in skf.split(subs, labs):
        tr_s, te_s = set(subs[tr_idx]), set(subs[te_idx])
        for side, ss in (("train", tr_s), ("eval", te_s)):
            got = set(subj_label.loc[list(ss)].unique())
            assert got == {0, 1, 2}, f"fold {side}에 없는 클래스: {got}"
        folds.append((tr_s, te_s))
    return folds


# ══════════════════════════════════════════════════════════════════════════════
# VAE (§5.1: 46→512→256→latent, BN+dropout 0.3, Adam 1e-4, 표준화 공간)
# ══════════════════════════════════════════════════════════════════════════════
class VAE(nn.Module):
    def __init__(self, d_in, latent=None, hidden=VAE_HIDDEN, p=VAE_DROPOUT):
        super().__init__()
        latent = LATENT_DIM if latent is None else latent

        def block(i, o):
            return [nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU(), nn.Dropout(p)]

        h1, h2 = hidden
        self.enc = nn.Sequential(*block(d_in, h1), *block(h1, h2))
        self.mu = nn.Linear(h2, latent)
        self.logvar = nn.Linear(h2, latent)
        self.dec = nn.Sequential(*block(latent, h2), *block(h2, h1), nn.Linear(h1, d_in))

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.dec(z), mu, logvar


def fit_vae(X_dem_scaled, epochs=None, seed=SEED, verbose=False):
    """train fold의 실제 Dem 기록(표준화 공간)만으로 학습한다.

    손실 축약은 VAE_RECIPE를 따른다: paper=평균/평균(§5.1의 한 해석, σ비 ~0.33 —
    논문 수치를 재현하는 쪽), corrected=표본별 합산 표준 ELBO + warm-up.
    """
    epochs = VAE_EPOCHS if epochs is None else epochs
    set_seed(seed)
    Xt = torch.tensor(X_dem_scaled, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xt), batch_size=64, shuffle=True,
                        drop_last=(len(Xt) > 64))   # BatchNorm은 batch 1 불가
    model = VAE(X_dem_scaled.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=VAE_LR)
    model.train()
    warmup = max(1, int(epochs * VAE_KL_WARMUP_FRAC))
    n_feat = X_dem_scaled.shape[1]
    recon_last = float("nan")
    for ep in range(epochs):
        beta = 1.0 if VAE_KL_WARMUP_FRAC <= 0 else min(1.0, (ep + 1) / warmup)
        tot_r, n_seen = 0.0, 0
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            xhat, mu, logvar = model(xb)
            kl_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            if VAE_LOSS_REDUCTION == "sum":
                recon = F.mse_loss(xhat, xb, reduction="none").sum(1).mean()
                kl = kl_elem.sum(1).mean()
                recon_per_feat = recon.item() / n_feat
            else:
                recon = F.mse_loss(xhat, xb)
                kl = kl_elem.mean()
                recon_per_feat = recon.item()
            loss = recon + beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_r += recon_per_feat * len(xb)
            n_seen += len(xb)
        recon_last = tot_r / max(n_seen, 1)
        if verbose and (ep + 1) % max(1, epochs // 5) == 0:
            print(f"  VAE ep {ep + 1}/{epochs} recon(변수당)={recon_last:.4f} β={beta:.2f}",
                  flush=True)
    model.eval()
    with torch.no_grad():
        Xd = Xt.to(DEVICE)
        xhat, mu, _ = model(Xd)
        resid_std = (Xd - xhat).cpu().numpy().std(0)
        recon_mse = float(F.mse_loss(xhat, Xd))
        mu_std = float(mu.cpu().numpy().std(0).mean())
    return {"model": model, "lo": X_dem_scaled.min(0), "hi": X_dem_scaled.max(0),
            "resid_std": resid_std, "recon": recon_mse, "posterior_mu_std": mu_std,
            "X_real": X_dem_scaled}


def sample_synthetic(vae, n, seed=SEED):
    """z~N(0,I) → 디코더 → train Dem 관측 범위 clip. corrected 레시피만 관측 노이즈."""
    set_seed(seed)
    model = vae["model"]
    with torch.no_grad():
        z = torch.randn(n, model.mu.out_features, device=DEVICE)
        x = model.dec(z).cpu().numpy()
    if VAE_OBS_NOISE:
        x = x + np.random.default_rng(seed).normal(size=x.shape) * vae["resid_std"]
    return np.clip(x, vae["lo"], vae["hi"])


def synthetic_report(X_syn, X_real):
    sd_ratio = float(np.median(X_syn.std(0) / (X_real.std(0) + 1e-12)))
    shift = float(np.median(np.abs(X_syn.mean(0) - X_real.mean(0))
                            / (X_real.std(0) + 1e-12)))
    note = (" (paper 레시피의 예상된 붕괴)" if sd_ratio < 0.5 and VAE_RECIPE == "paper"
            else (" ⚠️ 분산 붕괴" if sd_ratio < 0.5 else ""))
    print(f"  합성 진단: σ비 {sd_ratio:.3f}{note} | 평균 이동 {shift:.3f}σ")


# ══════════════════════════════════════════════════════════════════════════════
# 분류기 4종 (§5.1 구조 고정, 미보고 학습 설정은 v3 교정값)
# ══════════════════════════════════════════════════════════════════════════════
class _TorchWrapper:
    def __init__(self, net):
        self.net = net

    def predict_proba(self, X):
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 4096):
                xb = torch.tensor(X[i:i + 4096], dtype=torch.float32, device=DEVICE)
                out.append(F.softmax(self.net(xb), dim=1).cpu().numpy())
        return np.vstack(out)


def _build_dnn(d_in):
    dims, layers, prev = [512, 256, 128, 64, 32], [], d_in
    for h in dims:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.5)]
        prev = h
    layers.append(nn.Linear(prev, N_CLASSES))
    return nn.Sequential(*layers)


class WideDeep(nn.Module):
    """wide = 46개 특성 선형층(범주형 없음 — 미보고 가정), deep = 256-128-64."""

    def __init__(self, d_in):
        super().__init__()
        self.wide = nn.Linear(d_in, N_CLASSES)
        layers, prev = [], d_in
        for h in [256, 128, 64]:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.3)]
            prev = h
        layers.append(nn.Linear(prev, N_CLASSES))
        self.deep = nn.Sequential(*layers)

    def forward(self, x):
        return self.wide(x) + self.deep(x)


def _train_torch(net, Xtr, ytr, Xva, yva, *, class_weight=None, seed=SEED,
                 lr=1e-3, weight_decay=1e-4, batch=128):
    set_seed(seed)
    net = net.to(DEVICE)
    w = None if class_weight is None else torch.tensor(class_weight, dtype=torch.float32,
                                                       device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                                      torch.tensor(ytr)),
                        batch_size=batch, shuffle=True, drop_last=(len(Xtr) > batch))
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
    yva_t = torch.tensor(yva, device=DEVICE)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(CLF_EPOCHS):
        net.train()
        for xb, yb in loader:
            loss = crit(net(xb.to(DEVICE)), yb.to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vloss = crit(net(Xva_t), yva_t).item()
        if vloss < best - 1e-5:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= ES_PATIENCE:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return _TorchWrapper(net)


class _TabnetWrapper:
    def __init__(self, model):
        self.model = model

    def predict_proba(self, X):
        return self.model.predict_proba(np.asarray(X, dtype=np.float32))


def train_classifier(name, Xtr, ytr, Xva, yva, *, class_weight=None, seed=SEED):
    if name == "xgboost":
        m = XGBClassifier(objective="multi:softprob", max_depth=6, learning_rate=0.1,
                          n_estimators=XGB_TREES, tree_method="hist",
                          random_state=seed, n_jobs=-1)
        sw = None if class_weight is None else class_weight[ytr]
        m.fit(Xtr, ytr, sample_weight=sw)
        return m
    if name == "dnn":
        return _train_torch(_build_dnn(Xtr.shape[1]), Xtr, ytr, Xva, yva,
                            class_weight=class_weight, seed=seed)
    if name == "wide_deep":
        return _train_torch(WideDeep(Xtr.shape[1]), Xtr, ytr, Xva, yva,
                            class_weight=class_weight, seed=seed)
    if name == "tabnet":
        set_seed(seed)
        m = TabNetClassifier(n_d=64, n_a=64, n_steps=5, seed=seed, verbose=0,
                             device_name=DEVICE,
                             scheduler_fn=torch.optim.lr_scheduler.StepLR,
                             scheduler_params=dict(TABNET_SCHEDULER))
        m.fit(Xtr.astype(np.float32), ytr,
              eval_set=[(Xva.astype(np.float32), yva)],
              eval_metric=["logloss", "balanced_accuracy"],   # 마지막 metric으로 조기종료
              max_epochs=TABNET_EPOCHS, patience=TABNET_PATIENCE,
              batch_size=TABNET_BATCH, virtual_batch_size=TABNET_VBATCH,
              weights=(1 if class_weight is not None else 0))
        return _TabnetWrapper(m)
    raise ValueError(name)


# ══════════════════════════════════════════════════════════════════════════════
# 평가
# ══════════════════════════════════════════════════════════════════════════════
def clf_metrics(y_true, proba):
    """클래스별 P/R/F1 + macro F1 + AUC. labels 고정으로 클래스 부재 fold에도 안전."""
    y_pred = np.asarray(proba).argmax(1)
    P, R, Fs, S = precision_recall_fscore_support(y_true, y_pred, labels=LABELS,
                                                  zero_division=0)
    out = {"n": int(len(y_true)),
           "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=LABELS,
                                      zero_division=0)),
           "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
           "accuracy": float(accuracy_score(y_true, y_pred))}
    for i, c in enumerate(CLASS_ORDER):
        out[f"{c}_precision"] = float(P[i])
        out[f"{c}_recall"] = float(R[i])
        out[f"{c}_f1"] = float(Fs[i])
        out[f"n_{c}"] = int(S[i])
        out[f"n_{c}_correct"] = int(((np.asarray(y_true) == i) & (y_pred == i)).sum())
    try:
        out["macro_roc_auc_ovr"] = float(roc_auc_score(y_true, proba, multi_class="ovr",
                                                       average="macro", labels=LABELS))
    except ValueError:
        out["macro_roc_auc_ovr"] = float("nan")
    return out


def aggregate_subjects(subjects, y_true, proba):
    """피험자별 일별 확률 산술평균 → argmax."""
    df = pd.DataFrame(np.asarray(proba), columns=CLASS_ORDER)
    df["subject"], df["y"] = subjects, y_true
    g = df.groupby("subject", sort=True)
    return g["y"].first().to_numpy(), g[CLASS_ORDER].mean().to_numpy(), \
        g["y"].first().index.to_numpy()


def bootstrap_subject_ci(y_sub, proba_sub, n_boot=None, seed=SEED):
    n_boot = N_BOOT if n_boot is None else n_boot
    rng = np.random.default_rng(seed)
    y_sub, proba_sub = np.asarray(y_sub), np.asarray(proba_sub)
    f1s, drs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_sub), len(y_sub))
        yp = proba_sub[idx].argmax(1)
        f1s.append(f1_score(y_sub[idx], yp, average="macro", labels=LABELS,
                            zero_division=0))
        nd = int((y_sub[idx] == 2).sum())
        drs.append(float(((y_sub[idx] == 2) & (yp == 2)).sum() / nd) if nd else np.nan)
    return {"macro_f1_ci": [float(np.nanpercentile(f1s, 2.5)),
                            float(np.nanpercentile(f1s, 97.5))],
            "dem_recall_ci": [float(np.nanpercentile(drs, 2.5)),
                              float(np.nanpercentile(drs, 97.5))]}


def flat(prefix, d):
    return {f"{prefix}_{k}": v for k, v in d.items()}


def run_condition(model_name, aug, *, Xtr, ytr, Xva, yva, Xte, yte, subte,
                  vae=None, n_syn=0, seed=SEED):
    """한 (분류기, 증강) 조건. 합성행은 train에만 — 검증·평가는 항상 실제 기록."""
    Xtr_f, ytr_f, cw = Xtr, ytr, None
    if aug == "vae":
        X_syn = sample_synthetic(vae, n_syn, seed=seed)
        Xtr_f = np.vstack([Xtr, X_syn])
        ytr_f = np.concatenate([ytr, np.full(len(X_syn), 2, dtype=ytr.dtype)])
    elif aug == "class_weight":
        cw = compute_class_weight("balanced", classes=np.array(LABELS), y=ytr)
    clf = train_classifier(model_name, Xtr_f, ytr_f, Xva, yva, class_weight=cw, seed=seed)
    proba = clf.predict_proba(Xte)
    rec = clf_metrics(yte, proba)
    y_sub, p_sub, _ = aggregate_subjects(subte, yte, proba)
    return rec, clf_metrics(y_sub, p_sub), proba, clf


# ══════════════════════════════════════════════════════════════════════════════
# 실험 A — 논문 절차 재구성 (행 단위 8:1:1, seed 반복, test·valid 이중 평가)
# ══════════════════════════════════════════════════════════════════════════════
def experiment_A(X_all, y_all, subject_all):
    t0 = time.time()
    print("\n" + "=" * 78)
    print("실험 A — 논문 절차 재구성 (행 8:1:1, 전처리 전체 적합=의도된 누수)")
    print("=" * 78)

    keep = outlier_keep(X_all)                      # IF rs=110 (논문 행 수와 일치)
    kept = {CLASS_ORDER[k]: int(((y_all == k) & keep).sum()) for k in LABELS}
    print(f"이상치 제거 후 {kept} | 논문 §5.1 {{'CN':7075,'MCI':3374,'Dem':515}} "
          f"(IF random_state={IF_RANDOM_STATE})")
    X_A, y_A, sub_A = X_all[keep], y_all[keep], subject_all[keep]
    Xs_A = StandardScaler().fit(X_A).transform(X_A)  # 전체 적합 — 논문 절차

    rows = []
    for si, seed in enumerate(SEEDS):
        set_seed(seed)
        idx = np.arange(len(Xs_A))
        i_tr, i_tmp = train_test_split(idx, test_size=0.2, stratify=y_A,
                                       random_state=seed)
        i_va, i_te = train_test_split(i_tmp, test_size=0.5, stratify=y_A[i_tmp],
                                      random_state=seed)
        if si == 0:
            print(f"train {len(i_tr)} / valid {len(i_va)} / test {len(i_te)} "
                  f"(논문 표 5: 8,771/1,098/1,095) | test Dem {int((y_A[i_te]==2).sum())}행"
                  f" (논문 51)")
            ov = len(set(sub_A[i_tr]) & set(sub_A[i_te]))
            print(f"train∩test 피험자 중복 {ov}명 — 행 분할의 구조적 누수 (의도된 재현)")
            print(f"VAE 레시피 {VAE_RECIPE}: latent {LATENT_DIM}, {VAE_EPOCHS} epoch")

        vae = fit_vae(Xs_A[i_tr][y_A[i_tr] == 2], seed=seed, verbose=(si == 0))
        print(f"[seed {seed}] VAE recon {vae['recon']:.4f} | "
              f"posterior μ-std {vae['posterior_mu_std']:.3f}", flush=True)
        synthetic_report(sample_synthetic(vae, 1000, seed=seed), vae["X_real"])

        for m in MODELS:
            for aug in ["none", "vae"]:
                rec, sub, _, clf = run_condition(
                    m, aug, Xtr=Xs_A[i_tr], ytr=y_A[i_tr],
                    Xva=Xs_A[i_va], yva=y_A[i_va],       # 명시적 valid (표 5)
                    Xte=Xs_A[i_te], yte=y_A[i_te], subte=sub_A[i_te],
                    vae=vae, n_syn=(N_SYNTHETIC_A if aug == "vae" else 0), seed=seed)
                # [v4 교정 2] 표 6 평가셋 불일치 대조용 — valid에서도 평가
                val_rec = clf_metrics(y_A[i_va], clf.predict_proba(Xs_A[i_va]))
                rows.append({"experiment": "A", "seed": seed, "model": m,
                             "augmentation": aug,
                             **flat("record", rec), **flat("valid", val_rec),
                             **flat("subject", sub)})
                print(f"  A[{seed}] {m:9s} {aug:4s} | test macro-F1 {rec['macro_f1']:.4f} "
                      f"Dem F1 {rec['Dem_f1']:.4f} | valid Dem F1 {val_rec['Dem_f1']:.4f}",
                      flush=True)

    A_all = pd.DataFrame(rows)
    num = A_all.select_dtypes("number").columns.drop("seed")
    A_mean = A_all.groupby(["experiment", "model", "augmentation"],
                           as_index=False)[list(num)].mean()
    print(f"실험 A 완료 — seed {SEEDS} · {(time.time() - t0) / 60:.1f}분")
    return A_all, A_mean


def report_paper_comparison(A_all):
    """논문 그림 3 + 표 6 대조 (seed 평균±표준편차, test·valid 병기)."""
    def ms(model, aug, col):
        v = A_all[(A_all.model == model) & (A_all.augmentation == aug)][col]
        return f"{v.mean():.4f}" + (f"±{v.std(ddof=0):.4f}" if len(v) > 1 else "")

    print("\n=== 논문 그림 3 대조 (기록 단위 F1, VAE 증강, test) ===")
    print(pd.DataFrame([{
        "모델": m,
        "CN(논문)": PAPER_F1_VAE[m]["CN"], "CN": ms(m, "vae", "record_CN_f1"),
        "MCI(논문)": PAPER_F1_VAE[m]["MCI"], "MCI": ms(m, "vae", "record_MCI_f1"),
        "Dem(논문)": PAPER_F1_VAE[m]["Dem"], "Dem": ms(m, "vae", "record_Dem_f1"),
        "평균(논문)": PAPER_F1_VAE[m]["Avg"], "평균": ms(m, "vae", "record_macro_f1"),
    } for m in MODELS]).to_string(index=False))

    print("\n=== 논문 표 6 대조 (Wide & Deep, test와 valid 병기) ===")
    print("표 6의 증강 전 행은 N=1,097~1,098(=valid), 증강 후 행은 N=1,095(=test)에서")
    print("측정된 정황이 있다(혼동행렬 역산) — 논문값이 어느 세트와 맞는지 함께 본다.")
    rows = []
    for aug in ("none", "vae"):
        p = PAPER_WD_TABLE6[aug]
        for prefix, setname in (("record", "test"), ("valid", "valid")):
            rows.append({
                "조건": "증강없음" if aug == "none" else "VAE증강", "세트": setname,
                "Dem F1(논문)": p["Dem_f1"], "Dem F1": ms("wide_deep", aug, f"{prefix}_Dem_f1"),
                "Dem R(논문)": p["Dem_recall"],
                "Dem R": ms("wide_deep", aug, f"{prefix}_Dem_recall"),
                "Dem P(논문)": p["Dem_precision"],
                "Dem P": ms("wide_deep", aug, f"{prefix}_Dem_precision"),
                "평균(논문)": p["Avg"], "평균": ms("wide_deep", aug, f"{prefix}_macro_f1"),
            })
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== 증강 효과 vs seed 변동 (test 기준) ===")
    eff = []
    for m in MODELS:
        for col, label in (("record_Dem_f1", "Dem F1"),
                           ("record_Dem_recall", "Dem recall"),
                           ("record_macro_f1", "macro F1")):
            piv = (A_all[A_all.model == m]
                   .pivot(index="seed", columns="augmentation", values=col))
            d = piv["vae"] - piv["none"]
            eff.append({"모델": m, "지표": label,
                        "증강없음": round(piv["none"].mean(), 4),
                        "VAE": round(piv["vae"].mean(), 4),
                        "Δ평균": round(d.mean(), 4),
                        "Δ표준편차": round(d.std(ddof=0), 4) if len(d) > 1 else np.nan,
                        "개선 seed": f"{int((d > 0).sum())}/{len(d)}"})
    print(pd.DataFrame(eff).to_string(index=False))
    print("Δ평균 절댓값 < Δ표준편차이면 그 효과는 seed 잡음과 구분되지 않는다.")


# ══════════════════════════════════════════════════════════════════════════════
# 실험 B — 피험자 층화 3-fold (증강 {없음, VAE, class weight}, seed 반복)
# ══════════════════════════════════════════════════════════════════════════════
def experiment_B(X_all, y_all, subject_all, subj_label):
    t0 = time.time()
    print("\n" + "=" * 78)
    print("실험 B — 피험자 단위 분할 (그 외 조건은 논문과 동일)")
    print("=" * 78)
    folds = subject_stratified_folds(subj_label, n_splits=3, seed=SEED)
    for k, (trs, tes) in enumerate(folds):
        lab_te = subj_label.loc[sorted(tes)]
        print(f"fold {k}: train {len(trs)}명 / eval {len(tes)}명 "
              f"(eval CN {int((lab_te == 0).sum())}, MCI {int((lab_te == 1).sum())}, "
              f"Dem {int((lab_te == 2).sum())})")

    rows, pool = [], {}
    for seed in SEEDS:
        for k, (trs, tes) in enumerate(folds):
            tr_mask = np.isin(subject_all, list(trs))
            te_mask = np.isin(subject_all, list(tes))
            keep_tr = outlier_keep(X_all[tr_mask])   # train fold에만 적합
            Xtr_raw = X_all[tr_mask][keep_tr]
            ytr = y_all[tr_mask][keep_tr]
            subtr = subject_all[tr_mask][keep_tr]
            scaler = StandardScaler().fit(Xtr_raw)
            Xtr = scaler.transform(Xtr_raw)
            Xte = scaler.transform(X_all[te_mask])
            yte, subte = y_all[te_mask], subject_all[te_mask]
            core, val = make_internal_val(subtr, ytr, seed=seed + k)
            X_dem = Xtr[core][ytr[core] == 2]
            vae = fit_vae(X_dem, seed=seed + k)
            n_syn = int(round(SYN_RATIO_B * len(X_dem)))
            if seed == SEEDS[0]:
                synthetic_report(sample_synthetic(vae, min(n_syn, 1000), seed=seed), X_dem)
            for m in MODELS:
                for aug in B_AUGS:
                    rec, sub, proba, _ = run_condition(
                        m, aug, Xtr=Xtr[core], ytr=ytr[core],
                        Xva=Xtr[val], yva=ytr[val],
                        Xte=Xte, yte=yte, subte=subte,
                        vae=vae, n_syn=(n_syn if aug == "vae" else 0), seed=seed + k)
                    rows.append({"experiment": "B", "seed": seed, "fold": k,
                                 "model": m, "augmentation": aug,
                                 "n_synthetic": n_syn if aug == "vae" else 0,
                                 **flat("record", rec), **flat("subject", sub)})
                    pool.setdefault((seed, m, aug), []).append((subte, yte, proba))
                    print(f"  B[{seed}] f{k} {m:9s} {aug:12s} | subj macro-F1 "
                          f"{sub['macro_f1']:.4f} Dem {sub['n_Dem_correct']}/{sub['n_Dem']}",
                          flush=True)

    B_fold_df = pd.DataFrame(rows)
    by_seed = {}
    for (seed, m, aug), parts in pool.items():
        subs = np.concatenate([p[0] for p in parts])
        ys = np.concatenate([p[1] for p in parts])
        pr = np.vstack([p[2] for p in parts])
        y_sub, p_sub, _ = aggregate_subjects(subs, ys, pr)
        met = clf_metrics(y_sub, p_sub)
        met.update(bootstrap_subject_ci(y_sub, p_sub))
        by_seed[f"{m}|{aug}|{seed}"] = met
    pooled = {}
    for m in MODELS:
        for aug in B_AUGS:
            ms_ = [by_seed[f"{m}|{aug}|{s}"] for s in SEEDS]
            agg = {k: float(np.mean([d[k] for d in ms_]))
                   for k, v in ms_[0].items() if isinstance(v, (int, float))}
            agg["macro_f1_std"] = float(np.std([d["macro_f1"] for d in ms_]))
            agg["macro_f1_ci"] = ms_[0]["macro_f1_ci"]
            pooled[f"{m}|{aug}"] = agg

    print(f"\n[실험 B pooled — 174명, seed {SEEDS} 평균]")
    print(pd.DataFrame([
        {"모델": m, "증강": a,
         "macro_F1": f"{pooled[f'{m}|{a}']['macro_f1']:.4f}"
                     + (f"±{pooled[f'{m}|{a}']['macro_f1_std']:.4f}" if len(SEEDS) > 1 else ""),
         "Dem 탐지": f"{pooled[f'{m}|{a}']['n_Dem_correct']:.1f}/12",
         "AUC": round(pooled[f'{m}|{a}']['macro_roc_auc_ovr'], 3)}
        for m in MODELS for a in B_AUGS]).to_string(index=False))
    print(f"실험 B 완료 — {(time.time() - t0) / 60:.1f}분")
    return B_fold_df, pooled, by_seed


# ══════════════════════════════════════════════════════════════════════════════
# 실험 C — Nested Group CV (outer 3 × inner 3, 격자 전량 평가)
# ══════════════════════════════════════════════════════════════════════════════
def experiment_C(X_all, y_all, subject_all, subj_label):
    t0 = time.time()
    print("\n" + "=" * 78)
    print("실험 C — Nested Group CV (분류기·증강을 inner CV가 선택)")
    print("=" * 78)
    clfs = ["xgboost", "dnn"] if QUICK_TEST else MODELS
    augs = ["none", "vae_r3"] if QUICK_TEST else ["none", "vae_r3", "vae_r9.71",
                                                  "class_weight"]
    cands = [(m, a) for m in clfs for a in augs]
    print(f"후보 {len(cands)}개 = 분류기 {clfs} × 증강 {augs}")

    def aug_spec(a):
        if a == "none":
            return "none", 0.0
        if a == "class_weight":
            return "class_weight", 0.0
        return "vae", float(a.split("_r")[1])

    def prep_fold(tr_subjects, te_subjects, seed):
        tr_mask = np.isin(subject_all, list(tr_subjects))
        te_mask = np.isin(subject_all, list(te_subjects))
        keep_tr = outlier_keep(X_all[tr_mask])
        Xtr_raw = X_all[tr_mask][keep_tr]
        ytr = y_all[tr_mask][keep_tr]
        subtr = subject_all[tr_mask][keep_tr]
        scaler = StandardScaler().fit(Xtr_raw)
        Xtr = scaler.transform(Xtr_raw)
        core, val = make_internal_val(subtr, ytr, seed=seed)
        X_dem = Xtr[core][ytr[core] == 2]
        return {"Xtr": Xtr[core], "ytr": ytr[core], "Xva": Xtr[val], "yva": ytr[val],
                "Xte": scaler.transform(X_all[te_mask]), "yte": y_all[te_mask],
                "subte": subject_all[te_mask], "vae": fit_vae(X_dem, seed=seed),
                "n_dem": len(X_dem)}

    folds = subject_stratified_folds(subj_label, n_splits=3, seed=SEED)
    sel_log, rows, pool = [], [], []
    for k, (otr, ote) in enumerate(folds):
        inner = subject_stratified_folds(subj_label.loc[sorted(otr)], n_splits=3,
                                         seed=SEED + 100 + k)
        scores = {c: [] for c in cands}
        for j, (itr, ite) in enumerate(inner):
            fj = prep_fold(itr, ite, seed=SEED + 10 * k + j)
            for cand in cands:
                m, a = cand
                aug, ratio = aug_spec(a)
                _, sub, _, _ = run_condition(
                    m, aug, Xtr=fj["Xtr"], ytr=fj["ytr"], Xva=fj["Xva"], yva=fj["yva"],
                    Xte=fj["Xte"], yte=fj["yte"], subte=fj["subte"],
                    vae=fj["vae"], n_syn=int(round(ratio * fj["n_dem"])),
                    seed=SEED + 10 * k + j)
                scores[cand].append(sub["macro_f1"])
            print(f"  C outer {k} inner {j} 완료 ({(time.time() - t0) / 60:.1f}분 경과)",
                  flush=True)
        best = max(cands, key=lambda c: float(np.mean(scores[c])))
        for cand in cands:
            sel_log.append({"outer_fold": k, "classifier": cand[0],
                            "augmentation": cand[1],
                            "inner_mean_subject_macro_f1": float(np.mean(scores[cand])),
                            "selected": cand == best})
        print(f"  C outer {k} 선택: {best[0]} + {best[1]} "
              f"(inner macro-F1 {float(np.mean(scores[best])):.4f})")
        fo = prep_fold(otr, ote, seed=SEED + k)
        m, a = best
        aug, ratio = aug_spec(a)
        rec, sub, proba, _ = run_condition(
            m, aug, Xtr=fo["Xtr"], ytr=fo["ytr"], Xva=fo["Xva"], yva=fo["yva"],
            Xte=fo["Xte"], yte=fo["yte"], subte=fo["subte"],
            vae=fo["vae"], n_syn=int(round(ratio * fo["n_dem"])), seed=SEED + k)
        rows.append({"experiment": "C", "fold": k, "selected_classifier": m,
                     "selected_augmentation": a,
                     **flat("record", rec), **flat("subject", sub)})
        pool.append((fo["subte"], fo["yte"], proba))
        print(f"  C outer {k} | subj macro-F1 {sub['macro_f1']:.4f} "
              f"Dem {sub['n_Dem_correct']}/{sub['n_Dem']}")

    C_outer_df = pd.DataFrame(rows)
    C_sel_df = pd.DataFrame(sel_log)
    subs = np.concatenate([p[0] for p in pool])
    ys = np.concatenate([p[1] for p in pool])
    pr = np.vstack([p[2] for p in pool])
    y_sub, p_sub, _ = aggregate_subjects(subs, ys, pr)
    C_pooled = clf_metrics(y_sub, p_sub)
    C_pooled.update(bootstrap_subject_ci(y_sub, p_sub))
    print(f"\n[실험 C pooled — {C_pooled['n']}명] macro-F1 {C_pooled['macro_f1']:.4f} "
          f"(95% CI {np.round(C_pooled['macro_f1_ci'], 3).tolist()}), "
          f"Dem {C_pooled['n_Dem_correct']}/{C_pooled['n_Dem']}")
    print(f"실험 C 완료 — {(time.time() - t0) / 60:.1f}분")
    return C_outer_df, C_sel_df, C_pooled


# ══════════════════════════════════════════════════════════════════════════════
# 시각화 (그림 4종 — Colab 셀 출력 + OUT_DIR PNG 저장)
# ══════════════════════════════════════════════════════════════════════════════
C_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
C_PAPER, C_TEXT, C_TEXT2 = "#52514e", "#0b0b0b", "#52514e"
C_GRID, C_SURFACE = "#e4e3df", "#fcfcfb"
MODEL_LABEL = {"xgboost": "XGBoost", "dnn": "DNN", "tabnet": "TabNet",
               "wide_deep": "Wide & Deep"}
MODEL_COLOR = dict(zip(MODELS, C_SERIES))


def _style():
    plt.rcParams.update({
        "figure.facecolor": C_SURFACE, "axes.facecolor": C_SURFACE,
        "axes.edgecolor": C_GRID, "axes.labelcolor": C_TEXT,
        "text.color": C_TEXT, "xtick.color": C_TEXT2, "ytick.color": C_TEXT2,
        "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "figure.dpi": 110})


def _finish(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(out_dir / name, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def fig1_paper_vs_A(A_mean, A_all, out_dir):
    classes = ["CN", "MCI", "Dem", "Avg"]
    a_vae = A_mean[A_mean["augmentation"] == "vae"].set_index("model")
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6), sharex=True, sharey=True)
    for ax, cls in zip(axes, classes):
        col = "record_macro_f1" if cls == "Avg" else f"record_{cls}_f1"
        for i, m in enumerate(MODELS):
            yy = len(MODELS) - 1 - i
            pv = PAPER_F1_VAE[m][cls]
            ov = float(a_vae.loc[m][col])
            ax.plot([pv, ov], [yy, yy], color=C_GRID, lw=2, zorder=1)
            if A_all["seed"].nunique() > 1:
                v = A_all[(A_all.model == m) & (A_all.augmentation == "vae")][col]
                ax.plot([v.min(), v.max()], [yy, yy], color=MODEL_COLOR[m], lw=5,
                        alpha=0.25, solid_capstyle="butt", zorder=2)
            ax.scatter([ov], [yy], s=52, color=MODEL_COLOR[m], zorder=3)
            ax.annotate(f"{ov:.3f}", (ov, yy), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=8,
                        color=MODEL_COLOR[m])
            ax.scatter([pv], [yy], s=52, facecolor=C_SURFACE, edgecolor=C_PAPER,
                       linewidth=1.6, zorder=2)
            ax.annotate(f"{pv:.3f}", (pv, yy), xytext=(0, -15),
                        textcoords="offset points", ha="center", fontsize=8,
                        color=C_PAPER)
        ax.set_title({"Avg": "Macro avg"}.get(cls, cls), fontsize=11)
        ax.set_ylim(-0.7, len(MODELS) - 0.3)
        ax.set_xlim(0.3, 1.02)
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([MODEL_LABEL[m] for m in MODELS[::-1]])
        ax.grid(axis="x")
        ax.grid(False, axis="y")
    handles = [plt.Line2D([], [], marker="o", ls="", mfc=C_SURFACE, mec=C_PAPER,
                          label="Paper (Fig.3/Table 6)"),
               plt.Line2D([], [], marker="o", ls="", color=C_TEXT2,
                          label="v4, mean over seeds"),
               plt.Line2D([], [], color=C_TEXT2, lw=5, alpha=0.25, label="seed min-max")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.12))
    fig.suptitle("Record-level F1 (VAE-augmented): paper vs v4", y=1.2, fontsize=13)
    _finish(fig, out_dir, "fig1_paper_vs_A.png")


def fig2_designs(A_mean, B_pooled, C_pooled, out_dir):
    augs = [a for a in ["none", "vae", "class_weight"]
            if (A_mean["augmentation"] == a).any()
            or any(k.endswith(f"|{a}") for k in B_pooled)]
    xpos = {"A": 0, "B": 1, "C": 2}
    fig, axes = plt.subplots(1, len(augs), figsize=(5.6 * len(augs), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, aug in zip(axes, augs):
        ends = []
        for m in MODELS:
            xs, ys = [], []
            row = A_mean[(A_mean.model == m) & (A_mean.augmentation == aug)]
            if len(row):
                xs.append(xpos["A"])
                ys.append(float(row["subject_macro_f1"].iloc[0]))
            if f"{m}|{aug}" in B_pooled:
                xs.append(xpos["B"])
                ys.append(float(B_pooled[f"{m}|{aug}"]["macro_f1"]))
            if xs:
                ax.plot(xs, ys, marker="o", ms=6, lw=2, color=MODEL_COLOR[m],
                        label=MODEL_LABEL[m])
                ends.append((xs[-1], ys[-1], f"{ys[-1]:.2f}", MODEL_COLOR[m]))
        ends.sort(key=lambda t: t[1], reverse=True)
        prev = None
        for lx, ly, txt, lc in ends:
            yy = ly if prev is None else min(ly, prev - 0.045)
            ax.text(lx + 0.08, yy, txt, va="center", fontsize=8, color=lc)
            prev = yy
        if C_pooled is not None:
            ax.scatter([xpos["C"]], [C_pooled["macro_f1"]], marker="D", s=64,
                       color=C_PAPER, zorder=3)
            ax.annotate(f"{C_pooled['macro_f1']:.2f}", (xpos["C"], C_pooled["macro_f1"]),
                        xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=8, color=C_PAPER)
        ax.set_xticks(list(xpos.values()))
        ax.set_xticklabels(["A\nrow split", "B\nsubject split", "C\nnested CV"])
        ax.set_xlim(-0.4, 2.5)
        ax.set_ylim(0, 1)
        ax.set_title(f"augmentation: {aug}", fontsize=11)
        ax.grid(axis="y")
        ax.grid(False, axis="x")
    axes[0].set_ylabel("Subject-level macro-F1")
    axes[-1].legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("Validation design vs subject-level macro-F1", fontsize=13)
    _finish(fig, out_dir, "fig2_designs.png")


def fig3_dem_detection(A_mean, B_pooled, C_pooled, out_dir):
    import matplotlib.patches as mpatches
    augs = [a for a in ["none", "vae", "class_weight"]
            if (A_mean["augmentation"] == a).any()
            or any(k.endswith(f"|{a}") for k in B_pooled)]
    w = 0.38
    x = np.arange(len(MODELS))
    fig, axes = plt.subplots(1, len(augs), figsize=(5.9 * len(augs), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, aug in zip(axes, augs):
        for j, m in enumerate(MODELS):
            row = A_mean[(A_mean.model == m) & (A_mean.augmentation == aug)]
            if len(row):
                r = row.iloc[0]
                v = float(r["subject_Dem_recall"])
                ax.bar(j - w / 2, v, w * 0.92, color=MODEL_COLOR[m], zorder=2)
                ax.annotate(f"{r['subject_n_Dem_correct']:.1f}/{r['subject_n_Dem']:.0f}",
                            (j - w / 2, v), xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=8)
            if f"{m}|{aug}" in B_pooled:
                d = B_pooled[f"{m}|{aug}"]
                ax.bar(j + w / 2, d["Dem_recall"], w * 0.92, facecolor=C_SURFACE,
                       edgecolor=MODEL_COLOR[m], hatch="//", linewidth=1.2, zorder=2)
                ax.annotate(f"{d['n_Dem_correct']:.1f}/12", (j + w / 2, d["Dem_recall"]),
                            xytext=(0, 3), textcoords="offset points", ha="center",
                            fontsize=8)
        if C_pooled is not None:
            ax.axhline(C_pooled["Dem_recall"], color=C_PAPER, lw=1.4, ls="--", zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS], fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_title(f"augmentation: {aug}", fontsize=11)
        ax.grid(axis="y")
        ax.grid(False, axis="x")
    axes[0].set_ylabel("Subject-level Dem recall")
    handles = [mpatches.Patch(facecolor=C_TEXT2, label="A (row split)"),
               mpatches.Patch(facecolor=C_SURFACE, edgecolor=C_TEXT2, hatch="//",
                              label="B (subject split, 12 Dem)")]
    if C_pooled is not None:
        handles.append(plt.Line2D([], [], color=C_PAPER, ls="--",
                                  label=f"C nested: {C_pooled['n_Dem_correct']}/12"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.09), fontsize=9)
    fig.suptitle("How many real Dem subjects are detected?", y=1.16, fontsize=13)
    _finish(fig, out_dir, "fig3_dem_detection.png")


def fig4_dispersion(A_all, B_fold_df, C_outer_df, out_dir):
    cols = []
    cols.append(("A\nrow split\n(pipelines x seeds)", [
        (float(r["subject_macro_f1"]), MODEL_COLOR.get(r["model"], C_PAPER),
         r["augmentation"] == "vae", False) for _, r in A_all.iterrows()]))
    cols.append(("B\nsubject split\n(folds x pipelines)", [
        (float(r["subject_macro_f1"]), MODEL_COLOR.get(r["model"], C_PAPER),
         r["augmentation"] == "vae", False) for _, r in B_fold_df.iterrows()]))
    cols.append(("C\nnested CV\n(outer folds)", [
        (float(r["subject_macro_f1"]), C_PAPER, True, True)
        for _, r in C_outer_df.iterrows()]))
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for xc, (_label, pts) in enumerate(cols):
        for v, c, filled, diamond in pts:
            ax.scatter([xc + rng.uniform(-0.14, 0.14)], [v],
                       s=52 if diamond else 40, marker="D" if diamond else "o",
                       facecolor=c if filled else C_SURFACE, edgecolor=c,
                       linewidth=1.2, zorder=2, alpha=0.9)
        med = float(np.median([v for v, *_ in pts]))
        ax.hlines(med, xc - 0.26, xc + 0.26, color=C_TEXT, lw=2, zorder=3)
        ax.annotate(f"median {med:.2f}", (xc + 0.3, med), va="center", fontsize=9,
                    color=C_TEXT)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([lab for lab, _ in cols], fontsize=9)
    ax.set_xlim(-0.5, len(cols) - 0.5 + 0.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Subject-level macro-F1")
    ax.grid(axis="y")
    ax.grid(False, axis="x")
    handles = [plt.Line2D([], [], marker="o", ls="", color=MODEL_COLOR[m],
                          label=MODEL_LABEL[m]) for m in MODELS]
    handles += [plt.Line2D([], [], marker="o", ls="", color=C_TEXT2,
                           label="filled = VAE aug"),
                plt.Line2D([], [], marker="o", ls="", mfc=C_SURFACE, mec=C_TEXT2,
                           label="open = no aug / class weight"),
                plt.Line2D([], [], marker="D", ls="", color=C_PAPER,
                           label="C outer fold")]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, fontsize=8)
    ax.set_title("Dispersion across folds and pipelines", fontsize=13)
    _finish(fig, out_dir, "fig4_dispersion.png")


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    t_all = time.time()
    set_seed()
    print(f"device: {DEVICE} | torch {torch.__version__} | QUICK_TEST={QUICK_TEST}")
    data_dir, out_dir = resolve_paths()
    print(f"DATA_DIR : {data_dir}")
    print(f"OUT_DIR  : {out_dir}")
    X_all, y_all, subject_all, subj_label = load_data(data_dir)

    if RUN_IF_SWEEP:
        if_seed_sweep(X_all, y_all)

    A_all, A_mean = experiment_A(X_all, y_all, subject_all)
    report_paper_comparison(A_all)
    B_fold_df, B_pooled, B_by_seed = experiment_B(X_all, y_all, subject_all, subj_label)
    C_outer_df, C_sel_df, C_pooled = experiment_C(X_all, y_all, subject_all, subj_label)

    # 저장
    A_all.to_csv(out_dir / "A_metrics.csv", index=False)
    A_mean.to_csv(out_dir / "A_metrics_seed_mean.csv", index=False)
    B_fold_df.to_csv(out_dir / "B_fold_metrics.csv", index=False)
    (out_dir / "B_pooled_subject_metrics.json").write_text(
        json.dumps(B_pooled, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "B_pooled_by_seed.json").write_text(
        json.dumps(B_by_seed, ensure_ascii=False, indent=1), encoding="utf-8")
    C_outer_df.to_csv(out_dir / "C_outer_metrics.csv", index=False)
    C_sel_df.to_csv(out_dir / "C_selection_log.csv", index=False)
    (out_dir / "C_pooled_subject_metrics.json").write_text(
        json.dumps(C_pooled, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps({
        "version": "v4_single_entrypoint", "seed": SEED, "seeds": SEEDS,
        "quick_test": QUICK_TEST, "if_random_state": IF_RANDOM_STATE,
        "vae_recipe": VAE_RECIPE, "latent_dim": LATENT_DIM, "vae_epochs": VAE_EPOCHS,
        "n_synthetic_A": N_SYNTHETIC_A, "syn_ratio_B": SYN_RATIO_B,
        "xgb_n_estimators": XGB_TREES,
        "tabnet": {"scheduler": TABNET_SCHEDULER, "patience": TABNET_PATIENCE,
                   "batch": TABNET_BATCH, "early_stop_metric": "balanced_accuracy"},
        "models": MODELS, "b_augs": B_AUGS,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    # 그림
    _style()
    fig1_paper_vs_A(A_mean, A_all, out_dir)
    fig2_designs(A_mean, B_pooled, C_pooled, out_dir)
    fig3_dem_detection(A_mean, B_pooled, C_pooled, out_dir)
    fig4_dispersion(A_all, B_fold_df, C_outer_df, out_dir)

    print(f"\n산출물 저장 → {out_dir}")
    for p in sorted(out_dir.iterdir()):
        print("  ", p.name)
    print(f"전체 완료 — {(time.time() - t_all) / 60:.1f}분")
    print("\n해석 주의: Dem은 독립 피험자 12명이다. 합성행은 새 피험자가 아니며,")
    print("피험자 단위 지표의 분모는 항상 실제 피험자 수다. QUICK_TEST 결과는 인용 금지.")


if __name__ == "__main__":
    main()
