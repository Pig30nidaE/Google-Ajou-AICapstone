"""
자체 최종 모델(EXPERIMENT_LOG.md 24~26절) 검증

주장: CN+MCI(162) vs Dementia(11, nia+219 제외), 피처 `activity_low_std` 단독,
      로지스틱회귀 -> ROC-AUC 0.9087

검증 방법 (YMJ 논문 검증에 쓴 것과 동일):
  A) 재현 : 그 설정 그대로 AUC 계산 + Bootstrap CI
  B) Permutation test #1 (피처 고정)
       -> 'activity_low_std 하나만 쓴다'가 사전에 정해진 것이라면 이게 올바른 검정
  C) Permutation test #2 (피처 선택 과정까지 포함)
       -> 실제로는 220개 후보에서 SHAP/AUC 스윕으로 이 피처를 '골랐으므로',
          라벨을 섞은 뒤에도 '똑같이 220개에서 최고를 고르는' 과정을 반복해야
          공정한 귀무분포가 된다.
       -> B와 C의 차이가 곧 '피처를 골랐다는 사실'이 만들어낸 낙관 편향.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
DATA = HERE / "patient_level_all_v2.csv"
EXCLUDE = "nia+219@rowan.kr"
CLAIMED_FEATURE = "activity_low_std"
DROP = {"EMAIL", "date", "DIAG_NM", "original_label", "label", "fold"}
N_SPLITS, N_REPEAT = 5, 20
N_PERM = 1000
SEED = 42


def load():
    df = pd.read_csv(DATA)
    df = df[df["EMAIL"] != EXCLUDE].reset_index(drop=True)
    y = (df["original_label"] == 2).astype(int).values      # CN+MCI=0, Dem=1
    feats = [c for c in df.columns if c not in DROP and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    return X, y, feats


def cv_auc(x1d, y, n_repeat=N_REPEAT):
    """단일(또는 소수) 피처 로지스틱회귀의 반복 층화 CV 평균 AUC."""
    X = x1d.reshape(-1, 1) if x1d.ndim == 1 else x1d
    aucs = []
    for r in range(n_repeat):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + r)
        oof = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            sc = StandardScaler()
            Xtr, Xte = sc.fit_transform(X[tr]), sc.transform(X[te])
            m = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
            m.fit(Xtr, y[tr])
            oof[te] = m.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(y, oof))
    return float(np.mean(aucs)), float(np.std(aucs))


def pick_best_feature(X, y, feats):
    """220개 후보 중 CV AUC가 가장 높은 단일 피처를 고른다 (사용자가 한 스윕의 재현)."""
    best, best_auc = None, -1
    Xv = X.to_numpy()
    for i, f in enumerate(feats):
        a, _ = cv_auc(Xv[:, i], y, n_repeat=3)      # 탐색은 3회 반복으로 경량화
        if a > best_auc:
            best_auc, best = a, f
    return best, best_auc


def main():
    X, y, feats = load()
    print(f"n={len(y)}  (CN+MCI={int((y==0).sum())}, Dem={int((y==1).sum())})  후보 피처={len(feats)}")
    print(f"검증 대상 주장: '{CLAIMED_FEATURE}' 단독 -> AUC 0.9087\n")

    # ---------- A) 재현 ----------
    xi = feats.index(CLAIMED_FEATURE)
    obs, sd = cv_auc(X.to_numpy()[:, xi], y)
    print("=" * 84)
    print("(A) 주장 재현")
    print("=" * 84)
    print(f"  {CLAIMED_FEATURE} 단독 CV AUC = {obs:.4f} ± {sd:.4f}   (주장값 0.9087)")

    rng = np.random.default_rng(7)
    xcol = X.to_numpy()[:, xi]
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() < 2:
            continue
        boots.append(roc_auc_score(y[idx], xcol[idx]))
    boots = np.array(boots)
    print(f"  Bootstrap 95% CI (단변량 AUC 기준) = "
          f"[{np.percentile(boots,2.5):.3f}, {np.percentile(boots,97.5):.3f}]")

    # ---------- B) permutation (피처 고정) ----------
    print("\n" + "=" * 84)
    print(f"(B) Permutation #1 — 피처를 '{CLAIMED_FEATURE}'로 고정한 채 라벨만 셔플 (N={N_PERM})")
    print("=" * 84)
    nullB = []
    for i in range(N_PERM):
        yp = rng.permutation(y)
        a, _ = cv_auc(xcol, yp, n_repeat=3)
        nullB.append(a)
        if (i + 1) % 250 == 0:
            print(f"  [{i+1}/{N_PERM}] null mean={np.mean(nullB):.4f}")
    nullB = np.array(nullB)
    pB = (1 + np.sum(nullB >= obs)) / (len(nullB) + 1)
    print(f"  귀무분포: {nullB.mean():.4f} ± {nullB.std():.4f}  (95%ile={np.percentile(nullB,95):.4f})")
    print(f"  p-value = {pB:.4f}")

    # ---------- C) permutation (피처 선택 과정 포함) ----------
    print("\n" + "=" * 84)
    print(f"(C) Permutation #2 — 라벨 셔플 후 '220개에서 최고 피처 고르기'까지 반복")
    print("=" * 84)
    best_real, best_real_auc = pick_best_feature(X, y, feats)
    print(f"  실제 데이터에서 스윕이 고르는 최고 피처: {best_real}  (AUC={best_real_auc:.4f})")

    N_PERM_C = 200      # 선택과정 포함이라 비용이 크므로 200회
    nullC = []
    for i in range(N_PERM_C):
        yp = rng.permutation(y)
        _, a = pick_best_feature(X, yp, feats)
        nullC.append(a)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{N_PERM_C}] null mean={np.mean(nullC):.4f}")
    nullC = np.array(nullC)
    pC = (1 + np.sum(nullC >= best_real_auc)) / (len(nullC) + 1)
    print(f"  귀무분포: {nullC.mean():.4f} ± {nullC.std():.4f}  (95%ile={np.percentile(nullC,95):.4f})")
    print(f"  p-value = {pC:.4f}")

    print("\n" + "=" * 84)
    print("[해석]")
    print(f"  (B) 피처가 사전에 정해져 있었다면      : p={pB:.4f}")
    print(f"  (C) 220개에서 골라낸 것이라면          : p={pC:.4f}")
    print(f"  선택 과정이 만드는 낙관 편향(귀무 상승): {nullC.mean()-nullB.mean():+.4f} AUC")

    out = {"observed_auc": obs, "observed_sd": sd,
           "bootstrap_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
           "permB": {"null_mean": float(nullB.mean()), "null_sd": float(nullB.std()),
                     "p_value": float(pB), "n": int(len(nullB))},
           "permC": {"best_feature": best_real, "best_auc": float(best_real_auc),
                     "null_mean": float(nullC.mean()), "null_sd": float(nullC.std()),
                     "p_value": float(pC), "n": int(len(nullC))}}
    (HERE / "own_final_model_verification.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {HERE/'own_final_model_verification.json'}")


if __name__ == "__main__":
    main()
