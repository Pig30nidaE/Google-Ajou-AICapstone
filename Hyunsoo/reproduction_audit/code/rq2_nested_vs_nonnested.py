"""
연구문제 2: Non-nested vs Repeated Nested 검증 비교
────────────────────────────────────────────────────────────────────────────
피험자 독립성이 이미 확보된 조건(1인 1행)에서, 모델 선택과 최종 평가를 분리하면
성능 추정치가 어떻게 달라지는가?

■ 분석자료
    피험자별 1인 1행 장기 특성 자료 (n=162, CN 111 / MCI 51)
    - 원본 수면 raw → 일별 파생지표 20종 → 환자단위 5계열 집계 (339 피처)
    - 이미 환자 단위이므로 '같은 사람이 train/test에 동시 등장'하는 누수는 원천적으로 없음
    - 따라서 여기서 관측되는 차이는 오직 '모델 선택과 평가의 분리 여부'에서만 발생

■ 비교조건
    (A) Non-nested Stratified K-fold
        전체 자료로 변수선택 + 하이퍼파라미터 + 클래스가중치를 고른 뒤,
        같은 자료의 K-fold 성능을 그대로 보고 (= 선택과 평가가 분리되지 않음)

    (B) Repeated Nested Stratified K-fold
        Inner CV  : 변수 선택(RFE), 하이퍼파라미터 탐색(C), 클래스 가중치 탐색
        Outer CV  : 모델 선택에 전혀 쓰이지 않은 피험자로 최종 평가
        위를 R회 반복하여 분포를 확보

■ 보고내용
    Accuracy, Precision, Recall, F1, ROC-AUC
    각각 평균 / 표준편차 / 95% 신뢰구간

■ 설계 메모
    - 임계값은 논문 관례에 따라 0.5 고정 (Suppl. Table 5: "at the fixed 0.5 threshold")
    - RFE 순위는 각 inner-train 안에서 기준 추정기(C=1.0, balanced)로 1회 산출한 뒤
      그 순위 위에서 C·class_weight·k를 탐색 (순위 산출과 최종 모델 튜닝을 분리).
      순위 산출 자체도 inner-train 안에서만 이루어지므로 outer-test 누수는 없음.
"""
from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
DATA = HERE / "ymj_faithful_features.csv"
META = {"EMAIL", "n_nights", "DIAG_NM", "original_label", "mci_label"}

N_OUTER, N_INNER = 10, 5
N_REPEAT = 10
THRESH = 0.5
BASE_SEED = 42

# 논문이 명시하지 않은 부분 = 탐색 대상
C_GRID = [0.1, 0.3, 1.0, 3.0]
CW_GRID = [None, "balanced"]
K_GRID = [20, 30, 50]

FAMILY_SUFFIXES = {
    "M": ["_mean"], "EM": ["_median", "_trimmed_mean", "_mode"],
    "Dist": ["_min", "_max", "_mad", "_kurtosis", "_range"],
    "Disp": ["_sd", "_cv", "_iqr"],
    "TS": ["_stv", "_tbv", "_rcv", "_mr", "_tbcr"],
}
ALL_SUF = sorted({s for v in FAMILY_SUFFIXES.values() for s in v}, key=len, reverse=True)

EXPERIMENTS = [
    ("Exp.10", ["M", "EM", "Dist", "Disp", "TS"], 0.859),
    ("Exp.11", ["EM", "Dist", "TS"], 0.861),          # 논문 최고 성능 메인모델
    ("Exp.12", ["EM", "Dist", "Disp", "TS"], 0.831),
]


def resolve(cols, fams):
    want = set(itertools.chain.from_iterable(FAMILY_SUFFIXES[f] for f in fams))
    return [c for c in cols
            if (m := next((s for s in ALL_SUF if c.endswith(s)), None)) is not None and m in want]


def lr(C, cw):
    return LogisticRegression(penalty="l2", C=C, class_weight=cw,
                              max_iter=5000, random_state=BASE_SEED)


def rfe_order(Xtr, ytr, C, cw):
    """주어진 (C, class_weight) 추정기로 RFE 전체 순위 산출 → top-k 슬라이싱용.
    변수 선택 자체가 하이퍼파라미터·클래스가중치에 의존하므로, 순위도 설정별로 구한다."""
    sel = RFE(lr(C, cw), n_features_to_select=1, step=5)
    sel.fit(Xtr, ytr)
    return np.argsort(sel.ranking_)          # 중요도 높은 순 인덱스


def metrics(y, prob):
    pred = (prob >= THRESH).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob),
    }


# ───────────────────────── (A) Non-nested ─────────────────────────
def non_nested(X, y, seed):
    """전체 자료로 전처리·변수선택·하이퍼파라미터를 확정한 뒤 K-fold 성능 보고."""
    Xn = X.to_numpy()
    Xs = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(Xn))
    skf = StratifiedKFold(N_OUTER, shuffle=True, random_state=seed)

    # (C, cw)별 RFE 순위를 전체 자료로 미리 산출 ← 전체 라벨 사용 (non-nested의 본질)
    order_cache = {(C, cw): rfe_order(Xs, y, C, cw)
                   for C, cw in itertools.product(C_GRID, CW_GRID)}

    best = None
    for C, cw, k in itertools.product(C_GRID, CW_GRID, K_GRID):
        idx = order_cache[(C, cw)][:min(k, Xs.shape[1])]
        oof = np.zeros(len(y))
        for tr, te in skf.split(Xs, y):
            m = lr(C, cw); m.fit(Xs[tr][:, idx], y[tr])
            oof[te] = m.predict_proba(Xs[te][:, idx])[:, 1]
        auc = roc_auc_score(y, oof)
        if best is None or auc > best[0]:
            best = (auc, C, cw, k, oof)                        # ← 같은 자료로 최고를 고름
    auc, C, cw, k, oof = best
    out = metrics(y, oof)
    out.update({"C": C, "class_weight": str(cw), "k": k})
    return out


# ─────────────────── (B) Repeated Nested ───────────────────
def nested(X, y, seed):
    """Inner: 변수선택+하이퍼파라미터+클래스가중치 / Outer: 미사용 피험자로 평가."""
    Xn = X.to_numpy()
    outer = StratifiedKFold(N_OUTER, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    picks = []

    for tr, te in outer.split(Xn, y):
        imp = SimpleImputer(strategy="median")
        Xtr, Xte = imp.fit_transform(Xn[tr]), imp.transform(Xn[te])
        sc = StandardScaler()
        Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)
        ytr = y[tr]

        inner = StratifiedKFold(N_INNER, shuffle=True, random_state=seed)
        folds = list(inner.split(Xtr, ytr))

        # inner fold × (C, cw) 별 RFE 순위 캐시 — 전부 inner-train 안에서만 산출
        orders = {(fi, C, cw): rfe_order(Xtr[i_tr], ytr[i_tr], C, cw)
                  for fi, (i_tr, _) in enumerate(folds)
                  for C, cw in itertools.product(C_GRID, CW_GRID)}

        best_cfg, best_score = None, -1
        for C, cw, k in itertools.product(C_GRID, CW_GRID, K_GRID):
            scores = []
            for fi, (i_tr, i_va) in enumerate(folds):
                idx = orders[(fi, C, cw)][:min(k, Xtr.shape[1])]
                m = lr(C, cw); m.fit(Xtr[i_tr][:, idx], ytr[i_tr])
                p = m.predict_proba(Xtr[i_va][:, idx])[:, 1]
                try:
                    scores.append(roc_auc_score(ytr[i_va], p))
                except ValueError:
                    pass
            s = float(np.mean(scores)) if scores else np.nan
            if not np.isnan(s) and s > best_score:
                best_score, best_cfg = s, (C, cw, k)

        C, cw, k = best_cfg
        picks.append({"C": C, "class_weight": str(cw), "k": k})
        od_full = rfe_order(Xtr, ytr, C, cw)                    # outer-train 전체로 재선택
        idx = od_full[:min(k, Xtr.shape[1])]
        m = lr(C, cw); m.fit(Xtr[:, idx], ytr)
        oof[te] = m.predict_proba(Xte[:, idx])[:, 1]

    out = metrics(y, oof)
    out["picks"] = picks
    return out


def summarize(runs, keys=("accuracy", "precision", "recall", "f1", "roc_auc")):
    s = {}
    for k in keys:
        v = np.array([r[k] for r in runs], dtype=float)
        n = len(v)
        mean, sd = float(v.mean()), float(v.std(ddof=1)) if n > 1 else 0.0
        if n > 1:
            half = st.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        else:
            half = 0.0
        s[k] = {"mean": mean, "sd": sd,
                "ci95": [float(mean - half), float(mean + half)],
                "min": float(v.min()), "max": float(v.max())}
    return s


def main():
    df = pd.read_csv(DATA)
    y = df["mci_label"].astype(int).values
    cols = [c for c in df.columns if c not in META]

    print("=" * 100)
    print("연구문제 2 : Non-nested vs Repeated Nested Stratified K-fold")
    print("=" * 100)
    print(f"분석자료  : 피험자별 1인 1행, n={len(y)} (CN={int((y==0).sum())}, MCI={int((y==1).sum())})")
    print(f"탐색공간  : C={C_GRID}, class_weight={CW_GRID}, RFE k={K_GRID}  (총 "
          f"{len(C_GRID)*len(CW_GRID)*len(K_GRID)}개 조합)")
    print(f"검증구조  : Outer {N_OUTER}-fold × {N_REPEAT}회 반복, Inner {N_INNER}-fold")
    print(f"임계값    : {THRESH} 고정\n")

    results = {}
    for name, fams, paper in EXPERIMENTS:
        X = df[resolve(cols, fams)]
        tag = " ← 논문 최고 성능 메인모델" if name == "Exp.11" else ""
        print("─" * 100)
        print(f"[{name}] {'+'.join(fams)}  (후보 {X.shape[1]}개, 논문 보고 AUC {paper:.3f}){tag}")
        print("─" * 100)

        nn_runs, ne_runs = [], []
        for r in range(N_REPEAT):
            seed = BASE_SEED + r
            nn_runs.append(non_nested(X, y, seed))
            ne_runs.append(nested(X, y, seed))
            print(f"   repeat {r+1:2d}/{N_REPEAT}  non-nested AUC={nn_runs[-1]['roc_auc']:.4f}  "
                  f"nested AUC={ne_runs[-1]['roc_auc']:.4f}")

        nn_s, ne_s = summarize(nn_runs), summarize(ne_runs)
        results[name] = {"paper_auc": paper, "n_candidates": int(X.shape[1]),
                         "non_nested": nn_s, "nested": ne_s,
                         "non_nested_runs": [{k: v for k, v in r.items() if k != "picks"} for r in nn_runs],
                         "nested_runs": [{k: v for k, v in r.items() if k != "picks"} for r in ne_runs],
                         "nested_picks": [r["picks"] for r in ne_runs]}

        print()
        hdr = "   {:<12}{:>26}{:>20}{:>24}{:>20}{:>9}".format(
            "지표", "Non-nested (평균±SD)", "95% CI", "Nested (평균±SD)", "95% CI", "차이")
        print(hdr)
        for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            a, b = nn_s[k], ne_s[k]
            a_ms = "{:.4f} ± {:.4f}".format(a["mean"], a["sd"])
            a_ci = "[{:.3f}, {:.3f}]".format(a["ci95"][0], a["ci95"][1])
            b_ms = "{:.4f} ± {:.4f}".format(b["mean"], b["sd"])
            b_ci = "[{:.3f}, {:.3f}]".format(b["ci95"][0], b["ci95"][1])
            print("   {:<12}{:>26}{:>20}{:>24}{:>20}{:>+9.4f}".format(
                k, a_ms, a_ci, b_ms, b_ci, a["mean"] - b["mean"]))

        # non-nested가 고른 설정 분포
        from collections import Counter
        cfgs = Counter((r["C"], r["class_weight"], r["k"]) for r in nn_runs)
        print(f"\n   non-nested가 고른 설정: {dict(cfgs)}")
        pk = Counter((p["C"], p["class_weight"], p["k"]) for run in results[name]["nested_picks"] for p in run)
        print(f"   nested inner-CV가 고른 설정(전체 fold): {dict(pk.most_common(5))}")
        print()

    (HERE / "rq2_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("=" * 100)
    print("[종합]")
    print("{:<9}{:>8}{:>20}{:>20}{:>10}".format(
        "Exp", "논문", "Non-nested AUC", "Nested AUC", "낙관편향"))
    for name, _, paper in EXPERIMENTS:
        a = results[name]["non_nested"]["roc_auc"]; b = results[name]["nested"]["roc_auc"]
        print("{:<9}{:>8.3f}{:>20}{:>20}{:>+10.4f}".format(
            name, paper,
            "{:.4f}±{:.4f}".format(a["mean"], a["sd"]),
            "{:.4f}±{:.4f}".format(b["mean"], b["sd"]),
            a["mean"] - b["mean"]))
    print(f"\nSaved to {HERE/'rq2_results.json'}")


if __name__ == "__main__":
    main()
