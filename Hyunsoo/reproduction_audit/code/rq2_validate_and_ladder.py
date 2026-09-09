"""
재현 타당성 검증 + 누수 사다리(leakage ladder)
────────────────────────────────────────────────────────────────────────────
문제의식: 내 non-nested 재현(AUC 0.888)이 논문 보고값(0.859)을 '넘어섰다'.
          누수 있는 파이프라인을 재현했다면 논문 근처에 떨어져야지 넘어서면 안 된다.
          → 내가 논문에 없는 선택 단계(하이퍼파라미터 탐색)를 추가로 얹었을 가능성.

■ PART 1 : 논문 Supplementary Table 6 재현 (재현 방식이 옳은지 외부 검증)
    논문은 Exp.10 피처셋으로 선형 4종 vs 비선형 3종을 비교하고 수치를 공개했다.
    이 표는 '내 파이프라인이 논문과 같은 자리에 있는지' 확인할 독립적 기준점이다.
    특히 비선형 모델의 극단적 붕괴(Sensitivity 0.04~0.22)가 재현되는지가 핵심.

■ PART 2 : 누수 사다리
    L0  Full nested            : 변수선택 + 하이퍼파라미터 전부 inner CV 안
    L1  RFE만 CV 밖 (기본 파라미터 고정, 탐색 없음)
    L2  RFE + 하이퍼파라미터 탐색 모두 CV 밖 (= 내가 앞서 한 것)
    논문 보고값이 L0~L2 중 어디에 놓이는지 보면, 논문이 실제로 어느 단계까지
    분리했는지 역추정할 수 있다.
"""
from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
DATA = HERE / "ymj_faithful_features.csv"
META = {"EMAIL", "n_nights", "DIAG_NM", "original_label", "mci_label"}
N_OUTER, N_INNER = 10, 5
N_REPEAT = 10
SEED = 42
THRESH = 0.5

C_GRID = [0.1, 0.3, 1.0, 3.0]
CW_GRID = [None, "balanced"]
K_GRID = [20, 30, 50]

FAM = {"M": ["_mean"], "EM": ["_median", "_trimmed_mean", "_mode"],
       "Dist": ["_min", "_max", "_mad", "_kurtosis", "_range"],
       "Disp": ["_sd", "_cv", "_iqr"],
       "TS": ["_stv", "_tbv", "_rcv", "_mr", "_tbcr"]}
ALL_SUF = sorted({s for v in FAM.values() for s in v}, key=len, reverse=True)
EXP10 = ["M", "EM", "Dist", "Disp", "TS"]

# 논문 Supplementary Table 6 (Exp.10 피처셋, CV mean, 임계값 최적화 이전)
PAPER_T6 = {
    "Logistic regression":  dict(acc=0.79, auc=0.86, sens=0.80, spec=0.78, f1=0.72),
    "Elastic-Net":          dict(acc=0.75, auc=0.86, sens=0.74, spec=0.76, f1=0.67),
    "Linear SVM":           dict(acc=0.75, auc=0.85, sens=0.58, spec=0.84, f1=0.61),
    "LDA":                  dict(acc=0.81, auc=0.87, sens=0.76, spec=0.83, f1=0.72),
    "Random Forest":        dict(acc=0.67, auc=0.62, sens=0.04, spec=0.97, f1=0.06),
    "XGBClassifier":        dict(acc=0.61, auc=0.54, sens=0.22, spec=0.80, f1=0.26),
    "RBF SVM":              dict(acc=0.66, auc=0.72, sens=0.10, spec=0.93, f1=0.14),
}


def resolve(cols, fams):
    want = set(itertools.chain.from_iterable(FAM[f] for f in fams))
    return [c for c in cols
            if (m := next((s for s in ALL_SUF if c.endswith(s)), None)) is not None and m in want]


def lr(C=1.0, cw=None):
    return LogisticRegression(penalty="l2", C=C, class_weight=cw, max_iter=5000, random_state=SEED)


def make_models():
    return {
        "Logistic regression": LogisticRegression(penalty="l2", C=1.0, max_iter=5000, random_state=SEED),
        "Elastic-Net": LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                                          C=1.0, max_iter=5000, random_state=SEED),
        "Linear SVM": SVC(kernel="linear", probability=True, random_state=SEED),
        "LDA": LinearDiscriminantAnalysis(),
        "Random Forest": RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=1),
        "XGBClassifier": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.1,
                                       eval_metric="logloss", random_state=SEED, n_jobs=1),
        "RBF SVM": SVC(kernel="rbf", probability=True, random_state=SEED),
    }


def met(y, prob):
    pred = (prob >= THRESH).astype(int)
    return dict(acc=accuracy_score(y, pred),
                auc=roc_auc_score(y, prob),
                sens=recall_score(y, pred, zero_division=0),
                spec=recall_score(1 - y, 1 - pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0))


def rfe_order(X, y, C=1.0, cw=None):
    s = RFE(lr(C, cw), n_features_to_select=1, step=5)
    s.fit(X, y)
    return np.argsort(s.ranking_)


# ─────────────────────────── PART 1 ───────────────────────────
def part1_table6(df, y, cols):
    """논문 Suppl. Table 6 재현: Exp.10 피처셋 + 7개 모델."""
    X = df[resolve(cols, EXP10)].to_numpy()
    Xs = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))
    # 논문 표현대로 'Exp.10 feature set' = RFE로 뽑힌 피처셋 (CV 밖에서 확정)
    order = rfe_order(Xs, y, 1.0, None)
    idx = order[:30]
    Xf = Xs[:, idx]

    print("=" * 108)
    print("PART 1 : 논문 Supplementary Table 6 재현 (Exp.10 피처셋, 7개 모델, 임계값 0.5)")
    print("=" * 108)
    print("{:<22}{:>26}{:>26}{:>30}".format("모델", "논문 (acc/auc/sens)", "재현 (acc/auc/sens)", "재현 spec/f1  (논문)"))
    print("-" * 108)

    out = {}
    for name, model in make_models().items():
        skf = StratifiedKFold(N_OUTER, shuffle=True, random_state=SEED)
        oof = np.zeros(len(y))
        for tr, te in skf.split(Xf, y):
            m = model.__class__(**model.get_params())
            m.fit(Xf[tr], y[tr])
            oof[te] = m.predict_proba(Xf[te])[:, 1]
        r = met(y, oof)
        p = PAPER_T6[name]
        out[name] = {"repro": r, "paper": p}
        print("{:<22}{:>26}{:>26}{:>30}".format(
            name,
            "{:.2f} / {:.2f} / {:.2f}".format(p["acc"], p["auc"], p["sens"]),
            "{:.2f} / {:.2f} / {:.2f}".format(r["acc"], r["auc"], r["sens"]),
            "{:.2f} / {:.2f}  ({:.2f} / {:.2f})".format(r["spec"], r["f1"], p["spec"], p["f1"])))

    lin = ["Logistic regression", "Elastic-Net", "Linear SVM", "LDA"]
    non = ["Random Forest", "XGBClassifier", "RBF SVM"]
    print("-" * 108)
    print("  선형 평균 AUC   : 논문 {:.3f}  재현 {:.3f}".format(
        np.mean([PAPER_T6[m]["auc"] for m in lin]), np.mean([out[m]["repro"]["auc"] for m in lin])))
    print("  비선형 평균 AUC : 논문 {:.3f}  재현 {:.3f}".format(
        np.mean([PAPER_T6[m]["auc"] for m in non]), np.mean([out[m]["repro"]["auc"] for m in non])))
    print("  비선형 평균 민감도(붕괴 재현 여부): 논문 {:.3f}  재현 {:.3f}".format(
        np.mean([PAPER_T6[m]["sens"] for m in non]), np.mean([out[m]["repro"]["sens"] for m in non])))
    pa = np.array([PAPER_T6[m]["auc"] for m in PAPER_T6])
    re = np.array([out[m]["repro"]["auc"] for m in PAPER_T6])
    print("  7개 모델 AUC 상관 r = {:+.3f},  MAE = {:.3f}".format(
        float(np.corrcoef(pa, re)[0, 1]), float(np.mean(np.abs(pa - re)))))
    return out


# ─────────────────────────── PART 2 ───────────────────────────
def ladder_L0(X, y, seed):
    """Full nested: 변수선택 + C + class_weight 전부 inner CV 안."""
    Xn = X.to_numpy()
    outer = StratifiedKFold(N_OUTER, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in outer.split(Xn, y):
        imp = SimpleImputer(strategy="median")
        Xtr, Xte = imp.fit_transform(Xn[tr]), imp.transform(Xn[te])
        sc = StandardScaler(); Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)
        ytr = y[tr]
        folds = list(StratifiedKFold(N_INNER, shuffle=True, random_state=seed).split(Xtr, ytr))
        orders = {(fi, C, cw): rfe_order(Xtr[i_tr], ytr[i_tr], C, cw)
                  for fi, (i_tr, _) in enumerate(folds)
                  for C, cw in itertools.product(C_GRID, CW_GRID)}
        best, bs = None, -1
        for C, cw, k in itertools.product(C_GRID, CW_GRID, K_GRID):
            sc_ = []
            for fi, (i_tr, i_va) in enumerate(folds):
                idx = orders[(fi, C, cw)][:k]
                m = lr(C, cw); m.fit(Xtr[i_tr][:, idx], ytr[i_tr])
                try:
                    sc_.append(roc_auc_score(ytr[i_va], m.predict_proba(Xtr[i_va][:, idx])[:, 1]))
                except ValueError:
                    pass
            s = float(np.mean(sc_)) if sc_ else -1
            if s > bs:
                bs, best = s, (C, cw, k)
        C, cw, k = best
        idx = rfe_order(Xtr, ytr, C, cw)[:k]
        m = lr(C, cw); m.fit(Xtr[:, idx], ytr)
        oof[te] = m.predict_proba(Xte[:, idx])[:, 1]
    return met(y, oof)


def ladder_L1(X, y, seed, C=1.0, cw=None, k=30):
    """RFE만 CV 밖 + 하이퍼파라미터는 기본값 고정 (탐색 없음)."""
    Xn = X.to_numpy()
    Xs = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(Xn))
    idx = rfe_order(Xs, y, C, cw)[:k]
    Xf = Xs[:, idx]
    skf = StratifiedKFold(N_OUTER, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(Xf, y):
        m = lr(C, cw); m.fit(Xf[tr], y[tr])
        oof[te] = m.predict_proba(Xf[te])[:, 1]
    return met(y, oof)


def ladder_L2(X, y, seed):
    """RFE + 하이퍼파라미터 탐색 모두 CV 밖 (같은 자료로 최고를 고름)."""
    Xn = X.to_numpy()
    Xs = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(Xn))
    cache = {(C, cw): rfe_order(Xs, y, C, cw) for C, cw in itertools.product(C_GRID, CW_GRID)}
    skf = StratifiedKFold(N_OUTER, shuffle=True, random_state=seed)
    best, bo = None, -1
    for C, cw, k in itertools.product(C_GRID, CW_GRID, K_GRID):
        idx = cache[(C, cw)][:k]
        oof = np.zeros(len(y))
        for tr, te in skf.split(Xs, y):
            m = lr(C, cw); m.fit(Xs[tr][:, idx], y[tr])
            oof[te] = m.predict_proba(Xs[te][:, idx])[:, 1]
        a = roc_auc_score(y, oof)
        if a > bo:
            bo, best = a, oof
    return met(y, best)


def part2_ladder(df, y, cols):
    X = df[resolve(cols, EXP10)]
    print("\n" + "=" * 108)
    print("PART 2 : 누수 사다리 — 논문 보고값(Exp.10 AUC 0.859)이 어느 단계에 놓이는가")
    print("=" * 108)

    rows = {}
    for label, fn in [("L0 full nested", ladder_L0),
                      ("L1 RFE만 CV밖 (기본값 고정)", ladder_L1),
                      ("L2 RFE+튜닝 모두 CV밖", ladder_L2)]:
        runs = [fn(X, y, SEED + r) for r in range(N_REPEAT)]
        arr = {m: np.array([x[m] for x in runs]) for m in ("acc", "auc", "sens", "spec", "f1")}
        rows[label] = {m: {"mean": float(v.mean()), "sd": float(v.std(ddof=1))} for m, v in arr.items()}
        print("  {:<30} AUC {:.4f} ± {:.4f}   Acc {:.4f}   Sens {:.4f}   Spec {:.4f}   F1 {:.4f}".format(
            label, arr["auc"].mean(), arr["auc"].std(ddof=1),
            arr["acc"].mean(), arr["sens"].mean(), arr["spec"].mean(), arr["f1"].mean()))

    print("  {:<30} AUC {:.4f}   Acc {:.4f}   Sens {:.4f}   Spec {:.4f}   F1 {:.4f}   ← 논문".format(
        "논문 Exp.10 (Table 5 / T6)", 0.859, 0.786, 0.800, 0.780, 0.720))
    return rows


def main():
    df = pd.read_csv(DATA)
    y = df["mci_label"].astype(int).values
    cols = [c for c in df.columns if c not in META]
    print("n={} (CN={}, MCI={}), Exp.10 후보 피처={}개\n".format(
        len(y), int((y == 0).sum()), int((y == 1).sum()), len(resolve(cols, EXP10))))

    t6 = part1_table6(df, y, cols)
    ld = part2_ladder(df, y, cols)

    (HERE / "rq2_validate_ladder.json").write_text(
        json.dumps({"table6": t6, "ladder": ld}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print("\nSaved to {}".format(HERE / "rq2_validate_ladder.json"))


if __name__ == "__main__":
    main()
