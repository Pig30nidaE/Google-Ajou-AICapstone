"""
YMJ(2026) 충실 재현 [2/2] : 검증된 피처셋으로 Table 1의 12개 실험 수행

Table 3 재현으로 일별 파생지표의 정확성이 확인된 피처셋(339개)을 사용.
논문 명시 사항 준수: Logistic regression, RFE, 10-fold CV, z-score 표준화.

각 실험을 두 방식으로 평가:
  NON-NESTED : 전체 라벨을 보고 RFE 1회 -> 고정 피처셋으로 10-fold CV
  NESTED     : outer-fold의 train 안에서만 impute/scale/RFE 수행
"""
from __future__ import annotations

import itertools
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_OUTER = 10
N_INNER = 5
HERE = Path(__file__).parent
DATA_PATH = HERE / "ymj_faithful_features.csv"
META = {"EMAIL", "n_nights", "DIAG_NM", "original_label", "mci_label"}

FAMILY_SUFFIXES = {
    "M":    ["_mean"],
    "EM":   ["_median", "_trimmed_mean", "_mode"],
    "Dist": ["_min", "_max", "_mad", "_kurtosis", "_range"],
    "Disp": ["_sd", "_cv", "_iqr"],
    "TS":   ["_stv", "_tbv", "_rcv", "_mr", "_tbcr"],
}
ALL_SUF = sorted({s for v in FAMILY_SUFFIXES.values() for s in v}, key=len, reverse=True)

EXPERIMENTS = [
    ("Exp.1",  ["M"],                             0.525),
    ("Exp.2",  ["EM"],                            0.667),
    ("Exp.3",  ["Dist"],                          0.701),
    ("Exp.4",  ["Disp"],                          0.605),
    ("Exp.5",  ["TS"],                            0.742),
    ("Exp.6",  ["EM", "Dist"],                    0.671),
    ("Exp.7",  ["EM", "Disp"],                    0.610),
    ("Exp.8",  ["EM", "TS"],                      0.773),
    ("Exp.9",  ["M", "EM", "Dist", "Disp"],       0.810),
    ("Exp.10", ["M", "EM", "Dist", "Disp", "TS"], 0.859),
    ("Exp.11", ["EM", "Dist", "TS"],              0.861),
    ("Exp.12", ["EM", "Dist", "Disp", "TS"],      0.831),
]
PAPER = np.array([e[2] for e in EXPERIMENTS])


def resolve(all_cols, families):
    want = set(itertools.chain.from_iterable(FAMILY_SUFFIXES[f] for f in families))
    out = []
    for c in all_cols:
        m = next((s for s in ALL_SUF if c.endswith(s)), None)
        if m is not None and m in want:
            out.append(c)
    return out


def est(C, cw):
    return LogisticRegression(penalty="l2", C=C, class_weight=cw,
                              max_iter=5000, random_state=RANDOM_STATE)


def selector(sel, n_feat, C, cw):
    e = est(C, cw)
    if sel == "rfecv":
        inner = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=RANDOM_STATE)
        return RFECV(estimator=e, step=5 if n_feat > 40 else 1, cv=inner,
                     scoring="roc_auc", min_features_to_select=min(5, n_feat), n_jobs=1)
    return RFE(estimator=e, n_features_to_select=min(int(sel), n_feat),
               step=5 if n_feat > 60 else 1)


def evaluate(X, y, *, nested, sel, C, cw):
    Xn = X.to_numpy()
    skf = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE)

    if not nested:
        Xi = SimpleImputer(strategy="median").fit_transform(Xn)
        Xs = StandardScaler().fit_transform(Xi)
        s = selector(sel, Xs.shape[1], C, cw)
        s.fit(Xs, y)
        Xf = s.transform(Xs)
        nsel = [int(Xf.shape[1])]

    oof = np.zeros(len(y)); fa = []; nsf = []
    for tr, te in skf.split(Xn, y):
        if nested:
            im = SimpleImputer(strategy="median")
            Xtr, Xte = im.fit_transform(Xn[tr]), im.transform(Xn[te])
            sc = StandardScaler()
            Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)
            s = selector(sel, Xtr.shape[1], C, cw)
            s.fit(Xtr, y[tr])
            Xtr, Xte = s.transform(Xtr), s.transform(Xte)
            nsf.append(int(Xtr.shape[1]))
        else:
            Xtr, Xte = Xf[tr], Xf[te]
        m = est(C, cw); m.fit(Xtr, y[tr])
        p = m.predict_proba(Xte)[:, 1]
        oof[te] = p
        try:
            fa.append(roc_auc_score(y[te], p))
        except ValueError:
            fa.append(np.nan)

    pr = (oof >= 0.5).astype(int)
    return {"auc_pooled": float(roc_auc_score(y, oof)),
            "auc_cvmean": float(np.nanmean(fa)),
            "acc": float(accuracy_score(y, pr)),
            "sens": float(recall_score(y, pr, zero_division=0)),
            "spec": float(recall_score(1 - y, 1 - pr, zero_division=0)),
            "n_selected": nsf if nested else nsel}


def main():
    df = pd.read_csv(DATA_PATH)
    y = df["mci_label"].astype(int).values
    cols = [c for c in df.columns if c not in META]
    expcols = {n: resolve(cols, f) for n, f, _ in EXPERIMENTS}

    print(f"n={len(df)} (CN={int((y==0).sum())}, MCI={int((y==1).sum())}), 후보 피처={len(cols)}")
    for n, f, _ in EXPERIMENTS[:5]:
        print(f"  {n} ({'+'.join(f)}): {len(expcols[n])}개")

    # ---------------- PHASE 1 : 논문 수치 재현 설정 탐색 (non-nested) ----------
    print("\n" + "=" * 96)
    print("PHASE 1: 논문 12개 AUC 재현 설정 탐색 (non-nested)")
    print("=" * 96)
    grid = list(itertools.product([0.01, 0.1, 1.0, 10.0], ["balanced", None],
                                  [10, 20, 30, 50, "rfecv"]))
    recs = []
    t0 = time.time()
    for i, (C, cw, sel) in enumerate(grid, 1):
        vp, vc = [], []
        for n, _, _ in EXPERIMENTS:
            r = evaluate(df[expcols[n]], y, nested=False, sel=sel, C=C, cw=cw)
            vp.append(r["auc_pooled"]); vc.append(r["auc_cvmean"])
        for agg, v in (("pooled", vp), ("cvmean", vc)):
            v = np.array(v)
            recs.append({"C": C, "class_weight": cw, "selection": sel, "aggregation": agg,
                         "auc_vec": [float(x) for x in v],
                         "rmse": float(np.sqrt(np.mean((v - PAPER) ** 2))),
                         "mae": float(np.mean(np.abs(v - PAPER))),
                         "r": float(np.corrcoef(v, PAPER)[0, 1])})
        if i % 5 == 0:
            print(f"  [{i:2d}/{len(grid)}] best RMSE so far = {min(r_['rmse'] for r_ in recs):.4f}")
    print(f"Phase 1 완료 ({time.time()-t0:.0f}s)")

    recs.sort(key=lambda d: d["rmse"])
    print("\n상위 6개 설정")
    print(f"{'rk':<4}{'C':<7}{'cw':<11}{'sel':<8}{'agg':<9}{'RMSE':<9}{'MAE':<9}{'r':<8}")
    for i, r_ in enumerate(recs[:6], 1):
        print(f"{i:<4}{r_['C']:<7}{str(r_['class_weight']):<11}{str(r_['selection']):<8}"
              f"{r_['aggregation']:<9}{r_['rmse']:<9.4f}{r_['mae']:<9.4f}{r_['r']:<+8.3f}")

    best = recs[0]
    print(f"\n[재현 레시피] C={best['C']}, class_weight={best['class_weight']}, "
          f"RFE={best['selection']}, agg={best['aggregation']}, non-nested")
    print(f"  논문과 RMSE={best['rmse']:.4f}  MAE={best['mae']:.4f}  r={best['r']:+.3f}")

    # ---------------- PHASE 2 : 같은 레시피로 nested ---------------------------
    print("\n" + "=" * 96)
    print("PHASE 2: 동일 레시피, RFE만 outer-fold 내부로 (nested)")
    print("=" * 96)
    key = "auc_pooled" if best["aggregation"] == "pooled" else "auc_cvmean"
    print(f"\n{'Exp':<8}{'families':<26}{'k':>5} |{'paper':>8}{'재현':>9}{'NESTED':>9} |{'gap':>8}")
    print("-" * 82)

    out = {}
    nn, ne = [], []
    for i, (n, f, pa) in enumerate(EXPERIMENTS):
        a_nn = best["auc_vec"][i]
        r = evaluate(df[expcols[n]], y, nested=True, sel=best["selection"],
                     C=best["C"], cw=best["class_weight"])
        a_ne = r[key]
        nn.append(a_nn); ne.append(a_ne)
        out[n] = {"paper": pa, "non_nested": a_nn, "nested": r, "gap": float(a_nn - a_ne),
                  "n_candidates": len(expcols[n])}
        print(f"{n:<8}{'+'.join(f):<26}{len(expcols[n]):>5} |{pa:>8.3f}{a_nn:>9.3f}{a_ne:>9.3f} |{a_nn-a_ne:>+8.3f}")

    nn, ne = np.array(nn), np.array(ne)
    print("-" * 82)
    print(f"\n[요약]")
    print(f"  논문 평균           : {PAPER.mean():.3f}")
    print(f"  재현(non-nested)평균: {nn.mean():.3f}  (논문과 MAE={np.mean(np.abs(nn-PAPER)):.3f}, r={np.corrcoef(nn,PAPER)[0,1]:+.3f})")
    print(f"  NESTED 평균         : {ne.mean():.3f}  (논문과 MAE={np.mean(np.abs(ne-PAPER)):.3f}, r={np.corrcoef(ne,PAPER)[0,1]:+.3f})")
    print(f"  평균 낙관 편향      : {(nn-ne).mean():+.3f} AUC")
    bi = int(np.argmax(ne))
    print(f"  최고 nested AUC     : {ne.max():.3f} ({EXPERIMENTS[bi][0]}, 논문 보고 {PAPER[bi]:.3f})")

    kk = np.array([out[n]["n_candidates"] for n, _, _ in EXPERIMENTS])
    gaps = nn - ne
    from scipy import stats as st
    rho, pv = st.spearmanr(kk, gaps)
    print(f"\n  후보피처수 vs 낙관편향: Spearman rho={rho:+.3f} (p={pv:.4f})")

    (HERE / "ymj_faithful_results.json").write_text(json.dumps(
        {"best_recipe": {k: best[k] for k in ("C", "class_weight", "selection", "aggregation", "rmse", "mae", "r")},
         "phase1_top": recs[:10], "phase2": out,
         "summary": {"paper_mean": float(PAPER.mean()), "non_nested_mean": float(nn.mean()),
                     "nested_mean": float(ne.mean()), "mean_optimism": float(gaps.mean()),
                     "best_nested": float(ne.max())}},
        indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nSaved to {HERE/'ymj_faithful_results.json'}")


if __name__ == "__main__":
    main()
