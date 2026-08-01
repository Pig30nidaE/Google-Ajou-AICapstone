"""
YMJ(2026) 재현 [검토] : 놓쳤을 수 있는 명세들을 변형(variant)으로 테스트

검토 대상:
  V0  baseline                : 원 단위로 피처 생성 -> 모델링 직전 z-score  (지금까지의 방식)
  V1  z-score-first           : 논문 서술 순서대로 '일별 데이터를 먼저 z-score' 후 피처 생성
                                (논문이 '무한값 보정'을 언급한 것과 정합 - CV의 분모가 0에 근접)
  V2  SRI 결측 채움 + 완전관측 : 'SRI_filled'(Fig.2) 단서 반영, 결측 행 제거
  V3  V1 + V2

각 변형마다
  (a) 논문 12개 AUC와 가장 잘 맞는 non-nested 레시피를 찾고
  (b) 동일 레시피에서 RFE만 fold 내부로 옮겼을 때(nested) 성능을 본다.

핵심 질문: "내가 놓친 명세 중에, nested 성능을 살려내는 것이 있는가?"
"""
from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import ymj_faithful_build as B

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_OUTER, N_INNER = 10, 5
HERE = Path(__file__).parent

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

FAMILY_SUFFIXES = B.__dict__.get("FAMILY_SUFFIXES", {
    "M": ["_mean"], "EM": ["_median", "_trimmed_mean", "_mode"],
    "Dist": ["_min", "_max", "_mad", "_kurtosis", "_range"],
    "Disp": ["_sd", "_cv", "_iqr"],
    "TS": ["_stv", "_tbv", "_rcv", "_mr", "_tbcr"]})
ALL_SUF = sorted({s for v in FAMILY_SUFFIXES.values() for s in v}, key=len, reverse=True)


def resolve(cols, fams):
    want = set(itertools.chain.from_iterable(FAMILY_SUFFIXES[f] for f in fams))
    return [c for c in cols
            if (m := next((s for s in ALL_SUF if c.endswith(s)), None)) is not None and m in want]


def build_variant(daily: pd.DataFrame, *, zscore_first: bool, fill_sri: bool) -> pd.DataFrame:
    d = daily.copy()
    metrics = B.DAILY_METRICS

    if fill_sri:
        d["SRI"] = d.groupby("EMAIL")["SRI"].transform(lambda s: s.ffill().bfill())
        d["SRI"] = d["SRI"].fillna(d["SRI"].median())

    if zscore_first:
        # 논문 서술 순서: 전처리 단계에서 수치형 변수 z-score -> 이후 피처 엔지니어링
        for m in metrics:
            mu, sd = d[m].mean(), d[m].std()
            d[m] = (d[m] - mu) / (sd if sd and sd > 0 else 1.0)

    rows = []
    for email, grp in d.groupby("EMAIL"):
        grp = grp.sort_values("date")
        f = {"EMAIL": email, "n_nights": len(grp)}
        for m in metrics:
            f.update(B.summarize(grp[m].to_numpy(dtype=float), m))
        rows.append(f)
    pat = pd.DataFrame(rows)
    pat = pat.replace([np.inf, -np.inf], np.nan)   # 논문: '무한값 보정'
    return pat


def est(C, cw):
    return LogisticRegression(penalty="l2", C=C, class_weight=cw,
                              max_iter=5000, random_state=RANDOM_STATE)


def selector(sel, n, C, cw):
    e = est(C, cw)
    if sel == "rfecv":
        inner = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=RANDOM_STATE)
        return RFECV(estimator=e, step=5 if n > 40 else 1, cv=inner, scoring="roc_auc",
                     min_features_to_select=min(5, n), n_jobs=1)
    return RFE(estimator=e, n_features_to_select=min(int(sel), n), step=5 if n > 60 else 1)


def evaluate(X, y, *, nested, sel, C, cw):
    Xn = X.to_numpy()
    skf = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE)
    if not nested:
        Xi = SimpleImputer(strategy="median").fit_transform(Xn)
        Xs = StandardScaler().fit_transform(Xi)
        s = selector(sel, Xs.shape[1], C, cw); s.fit(Xs, y); Xf = s.transform(Xs)
    oof = np.zeros(len(y)); fa = []
    for tr, te in skf.split(Xn, y):
        if nested:
            im = SimpleImputer(strategy="median")
            Xtr, Xte = im.fit_transform(Xn[tr]), im.transform(Xn[te])
            sc = StandardScaler(); Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)
            s = selector(sel, Xtr.shape[1], C, cw); s.fit(Xtr, y[tr])
            Xtr, Xte = s.transform(Xtr), s.transform(Xte)
        else:
            Xtr, Xte = Xf[tr], Xf[te]
        m = est(C, cw); m.fit(Xtr, y[tr])
        p = m.predict_proba(Xte)[:, 1]; oof[te] = p
        try: fa.append(roc_auc_score(y[te], p))
        except ValueError: fa.append(np.nan)
    return {"pooled": float(roc_auc_score(y, oof)), "cvmean": float(np.nanmean(fa)), "oof": oof}


def run_variant(name, pat, labels):
    m = pat.merge(labels, on="EMAIL", how="inner")
    m = m[m["n_nights"] >= B.MIN_NIGHTS]
    m = m[m["original_label"].isin([0, 1])].reset_index(drop=True)
    m["mci_label"] = m["original_label"].astype(int)
    y = m["mci_label"].values
    meta = {"EMAIL", "n_nights", "DIAG_NM", "original_label", "mci_label"}
    nan_rate = m[[c for c in m.columns if c not in meta]].isna().mean()
    drop = nan_rate[nan_rate > 0.2].index.tolist()
    if drop:
        m = m.drop(columns=drop)
    cols = [c for c in m.columns if c not in meta]

    print(f"\n{'='*94}\n[{name}]  n={len(m)} (CN={int((y==0).sum())}, MCI={int((y==1).sum())}), "
          f"피처={len(cols)}개, 결측>20%제외={len(drop)}개\n{'='*94}")

    grid = list(itertools.product([0.01, 0.1, 1.0], ["balanced", None], [10, 20, 30, "rfecv"]))
    best = None
    for C, cw, sel in grid:
        vp, vc = [], []
        for n_, f_, _ in EXPERIMENTS:
            r = evaluate(m[resolve(cols, f_)], y, nested=False, sel=sel, C=C, cw=cw)
            vp.append(r["pooled"]); vc.append(r["cvmean"])
        for agg, v in (("pooled", vp), ("cvmean", vc)):
            v = np.array(v)
            rmse = float(np.sqrt(np.mean((v - PAPER) ** 2)))
            if best is None or rmse < best["rmse"]:
                best = {"C": C, "cw": cw, "sel": sel, "agg": agg, "rmse": rmse,
                        "mae": float(np.mean(np.abs(v - PAPER))),
                        "r": float(np.corrcoef(v, PAPER)[0, 1]), "vec": v}

    print(f"  최적 non-nested 레시피: C={best['C']}, cw={best['cw']}, RFE={best['sel']}, "
          f"agg={best['agg']}")
    print(f"    -> 논문과 RMSE={best['rmse']:.4f}, MAE={best['mae']:.4f}, r={best['r']:+.3f}, "
          f"평균AUC={best['vec'].mean():.3f} (논문 {PAPER.mean():.3f})")

    key = best["agg"]
    ne = []
    for n_, f_, _ in EXPERIMENTS:
        r = evaluate(m[resolve(cols, f_)], y, nested=True, sel=best["sel"],
                     C=best["C"], cw=best["cw"])
        ne.append(r[key])
    ne = np.array(ne)
    print(f"    -> 동일 레시피 NESTED: 평균AUC={ne.mean():.3f}, 최고={ne.max():.3f} "
          f"({EXPERIMENTS[int(np.argmax(ne))][0]}), 평균편향={(best['vec']-ne).mean():+.3f}")
    return {"variant": name, "n": len(m), "n_features": len(cols),
            "recipe": {k: str(best[k]) for k in ("C", "cw", "sel", "agg")},
            "rmse": best["rmse"], "mae": best["mae"], "r": best["r"],
            "non_nested_vec": [float(x) for x in best["vec"]],
            "nested_vec": [float(x) for x in ne],
            "non_nested_mean": float(best["vec"].mean()), "nested_mean": float(ne.mean()),
            "nested_best": float(ne.max()), "optimism": float((best["vec"] - ne).mean())}


def main():
    daily = pd.read_csv(HERE / "ymj_faithful_daily.csv")
    labels = pd.concat([B.preprocess_label(B.read_csv_flexible(B.TRAIN_LABEL)),
                        B.preprocess_label(B.read_csv_flexible(B.VAL_LABEL))],
                       ignore_index=True).drop_duplicates("EMAIL")

    results = []
    for name, zs, fs in [("V0 baseline", False, False),
                         ("V1 z-score-first", True, False),
                         ("V2 SRI채움", False, True),
                         ("V3 z-first+SRI채움", True, True)]:
        pat = build_variant(daily, zscore_first=zs, fill_sri=fs)
        results.append(run_variant(name, pat, labels))

    print("\n" + "=" * 94)
    print("변형별 종합 비교")
    print("=" * 94)
    print(f"{'variant':<22}{'논문일치RMSE':>13}{'r':>8}{'nonNest평균':>12}{'NESTED평균':>12}{'NESTED최고':>12}")
    print("-" * 94)
    for r in results:
        print(f"{r['variant']:<22}{r['rmse']:>13.4f}{r['r']:>+8.3f}"
              f"{r['non_nested_mean']:>12.3f}{r['nested_mean']:>12.3f}{r['nested_best']:>12.3f}")
    print("-" * 94)
    print(f"{'(논문 보고값)':<22}{'-':>13}{'-':>8}{PAPER.mean():>12.3f}{'-':>12}{PAPER.max():>12.3f}")

    (HERE / "ymj_variants_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {HERE/'ymj_variants_results.json'}")


if __name__ == "__main__":
    main()
