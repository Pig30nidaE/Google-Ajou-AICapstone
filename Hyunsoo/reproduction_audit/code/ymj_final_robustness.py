"""
YMJ(2026) 재현 [최종 검토] : 논문 Supplementary Table 3/4의 민감도 축까지 전부 커버

  (A) 변동성 윈도우 감도   : 5, 7, 14, 21일  (논문 Suppl. Table 4)
  (B) 최소 야간수 감도     : >=35, >=40, >=45 (논문 Suppl. Table 3)
  (C) CV 시드 안정성       : 10개 시드      (논문 미보고 - 내가 추가하는 검증)

기준 피처셋은 논문 일치도가 가장 높았던 V3(z-score-first + SRI 채움)를 사용.
각 축에서 non-nested / nested 를 함께 보고, 'nested를 살려내는 설정이 있는가'를 확인한다.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import ymj_faithful_build as B
from ymj_missed_variants import EXPERIMENTS, PAPER, resolve

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
N_OUTER, N_INNER = 10, 5
RECIPE = dict(C=0.1, cw=None, sel=30)      # V3 최적 non-nested 레시피
HEAD = ["Exp.10", "Exp.11", "Exp.12"]      # 논문 헤드라인 실험


def build_features(daily, window):
    """V3 방식(z-score-first + SRI 채움) + 지정 윈도우로 환자단위 피처 생성."""
    d = daily.copy()
    d["SRI"] = d.groupby("EMAIL")["SRI"].transform(lambda s: s.ffill().bfill())
    d["SRI"] = d["SRI"].fillna(d["SRI"].median())
    for m in B.DAILY_METRICS:
        mu, sd = d[m].mean(), d[m].std()
        d[m] = (d[m] - mu) / (sd if sd and sd > 0 else 1.0)

    rows = []
    for email, grp in d.groupby("EMAIL"):
        grp = grp.sort_values("date")
        f = {"EMAIL": email, "n_nights": len(grp)}
        for m in B.DAILY_METRICS:
            # 윈도우를 명시적으로 전달 (기본인자 고정 문제 회피)
            f.update(B.summarize(grp[m].to_numpy(dtype=float), m, w=window))
        rows.append(f)
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def prep(pat, labels, min_nights):
    m = pat.merge(labels, on="EMAIL", how="inner")
    m = m[m["n_nights"] >= min_nights]
    m = m[m["original_label"].isin([0, 1])].reset_index(drop=True)
    m["mci_label"] = m["original_label"].astype(int)
    meta = {"EMAIL", "n_nights", "DIAG_NM", "original_label", "mci_label"}
    cols = [c for c in m.columns if c not in meta]
    nr = m[cols].isna().mean()
    drop = nr[nr > 0.2].index.tolist()
    if drop:
        m = m.drop(columns=drop)
        cols = [c for c in cols if c not in drop]
    return m, m["mci_label"].values, cols


def est():
    return LogisticRegression(penalty="l2", C=RECIPE["C"], class_weight=RECIPE["cw"],
                              max_iter=5000, random_state=42)


def evaluate(X, y, *, nested, seed=42):
    Xn = X.to_numpy()
    skf = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=seed)
    k = min(RECIPE["sel"], Xn.shape[1])
    if not nested:
        Xi = SimpleImputer(strategy="median").fit_transform(Xn)
        Xs = StandardScaler().fit_transform(Xi)
        s = RFE(est(), n_features_to_select=k, step=5 if Xs.shape[1] > 60 else 1)
        s.fit(Xs, y); Xf = s.transform(Xs)
    oof = np.zeros(len(y))
    for tr, te in skf.split(Xn, y):
        if nested:
            im = SimpleImputer(strategy="median")
            Xtr, Xte = im.fit_transform(Xn[tr]), im.transform(Xn[te])
            sc = StandardScaler(); Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)
            s = RFE(est(), n_features_to_select=k, step=5 if Xtr.shape[1] > 60 else 1)
            s.fit(Xtr, y[tr]); Xtr, Xte = s.transform(Xtr), s.transform(Xte)
        else:
            Xtr, Xte = Xf[tr], Xf[te]
        mm = est(); mm.fit(Xtr, y[tr])
        oof[te] = mm.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(y, oof))


def main():
    daily = pd.read_csv(HERE / "ymj_faithful_daily.csv")
    labels = pd.concat([B.preprocess_label(B.read_csv_flexible(B.TRAIN_LABEL)),
                        B.preprocess_label(B.read_csv_flexible(B.VAL_LABEL))],
                       ignore_index=True).drop_duplicates("EMAIL")
    out = {}

    # ---------- (A) 윈도우 감도 -------------------------------------------------
    print("=" * 88)
    print("(A) 변동성 윈도우 감도  [논문 Suppl. Table 4: Exp.10 최적 window=5, AUC=0.867]")
    print("=" * 88)
    print(f"{'window':<9}{'Exp':<9}{'논문':>8}{'nonNested':>12}{'NESTED':>10}{'gap':>9}")
    print("-" * 88)
    win_res = {}
    for w in [5, 7, 14, 21]:
        pat = build_features(daily, w)
        m, y, cols = prep(pat, labels, 35)
        for name in HEAD:
            fams = next(f for n_, f, _ in EXPERIMENTS if n_ == name)
            pa = next(p for n_, _, p in EXPERIMENTS if n_ == name)
            X = m[resolve(cols, fams)]
            nn = evaluate(X, y, nested=False)
            ne = evaluate(X, y, nested=True)
            win_res[f"w{w}_{name}"] = {"non_nested": nn, "nested": ne}
            print(f"{w:<9}{name:<9}{pa:>8.3f}{nn:>12.3f}{ne:>10.3f}{nn-ne:>+9.3f}")
    out["window"] = win_res

    # ---------- (B) 최소 야간수 감도 -------------------------------------------
    print("\n" + "=" * 88)
    print("(B) 최소 야간수 감도  [논문 Suppl. Table 3]")
    print("=" * 88)
    print(f"{'minNights':<11}{'n':>5}{'Exp':<9}{'논문(≥35)':>11}{'nonNested':>12}{'NESTED':>10}")
    print("-" * 88)
    pat7 = build_features(daily, 7)
    mn_res = {}
    for mn in [35, 40, 45]:
        m, y, cols = prep(pat7, labels, mn)
        for name in HEAD:
            fams = next(f for n_, f, _ in EXPERIMENTS if n_ == name)
            pa = next(p for n_, _, p in EXPERIMENTS if n_ == name)
            X = m[resolve(cols, fams)]
            nn = evaluate(X, y, nested=False)
            ne = evaluate(X, y, nested=True)
            mn_res[f"n{mn}_{name}"] = {"n": int(len(m)), "non_nested": nn, "nested": ne}
            print(f"{mn:<11}{len(m):>5}{name:<9}{pa:>11.3f}{nn:>12.3f}{ne:>10.3f}")
    out["min_nights"] = mn_res

    # ---------- (C) 시드 안정성 -------------------------------------------------
    print("\n" + "=" * 88)
    print("(C) CV 시드 안정성 (10개 시드, window=7, >=35박)")
    print("=" * 88)
    m, y, cols = prep(pat7, labels, 35)
    seeds = [0, 1, 7, 13, 21, 42, 77, 100, 202, 777]
    seed_res = {}
    print(f"{'Exp':<9}{'논문':>8}  {'nonNested mean±sd':>22}   {'NESTED mean±sd':>22}{'NESTED range':>18}")
    print("-" * 88)
    for name in HEAD:
        fams = next(f for n_, f, _ in EXPERIMENTS if n_ == name)
        pa = next(p for n_, _, p in EXPERIMENTS if n_ == name)
        X = m[resolve(cols, fams)]
        nns = [evaluate(X, y, nested=False, seed=s) for s in seeds]
        nes = [evaluate(X, y, nested=True, seed=s) for s in seeds]
        seed_res[name] = {"non_nested": nns, "nested": nes}
        print(f"{name:<9}{pa:>8.3f}  {np.mean(nns):>10.3f} ± {np.std(nns):<9.3f}   "
              f"{np.mean(nes):>10.3f} ± {np.std(nes):<9.3f}"
              f"{f'[{min(nes):.3f}, {max(nes):.3f}]':>18}")
    out["seeds"] = seed_res

    print("\n" + "=" * 88)
    print("[최종 판정]")
    all_nested = [v["nested"] for v in win_res.values()] + \
                 [v["nested"] for v in mn_res.values()] + \
                 [x for v in seed_res.values() for x in v["nested"]]
    print(f"  검토한 모든 설정에서의 nested AUC: 평균 {np.mean(all_nested):.3f}, "
          f"최댓값 {np.max(all_nested):.3f}  (총 {len(all_nested)}회 평가)")
    print(f"  논문 헤드라인 AUC 0.859~0.861 에 도달한 설정: "
          f"{sum(1 for a in all_nested if a >= 0.85)}개")

    (HERE / "ymj_final_robustness.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {HERE/'ymj_final_robustness.json'}")


if __name__ == "__main__":
    main()
