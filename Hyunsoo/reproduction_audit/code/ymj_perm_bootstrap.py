"""
YMJ(2026) 재현 [최종] : Permutation test + Bootstrap CI

두 파이프라인 각각에 대해 수행:
  NON-NESTED : RFE를 전체 라벨 보고 1회 -> 고정 피처셋으로 10-fold CV  (논문 추정 방식)
  NESTED     : outer-fold train 안에서만 impute/scale/RFE                (누수 없음)

(1) Permutation test
    라벨을 무작위로 섞고 '동일 파이프라인 전체'를 재실행해 귀무분포를 만든다.
    핵심 예측:
      - nested     의 귀무분포는 0.5 근처에 놓인다 (정상)
      - non-nested 의 귀무분포는 0.5보다 위로 부풀려진다
        (라벨이 무작위여도 selection이 우연한 피처를 찾아내고, 같은 CV로 평가되므로)
      => 부풀려진 정도 자체가 selection bias의 크기다.

(2) Bootstrap CI
    참가자를 복원추출하여 OOF 예측으로부터 AUC의 백분위 신뢰구간을 구한다.
    (논문 limitation의 "wide CIs / sensitivity to resampling" 확인)
"""
from __future__ import annotations

import json
import time
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
from ymj_missed_variants import EXPERIMENTS, resolve

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
N_OUTER = 10
WINDOW = 7
MIN_NIGHTS = 35
RECIPE = dict(C=0.1, cw=None, k=30)
TARGETS = ["Exp.10", "Exp.11", "Exp.12"]

N_PERM = 1000
N_BOOT = 2000


def build_v3(daily, window=WINDOW):
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
            f.update(B.summarize(grp[m].to_numpy(dtype=float), m, w=window))
        rows.append(f)
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def est():
    return LogisticRegression(penalty="l2", C=RECIPE["C"], class_weight=RECIPE["cw"],
                              max_iter=5000, random_state=42)


def run_pipeline(Xn, y, *, nested, seed=42):
    """OOF 확률을 반환."""
    skf = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=seed)
    k = min(RECIPE["k"], Xn.shape[1])
    step = 5 if Xn.shape[1] > 60 else 1

    if not nested:
        Xi = SimpleImputer(strategy="median").fit_transform(Xn)
        Xs = StandardScaler().fit_transform(Xi)
        s = RFE(est(), n_features_to_select=k, step=step)
        s.fit(Xs, y)
        Xf = s.transform(Xs)

    oof = np.zeros(len(y))
    for tr, te in skf.split(Xn, y):
        if nested:
            im = SimpleImputer(strategy="median")
            Xtr, Xte = im.fit_transform(Xn[tr]), im.transform(Xn[te])
            sc = StandardScaler()
            Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)
            s = RFE(est(), n_features_to_select=k, step=step)
            s.fit(Xtr, y[tr])
            Xtr, Xte = s.transform(Xtr), s.transform(Xte)
        else:
            Xtr, Xte = Xf[tr], Xf[te]
        m = est()
        m.fit(Xtr, y[tr])
        oof[te] = m.predict_proba(Xte)[:, 1]
    return oof


def bootstrap_ci(y, probs, n_boot=N_BOOT, seed=7):
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        aucs.append(roc_auc_score(yy, probs[idx]))
    a = np.asarray(aucs)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), float(a.mean()), float(a.std())


def main():
    daily = pd.read_csv(HERE / "ymj_faithful_daily.csv")
    labels = pd.concat([B.preprocess_label(B.read_csv_flexible(B.TRAIN_LABEL)),
                        B.preprocess_label(B.read_csv_flexible(B.VAL_LABEL))],
                       ignore_index=True).drop_duplicates("EMAIL")
    pat = build_v3(daily)
    m = pat.merge(labels, on="EMAIL", how="inner")
    m = m[m["n_nights"] >= MIN_NIGHTS]
    m = m[m["original_label"].isin([0, 1])].reset_index(drop=True)
    y = m["original_label"].astype(int).values
    meta = {"EMAIL", "n_nights", "DIAG_NM", "original_label"}
    cols = [c for c in m.columns if c not in meta]
    nr = m[cols].isna().mean()
    cols = [c for c in cols if nr[c] <= 0.2]

    print(f"n={len(m)} (CN={int((y==0).sum())}, MCI={int((y==1).sum())}), 피처={len(cols)}")
    print(f"레시피: C={RECIPE['C']}, cw={RECIPE['cw']}, RFE k={RECIPE['k']}, "
          f"window={WINDOW}, >={MIN_NIGHTS}박\n")

    Xs = {t: m[resolve(cols, next(f for n_, f, _ in EXPERIMENTS if n_ == t))].to_numpy()
          for t in TARGETS}
    paper = {t: next(p for n_, _, p in EXPERIMENTS if n_ == t) for t in TARGETS}

    # ---------------- 관측값 + Bootstrap CI --------------------------------
    print("=" * 96)
    print("(1) 관측 성능 및 Bootstrap 95% CI  (참가자 복원추출 2000회)")
    print("=" * 96)
    print(f"{'Exp':<9}{'mode':<12}{'AUC':>8}{'95% CI':>20}{'boot mean±sd':>20}   {'논문':>7}")
    print("-" * 96)
    observed = {}
    for t in TARGETS:
        observed[t] = {}
        for mode in ["non_nested", "nested"]:
            oof = run_pipeline(Xs[t], y, nested=(mode == "nested"))
            auc = float(roc_auc_score(y, oof))
            lo, hi, bm, bs = bootstrap_ci(y, oof)
            observed[t][mode] = {"auc": auc, "oof": oof.tolist(),
                                 "ci_lo": lo, "ci_hi": hi, "boot_mean": bm, "boot_sd": bs}
            print(f"{t:<9}{mode:<12}{auc:>8.3f}{f'[{lo:.3f}, {hi:.3f}]':>20}"
                  f"{f'{bm:.3f} ± {bs:.3f}':>20}   {paper[t]:>7.3f}")

    # ---------------- Permutation test ------------------------------------
    print("\n" + "=" * 96)
    print(f"(2) Permutation test  (N={N_PERM}, 라벨 셔플 후 동일 파이프라인 전체 재실행)")
    print("=" * 96)

    t0 = time.time()
    probe = time.time()
    run_pipeline(Xs["Exp.12"], y, nested=True)
    per_nested = time.time() - probe
    print(f"  nested 1회 ≈ {per_nested:.2f}s -> 예상 소요 ≈ "
          f"{per_nested*N_PERM*len(TARGETS)/60:.0f}분 (nested만)\n")

    rng = np.random.default_rng(20260731)
    perm = {t: {"non_nested": [], "nested": []} for t in TARGETS}

    for i in range(N_PERM):
        y_perm = rng.permutation(y)
        for t in TARGETS:
            for mode in ["non_nested", "nested"]:
                oof = run_pipeline(Xs[t], y_perm, nested=(mode == "nested"))
                perm[t][mode].append(float(roc_auc_score(y_perm, oof)))

        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  [{i+1:4d}/{N_PERM}] {el/60:5.1f}분 경과, ETA {el/(i+1)*(N_PERM-i-1)/60:5.1f}분 "
                  f"| Exp.12 null: nonNest={np.mean(perm['Exp.12']['non_nested']):.3f} "
                  f"nested={np.mean(perm['Exp.12']['nested']):.3f}")
            (HERE / "ymj_perm_partial.json").write_text(
                json.dumps({"n_done": i + 1, "perm": perm}, indent=2), encoding="utf-8")

    # ---------------- 결과 요약 --------------------------------------------
    print("\n" + "=" * 96)
    print("Permutation test 결과")
    print("=" * 96)
    print(f"{'Exp':<9}{'mode':<12}{'관측AUC':>9}{'귀무분포 mean±sd':>22}{'95%ile':>9}{'p-value':>10}")
    print("-" * 96)
    results = {}
    for t in TARGETS:
        results[t] = {}
        for mode in ["non_nested", "nested"]:
            null = np.asarray(perm[t][mode])
            obs = observed[t][mode]["auc"]
            p = float((1 + np.sum(null >= obs)) / (len(null) + 1))
            results[t][mode] = {
                "observed_auc": obs, "null_mean": float(null.mean()),
                "null_sd": float(null.std()), "null_p95": float(np.percentile(null, 95)),
                "p_value": p, "ci_lo": observed[t][mode]["ci_lo"],
                "ci_hi": observed[t][mode]["ci_hi"], "n_perm": int(len(null)),
            }
            print(f"{t:<9}{mode:<12}{obs:>9.3f}"
                  f"{f'{null.mean():.3f} ± {null.std():.3f}':>22}"
                  f"{np.percentile(null,95):>9.3f}{p:>10.4f}")

    print("\n" + "=" * 96)
    print("[핵심 해석]")
    nn_null = np.mean([results[t]["non_nested"]["null_mean"] for t in TARGETS])
    ne_null = np.mean([results[t]["nested"]["null_mean"] for t in TARGETS])
    print(f"  귀무분포 평균 (라벨이 완전 무작위인데도 나오는 AUC):")
    print(f"    non-nested : {nn_null:.3f}   <- 0.5보다 {nn_null-0.5:+.3f} 부풀려짐 = selection bias 실측치")
    print(f"    nested     : {ne_null:.3f}   <- 0.5에 정상적으로 수렴")

    (HERE / "ymj_perm_bootstrap_results.json").write_text(
        json.dumps({"results": results,
                    "observed": {t: {mo: {k: v for k, v in observed[t][mo].items() if k != "oof"}
                                     for mo in observed[t]} for t in TARGETS},
                    "n_perm": N_PERM, "n_boot": N_BOOT}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"\nSaved to {HERE/'ymj_perm_bootstrap_results.json'}")


if __name__ == "__main__":
    main()
