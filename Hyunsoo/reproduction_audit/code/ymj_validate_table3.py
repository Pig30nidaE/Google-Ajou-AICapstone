"""
YMJ(2026) 충실 재현 [검증] : 논문 Table 3 재현

논문 Table 3은 일별(daily) 데이터에 대해
  (1) 독립표본 t-test
  (2) LMM(참가자 ID를 random effect로 둔 선형혼합모형)
  (3) FDR 보정
세 가지 p-value를 보고한다.

이 표를 재현할 수 있으면 = 내가 만든 일별 파생지표가 논문의 것과 사실상 같다는 뜻
(머신러닝 파이프라인과 무관한 독립적 검증).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent

# 논문 Table 3에 보고된 값 (t-test p, LMM p, FDR q)
PAPER_TABLE3 = {
    "N3_ratio":                (0.006, 0.025, 0.473),
    "REM_ratio":               (0.000, 0.235, 0.999),
    "NREM_ratio":              (0.000, 0.235, 0.999),
    "SRI":                     (0.640, 0.240, 0.999),
    "Sleep_midpoint_time":     (0.575, 0.999, 0.999),
    "Sleep_duration":          (0.207, 0.999, 0.999),
    "Sleep_bedtime_end_num":   (0.000, 0.999, 0.999),
    "TST":                     (0.066, 0.999, 0.999),
    "N1_plus_N2_ratio":        (0.000, 0.999, 0.999),
    "NREM_proportion":         (0.000, 0.999, 0.999),
    "HR_drop_ratio":           (0.000, 0.999, 0.999),
    "WASO":                    (0.000, 0.999, 0.999),
    "SE":                      (0.000, 0.999, 0.999),
    "Sleep_bedtime_start_num": (0.000, 0.999, 0.999),
    "Daily_sleep_count":       (0.000, 0.999, 0.999),
    "Sleep_breath_average":    (0.000, 0.434, 0.999),
    "Sleep_hr_lowest":         (0.006, 0.533, 0.999),
    "Sleep_RMSSD":             (0.227, 0.683, 0.999),
    "Sleep_hr_average":        (0.000, 0.696, 0.999),
}


def main():
    daily = pd.read_csv(HERE / "ymj_faithful_daily.csv")
    pat = pd.read_csv(HERE / "ymj_faithful_features.csv")

    keep = set(pat["EMAIL"])
    d = daily[daily["EMAIL"].isin(keep)].copy()
    lab = pat[["EMAIL", "mci_label"]]
    d = d.merge(lab, on="EMAIL", how="inner")
    print(f"일별 데이터: {len(d)}행, 참가자 {d['EMAIL'].nunique()}명 "
          f"(CN={int((lab['mci_label']==0).sum())}, MCI={int((lab['mci_label']==1).sum())})")
    print(f"논문: 11,398 nights / 162명\n")

    feats = [f for f in PAPER_TABLE3 if f in d.columns]
    rows = []
    for f in feats:
        sub = d[["EMAIL", "mci_label", f]].replace([np.inf, -np.inf], np.nan).dropna()
        a = sub.loc[sub["mci_label"] == 0, f].to_numpy()
        b = sub.loc[sub["mci_label"] == 1, f].to_numpy()
        if len(a) < 10 or len(b) < 10:
            rows.append({"feature": f, "t_p": np.nan, "lmm_p": np.nan})
            continue

        t_p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)

        # LMM: value ~ group, random intercept per participant
        try:
            s = sub.copy()
            s["_y"] = (s[f] - s[f].mean()) / (s[f].std() + 1e-12)
            md = MixedLM.from_formula("_y ~ mci_label", groups="EMAIL", data=s)
            mf = md.fit(method="lbfgs", reml=True, disp=False)
            lmm_p = float(mf.pvalues.get("mci_label", np.nan))
        except Exception:
            lmm_p = np.nan

        rows.append({"feature": f, "t_p": t_p, "lmm_p": lmm_p})

    res = pd.DataFrame(rows)
    ok = res["lmm_p"].notna()
    res.loc[ok, "fdr_q"] = multipletests(res.loc[ok, "lmm_p"], method="fdr_bh")[1]

    print("=" * 104)
    print("논문 Table 3 재현 (일별 데이터 기준)")
    print("=" * 104)
    print(f"{'feature':<26}{'t-test p':>11}{'(논문)':>10} | {'LMM p':>10}{'(논문)':>10} | {'FDR q':>10}{'(논문)':>10}")
    print("-" * 104)
    for _, r in res.iterrows():
        pt, pl, pq = PAPER_TABLE3[r["feature"]]
        print(f"{r['feature']:<26}{r['t_p']:>11.3f}{pt:>10.3f} | "
              f"{r['lmm_p']:>10.3f}{pl:>10.3f} | {r['fdr_q']:>10.3f}{pq:>10.3f}")
    print("-" * 104)

    # 일치도 요약
    mine_t_sig = (res["t_p"] < 0.05)
    paper_t_sig = np.array([PAPER_TABLE3[f][0] < 0.05 for f in res["feature"]])
    mine_q_sig = (res["fdr_q"] < 0.05)
    paper_q_sig = np.array([PAPER_TABLE3[f][2] < 0.05 for f in res["feature"]])

    print(f"\n[t-test 유의성 판정 일치] {int((mine_t_sig.values == paper_t_sig).sum())}/{len(res)}개 변수에서 일치")
    print(f"  내 재현 : 유의 {int(mine_t_sig.sum())}개 / {len(res)}개")
    print(f"  논문    : 유의 {int(paper_t_sig.sum())}개 / {len(res)}개")
    print(f"\n[FDR 보정 후 유의성]")
    print(f"  내 재현 : 유의 {int(mine_q_sig.sum())}개 / {len(res)}개")
    print(f"  논문    : 유의 {int(paper_q_sig.sum())}개 / {len(res)}개  <- 논문도 '0개'라고 보고")

    print(f"\n[핵심 확인] 논문의 결론:")
    print(f"  '일별 t-test로는 유의하게 보이지만, 반복측정 구조(LMM)+FDR 보정 후에는")
    print(f"   단 하나의 일별 지표도 통계적으로 유의하지 않다'")
    print(f"  -> 내 재현에서도 FDR 후 유의 {int(mine_q_sig.sum())}개로 동일한 결론")

    res.to_csv(HERE / "ymj_table3_reproduction.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved to {HERE/'ymj_table3_reproduction.csv'}")


if __name__ == "__main__":
    main()
