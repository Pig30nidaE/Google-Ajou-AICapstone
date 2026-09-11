"""Synthetic new-cohort generator in the AI-Hub / Oura v1 export schema (wiring tests only; numbers are never reported).

Writes to <out>/:
  activity.csv, sleep.csv           - the wearable export (analyst side, no labels, no covariates)
  manifest.csv                      - steward-supplied label-free subject list (column: sid)
  labels_SEALED.csv                 - steward side: sid, DIAG_NM, age, sex, edu_years, edu_cat, gait_speed, site, enroll_month
Options inject the QC cases the sealed scorer must handle: calendar gaps, duplicate rows, malformed MET rows,
MET==0.0 no-data runs, a subject with < 35 joined days, a subject with NaN activity_low, extra sids not in the manifest,
withdrawn sids (in manifest, no data) and a duplicated manifest row.

Usage: python make_synthetic_cohort.py --out DIR [--n 120] [--days 45] [--seed 7] [--dirty]
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def make(out: Path, n: int = 120, days: int = 45, seed: int = 7, dirty: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    n_cn, n_mci = int(round(n * 0.6)), int(round(n * 0.3)); n_dem = n - n_cn - n_mci
    classes = ["CN"] * n_cn + ["MCI"] * n_mci + ["Dem"] * n_dem
    sids = [f"syn+{i:04d}@example.org" for i in range(n)]
    act_rows, slp_rows, lab_rows = [], [], []
    for j, (s, c) in enumerate(zip(sids, classes)):
        n_days = days if not (dirty and j == 0) else 30                      # j==0: < 35 days -> insufficient_days
        start = pd.Timestamp("2027-03-01") + pd.Timedelta(int(rng.integers(0, 60)), unit="D")
        drop = set(rng.choice(np.arange(n_days), size=int(rng.integers(0, 4)), replace=False).tolist())
        disp = {"CN": 80, "MCI": 70, "Dem": 50}[c] * rng.uniform(0.7, 1.3); level = rng.uniform(200, 350)
        age = {"CN": 71, "MCI": 74, "Dem": 78}[c] + rng.normal(0, 6); edu = max(0, {"CN": 10, "MCI": 9, "Dem": 7}[c] + rng.normal(0, 4))
        lab_rows.append({"sid": s, "DIAG_NM": c, "age": round(float(age), 1), "sex": rng.choice(["F", "M"]),
                         "edu_years": round(float(edu), 1), "edu_cat": ["none", "elementary", "middle", "high", "college"][min(4, int(edu // 3))],
                         "gait_speed": round(float({"CN": 1.05, "MCI": 0.98, "Dem": 0.8}[c] + rng.normal(0, 0.15)), 2) if rng.random() < 0.92 else np.nan,
                         "site": rng.choice(["A", "B"]), "enroll_month": int(start.month)})
        for d in range(n_days):
            if d in drop:
                continue
            day = start + pd.Timedelta(d, unit="D")
            met = np.full(1440, 0.9)
            on, off = int(rng.integers(90, 220)), int(rng.integers(950, 1150)); t = on
            while t < off:
                n1 = min(int(rng.integers(5, 90)), off - t); met[t:t + n1] = rng.choice([0.9, 1.1, 1.2, 1.3], size=n1); t += n1
                if t >= off:
                    break
                n2 = min(int(rng.integers(5, 60)), off - t); met[t:t + n2] = rng.uniform(1.6, 3.5, size=n2); t += n2
            if rng.random() < 0.15:
                a = int(rng.integers(0, 1400)); met[a:a + int(rng.integers(5, 120))] = 0.1
            if dirty and j == 1 and d == 3:
                met[100:600] = 0.0                                             # MET==0.0 no-data run (> 60 min)
            low = max(0, int(level + rng.normal(0, disp)))
            if dirty and j == 2:
                low = np.nan                                                    # NaN activity_low -> feature_unavailable
            medium = int(rng.integers(10, 90)); nonwear = int((met < 0.5).sum())
            met_str = "/".join(f"{v:.1f}" for v in met) + "/"
            if dirty and j == 3 and d == 5:
                met_str = "/".join(f"{v:.1f}" for v in met[:1400]) + "/"       # malformed MET (1400 tokens) -> schema_fail day
            act_rows.append({"EMAIL": s, "activity_day_start": day.strftime("%Y-%m-%dT04:00:00+09:00"),
                             "activity_low": low, "activity_medium": medium, "activity_non_wear": nonwear,
                             "activity_total": (0 if pd.isna(low) else int(low)) + medium,
                             "CONVERT(activity_met_1min USING utf8)": met_str})
            if dirty and j == 4 and d == 2:                                     # duplicate activity row (same sid, date)
                act_rows.append(dict(act_rows[-1], activity_total=act_rows[-1]["activity_total"] - 5))
            wake = day + pd.Timedelta(int(rng.integers(300, 480)), unit="m"); bed = wake - pd.Timedelta(int(rng.integers(360, 600)), unit="m")
            dur = int((wake - bed).total_seconds())
            slp_rows.append({"EMAIL": s, "sleep_bedtime_start": bed.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
                             "sleep_bedtime_end": wake.strftime("%Y-%m-%dT%H:%M:%S+09:00"), "sleep_duration": dur, "sleep_total": int(dur * 0.85)})
            if dirty and j == 5 and d == 4:                                     # duplicate sleep row (same end date, shorter)
                slp_rows.append(dict(slp_rows[-1], sleep_duration=dur - 3600, sleep_total=int((dur - 3600) * 0.85)))
    A, Sl, L = pd.DataFrame(act_rows), pd.DataFrame(slp_rows), pd.DataFrame(lab_rows)
    manifest = pd.DataFrame({"sid": sids})
    if dirty:
        extra = A[A["EMAIL"] == sids[-1]].copy(); extra["EMAIL"] = "syn+9999@example.org"            # data not in manifest
        A = pd.concat([A, extra], ignore_index=True)
        Sl = pd.concat([Sl, Sl[Sl["EMAIL"] == sids[-1]].assign(EMAIL="syn+9999@example.org")], ignore_index=True)
        manifest = pd.concat([manifest, pd.DataFrame({"sid": ["syn+8888@example.org", sids[7]]})], ignore_index=True)  # withdrawn + duplicate row
    A.to_csv(out / "activity.csv", index=False); Sl.to_csv(out / "sleep.csv", index=False)
    manifest.to_csv(out / "manifest.csv", index=False); L.to_csv(out / "labels_SEALED.csv", index=False)
    return {"n": n, "activity_rows": len(A), "sleep_rows": len(Sl), "manifest_rows": len(manifest), "dirty": dirty, "out": str(out)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--days", type=int, default=45); ap.add_argument("--seed", type=int, default=7); ap.add_argument("--dirty", action="store_true")
    a = ap.parse_args(); print(make(Path(a.out), a.n, a.days, a.seed, a.dirty))
