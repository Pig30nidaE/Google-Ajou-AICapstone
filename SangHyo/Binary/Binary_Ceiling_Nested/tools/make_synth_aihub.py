"""Synthetic AI-Hub-layout cohort for wiring tests of Binary_Ceiling_Nested (numbers never reported).

Writes <out>/Data/{1.Training,2.Validation}/{SourceData/{1.Gait,2.Sleep},LabelingData/{1.Gait,2.Sleep}}
with the exact columns the notebook reads. 174 subjects (CN 111 / MCI 51 / Dem 12), 40-60 joined days each,
1440-token MET strings and 288-token class_5min strings (both with trailing '/'), sleep rows keyed by bedtime_end date.
A weak class signal in day-to-day dispersion is injected so the wiring shows non-degenerate AUCs.

Usage: python make_synth_aihub.py <out_dir> [seed]
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

out = Path(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 11
rng = np.random.default_rng(seed)
classes = ["CN"] * 111 + ["MCI"] * 51 + ["Dem"] * 12
sids = [f"syn{i:04d}@example.org" for i in range(174)]
order = rng.permutation(174)
train_set = set(order[:141].tolist())

def met_to_class(met):
    e = met.reshape(288, 5).mean(1)
    c = np.full(288, 2, dtype=int)
    c[e < 0.5] = 0; c[(e >= 0.5) & (e < 1.0)] = 1; c[(e >= 1.5) & (e < 3.0)] = 3; c[(e >= 3.0) & (e < 6.0)] = 4; c[e >= 6.0] = 5
    return c

act = {True: [], False: []}; slp = {True: [], False: []}; lab = {True: [], False: []}
for j, (s, c) in enumerate(zip(sids, classes)):
    tr = j in train_set
    n_days = int(rng.integers(40, 61))
    start = pd.Timestamp("2020-06-01") + pd.Timedelta(int(rng.integers(0, 120)), unit="D")
    drop = set(rng.choice(np.arange(n_days), size=int(rng.integers(0, 4)), replace=False).tolist())
    disp = {"CN": 85, "MCI": 70, "Dem": 50}[c] * rng.uniform(0.75, 1.25); level = rng.uniform(180, 380)
    bout_disp = {"CN": 45, "MCI": 38, "Dem": 30}[c] * rng.uniform(0.7, 1.3)
    lab[tr].append({"SAMPLE_EMAIL": s, "DIAG_NM": c})
    for d in range(n_days):
        if d in drop:
            continue
        day = start + pd.Timedelta(d, unit="D")
        met = np.full(1440, 0.9)
        on, off = int(rng.integers(90, 240)), int(rng.integers(950, 1200)); t = on
        longest_target = max(10, int(rng.normal(60, bout_disp)))
        while t < off:
            n1 = min(int(rng.integers(5, longest_target + 5)), off - t); met[t:t + n1] = rng.choice([0.9, 1.1, 1.2, 1.3], size=n1); t += n1
            if t >= off:
                break
            n2 = min(int(rng.integers(5, 60)), off - t); met[t:t + n2] = rng.uniform(1.6, 3.5, size=n2); t += n2
        if rng.random() < 0.12:
            a = int(rng.integers(0, 1400)); met[a:a + int(rng.integers(5, 120))] = 0.1
        if rng.random() < 0.01:
            met[200:400] = 0.0                                       # no-data sentinel run
        cls = met_to_class(met)
        low = max(0, int(level + rng.normal(0, disp)))
        medium = int(rng.integers(5, 90)); nonwear = int((met < 0.5).sum()); steps = int(max(0, 6000 + 25 * low + rng.normal(0, 1500)))
        met_str = "/".join(f"{v:.1f}" for v in met) + "/"
        cls_str = "/".join(str(int(v)) for v in cls) + "/"
        if rng.random() < 0.012:
            cls_str = "/".join(str(int(v)) for v in cls[:200]) + "/"    # truncated class stream (length gate)
        act[tr].append({"EMAIL": s, "activity_day_start": day.strftime("%Y-%m-%dT04:00:00+09:00"),
                        "activity_low": low, "activity_medium": medium, "activity_non_wear": nonwear,
                        "activity_steps": steps, "activity_total": low + medium,
                        "CONVERT(activity_class_5min USING utf8)": cls_str,
                        "CONVERT(activity_met_1min USING utf8)": met_str})
        dur = int(rng.normal(7.2 * 3600, 3600))
        bt_end = day + pd.Timedelta(int(rng.integers(5 * 60, 9 * 60)), unit="m")
        bt_start = bt_end - pd.Timedelta(dur, unit="s")
        slp[tr].append({"EMAIL": s, "sleep_bedtime_start": bt_start.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
                        "sleep_bedtime_end": bt_end.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
                        "sleep_duration": dur, "sleep_total": int(dur * rng.uniform(0.8, 0.95))})
        if rng.random() < 0.02:                                         # duplicate sleep row (shorter)
            slp[tr].append({**slp[tr][-1], "sleep_duration": dur - 600})

for tr, split, pa, ps, pl in [(True, "1.Training", "train_activity.csv", "train_sleep.csv", "training_label.csv"),
                              (False, "2.Validation", "val_activity.csv", "val_sleep.csv", "val_label.csv")]:
    for sub in ["SourceData/1.Gait", "SourceData/2.Sleep", "LabelingData/1.Gait", "LabelingData/2.Sleep"]:
        (out / "Data" / split / sub).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(act[tr]).to_csv(out / "Data" / split / "SourceData" / "1.Gait" / pa, index=False)
    pd.DataFrame(slp[tr]).to_csv(out / "Data" / split / "SourceData" / "2.Sleep" / ps, index=False)
    for g in ["1.Gait", "2.Sleep"]:
        pd.DataFrame(lab[tr]).to_csv(out / "Data" / split / "LabelingData" / g / pl, index=False)
print("written", out / "Data", "| train", len(lab[True]), "val", len(lab[False]), "| activity rows", len(act[True]) + len(act[False]))
