"""합성 AI-Hub 레이아웃 코호트 — Binary_DemScreen_NoMMSE_Nested 배선 검증 전용.
여기서 나오는 숫자는 어디에도 보고하지 않는다.  사용법: python make_synth_demscreen.py <out_dir> [seed]
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

out = Path(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 11
rng = np.random.default_rng(seed)
classes = ["CN"] * 111 + ["MCI"] * 51 + ["Dem"] * 12
sids = [f"syn{i:04d}@example.org" for i in range(174)]
order = rng.permutation(174); train_set = set(order[:141].tolist())

def met_to_class(met):
    e = met.reshape(288, 5).mean(1)
    c = np.full(288, 2, dtype=int)
    c[e < 0.5] = 0; c[(e >= 0.5) & (e < 1.0)] = 1
    c[(e >= 1.5) & (e < 3.0)] = 3; c[(e >= 3.0) & (e < 6.0)] = 4; c[e >= 6.0] = 5
    return c

act = {True: [], False: []}; slp = {True: [], False: []}; lab = {True: [], False: []}
for j, (s, c) in enumerate(zip(sids, classes)):
    tr = j in train_set
    n_days = int(rng.integers(40, 61))
    start = pd.Timestamp("2020-06-01") + pd.Timedelta(int(rng.integers(0, 150)), unit="D")
    drop = set(rng.choice(np.arange(n_days), size=int(rng.integers(0, 4)), replace=False).tolist())
    disp = {"CN": 85, "MCI": 72, "Dem": 45}[c] * rng.uniform(0.75, 1.25)   # ← 신호는 '분산'
    level = rng.uniform(180, 380)
    lab[tr].append({"SAMPLE_EMAIL": s, "DIAG_NM": c})
    for d in range(n_days):
        if d in drop:
            continue
        day = start + pd.Timedelta(d, unit="D")
        met = np.full(1440, 0.9)
        on, off = int(rng.integers(90, 240)), int(rng.integers(950, 1200)); t = on
        while t < off:
            n1 = min(int(rng.integers(5, 70)), off - t); met[t:t + n1] = rng.choice([0.9, 1.1, 1.2, 1.3], size=n1); t += n1
            if t >= off:
                break
            n2 = min(int(rng.integers(5, 60)), off - t); met[t:t + n2] = rng.uniform(1.6, 3.5, size=n2); t += n2
        if rng.random() < 0.10:
            a = int(rng.integers(0, 1400)); met[a:a + int(rng.integers(5, 100))] = 0.1
        cls = met_to_class(met)
        low = max(0, int(level + rng.normal(0, disp)))
        med_, hi_ = int(rng.integers(5, 90)), int(rng.integers(0, 25))
        nonwear = int((met < 0.5).sum()); steps = int(max(0, 6000 + 25 * low + rng.normal(0, 1500)))
        row = {"EMAIL": s, "activity_day_start": day.strftime("%Y-%m-%dT04:00:00+09:00"),
               "activity_non_wear": nonwear, "activity_score": rng.integers(50, 95),
               "activity_steps": steps, "activity_rest": int(rng.integers(400, 800)),
               "activity_inactive": int(rng.integers(300, 700)), "activity_low": low,
               "activity_medium": med_, "activity_high": hi_,
               "activity_daily_movement": int(steps * rng.uniform(.5, .9)),
               "activity_average_met": float(np.round(met.mean(), 3)),
               "activity_cal_active": int(rng.integers(100, 900)),
               "activity_cal_total": int(rng.integers(1500, 3000)),
               "activity_inactivity_alerts": int(rng.integers(0, 6)),
               "activity_met_min_high": hi_, "activity_met_min_medium": med_,
               "activity_met_min_low": low, "activity_met_min_inactive": int(rng.integers(200, 600)),
               "activity_total": low + med_ + hi_,
               "CONVERT(activity_class_5min USING utf8)": "/".join(map(str, cls.tolist())) + "/",
               "CONVERT(activity_met_1min USING utf8)": "/".join(f"{v:.1f}" for v in met) + "/"}
        for k in ("meet_daily_targets", "move_every_hour", "recovery_time", "stay_active",
                  "training_frequency", "training_volume"):
            row[f"activity_score_{k}"] = int(rng.integers(40, 100))
        act[tr].append(row)
        dur = int(rng.normal(7.2 * 3600, 3600)); dur = max(3600, dur)
        bt_end = day + pd.Timedelta(int(rng.integers(5 * 60, 9 * 60)), unit="m")
        bt_start = bt_end - pd.Timedelta(dur, unit="s")
        nep = max(12, int(round(dur / 300)))
        hyp = rng.choice([1, 2, 3, 4], size=nep, p=[.17, .52, .13, .18])
        hyp[0] = 4
        hr = np.clip(rng.normal(58, 6, nep), 35, 110).astype(int)
        hr[rng.random(nep) < .10] = 0
        rms = np.clip(rng.normal(35, 12, nep), 5, 200).astype(int)
        rms[rng.random(nep) < .10] = 0
        deep = int(dur * rng.uniform(.10, .22)); light = int(dur * rng.uniform(.40, .60))
        rem = int(dur * rng.uniform(.10, .22)); awake = max(0, dur - deep - light - rem)
        slp[tr].append({
            "EMAIL": s, "sleep_bedtime_start": bt_start.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
            "sleep_bedtime_end": bt_end.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
            "sleep_duration": dur, "sleep_efficiency": int(rng.integers(70, 96)),
            "sleep_awake": awake, "sleep_deep": deep, "sleep_light": light, "sleep_rem": rem,
            "sleep_restless": int(rng.integers(5, 60)), "sleep_onset_latency": int(rng.integers(60, 1800)),
            "sleep_midpoint_time": int(rng.integers(8000, 16000)),
            "sleep_hr_average": float(np.round(rng.normal(58, 5), 1)),
            "sleep_hr_lowest": float(np.round(rng.normal(50, 5), 1)),
            "sleep_rmssd": int(rng.integers(15, 70)), "sleep_breath_average": float(np.round(rng.normal(15, 2), 1)),
            "sleep_score": int(rng.integers(40, 95)), "sleep_score_deep": int(rng.integers(20, 100)),
            "sleep_temperature_deviation": float(np.round(rng.normal(0, .3), 2)),
            "sleep_score_alignment": int(rng.integers(20, 100)), "sleep_score_disturbances": int(rng.integers(20, 100)),
            "sleep_score_efficiency": int(rng.integers(20, 100)), "sleep_score_latency": int(rng.integers(20, 100)),
            "sleep_score_rem": int(rng.integers(20, 100)), "sleep_score_total": int(rng.integers(20, 100)),
            "sleep_total": deep + light + rem,
            "CONVERT(sleep_hypnogram_5min USING utf8)": "/".join(map(str, hyp.tolist())) + "/",
            "CONVERT(sleep_hr_5min USING utf8)": "/".join(map(str, hr.tolist())) + "/",
            "CONVERT(sleep_rmssd_5min USING utf8)": "/".join(map(str, rms.tolist())) + "/"})

for tr, split, pa, ps, pl in [(True, "1.Training", "train_activity.csv", "train_sleep.csv", "training_label.csv"),
                              (False, "2.Validation", "val_activity.csv", "val_sleep.csv", "val_label.csv")]:
    for sub in ["SourceData/1.Gait", "SourceData/2.Sleep", "LabelingData/1.Gait", "LabelingData/2.Sleep"]:
        (out / "Data" / split / sub).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(act[tr]).to_csv(out / "Data" / split / "SourceData" / "1.Gait" / pa, index=False)
    pd.DataFrame(slp[tr]).to_csv(out / "Data" / split / "SourceData" / "2.Sleep" / ps, index=False)
    for g in ["1.Gait", "2.Sleep"]:
        pd.DataFrame(lab[tr]).to_csv(out / "Data" / split / "LabelingData" / g / pl, index=False)
print("written", out / "Data", "| train", len(lab[True]), "val", len(lab[False]),
      "| activity rows", len(act[True]) + len(act[False]))
