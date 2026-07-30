"""
V26/V29 노트북의 cell-6 전처리 로직을 그대로 재현하여
로컬 원본 CSV로부터 patient_level_all_v2.csv를 생성한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

DATA_ROOT = Path("/Users/sasaek/코딩/ML/aihub_data")
OUT_DIR = Path("/private/tmp/claude-501/-Users-sasaek----ML-aihub-data/75346f87-d882-44a3-87ed-52441e217762/scratchpad/nia219_experiment")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

TRAIN_ACTIVITY = DATA_ROOT / "1.Training/원천데이터/1.걸음걸이/train_activity.csv"
VAL_ACTIVITY = DATA_ROOT / "2.Validation/원천데이터/1.걸음걸이/val_activity.csv"
TRAIN_SLEEP = DATA_ROOT / "1.Training/원천데이터/2.수면/train_sleep.csv"
VAL_SLEEP = DATA_ROOT / "2.Validation/원천데이터/2.수면/val_sleep.csv"
TRAIN_LABEL = DATA_ROOT / "1.Training/라벨링데이터/3.인지기능/training_label.csv"
VAL_LABEL = DATA_ROOT / "2.Validation/라벨링데이터/3.인지기능/val_label.csv"


def read_csv_flexible(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Failed to read {path} with common Korean/UTF-8 encodings.")


def coerce_numeric_columns(df: pd.DataFrame, keep: set[str]) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col not in keep:
            converted = pd.to_numeric(df[col], errors="coerce")
            if not converted.isna().all():
                df[col] = converted
    return df


def preprocess_label(label_df: pd.DataFrame) -> pd.DataFrame:
    label_df = label_df.copy()
    if "SAMPLE_EMAIL" in label_df.columns:
        label_df = label_df.rename(columns={"SAMPLE_EMAIL": "EMAIL"})

    required = {"EMAIL", "DIAG_NM"}
    missing = required - set(label_df.columns)
    if missing:
        raise ValueError(f"Label file is missing columns: {sorted(missing)}")

    original_label_map = {"CN": 0, "MCI": 1, "Dem": 2, "Dementia": 2}
    binary_label_map = {"CN": 0, "MCI": 1, "Dem": 1, "Dementia": 1}
    label_df["original_label"] = label_df["DIAG_NM"].map(original_label_map)
    label_df["label"] = label_df["DIAG_NM"].map(binary_label_map)

    if label_df["label"].isna().any():
        counts = label_df["DIAG_NM"].value_counts(dropna=False)
        raise ValueError(f"Unmapped DIAG_NM values found:\n{counts}")

    return label_df[["EMAIL", "DIAG_NM", "original_label", "label"]].drop_duplicates("EMAIL")


def parse_slash_sequence(value, dtype=float) -> np.ndarray:
    if pd.isna(value):
        return np.array([], dtype=float)

    text = str(value).strip()
    if not text or text == "...":
        return np.array([], dtype=float)

    values = []
    for token in text.split("/"):
        token = token.strip()
        if not token or token == "...":
            continue
        try:
            values.append(dtype(token))
        except ValueError:
            continue

    return np.array(values, dtype=float)


def numeric_sequence_stats(seq: np.ndarray, prefix: str, remove_zero: bool = False) -> dict[str, float]:
    arr = np.asarray(seq, dtype=float)
    arr = arr[arr != -1]
    if remove_zero:
        arr = arr[arr != 0]

    if len(arr) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_var": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_q25": np.nan,
            f"{prefix}_q75": np.nan,
            f"{prefix}_iqr": np.nan,
            f"{prefix}_valid_count": 0,
        }

    q25 = np.percentile(arr, 25)
    q75 = np.percentile(arr, 75)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_var": float(np.var(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_q25": float(q25),
        f"{prefix}_q75": float(q75),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_valid_count": int(len(arr)),
    }


def activity_class_features(seq: np.ndarray) -> dict[str, float]:
    arr = np.asarray(seq, dtype=float)
    arr = arr[arr != -1]
    total = len(arr)
    out: dict[str, float] = {}

    for level in range(6):
        count = int(np.sum(arr == level))
        out[f"activity_class_{level}_count"] = count
        out[f"activity_class_{level}_ratio"] = count / total if total else np.nan

    out["activity_rest_ratio"] = out["activity_class_1_ratio"]
    out["activity_inactive_ratio"] = out["activity_class_2_ratio"]
    out["activity_active_ratio"] = (
        out["activity_class_3_ratio"] + out["activity_class_4_ratio"] + out["activity_class_5_ratio"]
        if total
        else np.nan
    )
    out["activity_not_worn_ratio"] = out["activity_class_0_ratio"]
    out["activity_class_valid_count"] = total
    return out


def sleep_hypnogram_features(seq: np.ndarray) -> dict[str, float]:
    arr = np.asarray(seq, dtype=float)
    arr = arr[(arr != -1) & (arr != 0)]
    total = len(arr)
    out: dict[str, float] = {}
    stage_map = {1: "deep", 2: "light", 3: "rem", 4: "awake"}

    for level, name in stage_map.items():
        count = int(np.sum(arr == level))
        out[f"sleep_{name}_count_5min"] = count
        out[f"sleep_{name}_ratio_5min"] = count / total if total else np.nan

    out["sleep_stage_transition_count"] = int(np.sum(arr[1:] != arr[:-1])) if total > 1 else 0
    out["sleep_hypnogram_valid_count"] = total
    return out


def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rows = []

    for _, row in df.iterrows():
        feats = {}
        feats.update(
            activity_class_features(
                parse_slash_sequence(row["CONVERT(activity_class_5min USING utf8)"], dtype=float)
            )
        )
        feats.update(
            numeric_sequence_stats(
                parse_slash_sequence(row["CONVERT(activity_met_1min USING utf8)"], dtype=float),
                "activity_met_1min",
                remove_zero=False,
            )
        )
        feats.update(
            numeric_sequence_stats(
                parse_slash_sequence(row["CONVERT(sleep_hr_5min USING utf8)"], dtype=float),
                "sleep_hr_5min",
                remove_zero=True,
            )
        )
        feats.update(
            numeric_sequence_stats(
                parse_slash_sequence(row["CONVERT(sleep_rmssd_5min USING utf8)"], dtype=float),
                "sleep_rmssd_5min",
                remove_zero=True,
            )
        )
        feats.update(
            sleep_hypnogram_features(
                parse_slash_sequence(row["CONVERT(sleep_hypnogram_5min USING utf8)"], dtype=float)
            )
        )
        rows.append(feats)

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def make_daily_table(activity: pd.DataFrame, sleep: pd.DataFrame, label: pd.DataFrame) -> pd.DataFrame:
    activity = coerce_numeric_columns(activity, keep={"EMAIL", "activity_day_start", "activity_day_end"})
    sleep = coerce_numeric_columns(sleep, keep={"EMAIL", "sleep_bedtime_start", "sleep_bedtime_end"})

    activity["activity_day_start_dt"] = pd.to_datetime(activity["activity_day_start"], errors="coerce")
    activity["activity_day_end_dt"] = pd.to_datetime(activity["activity_day_end"], errors="coerce")
    sleep["sleep_bedtime_start_dt"] = pd.to_datetime(sleep["sleep_bedtime_start"], errors="coerce")
    sleep["sleep_bedtime_end_dt"] = pd.to_datetime(sleep["sleep_bedtime_end"], errors="coerce")

    activity["date"] = activity["activity_day_start_dt"].dt.date
    sleep["date"] = sleep["sleep_bedtime_end_dt"].dt.date

    for col in ("activity_day_start_dt", "sleep_bedtime_start_dt", "sleep_bedtime_end_dt", "date"):
        source = activity if col.startswith("activity") else sleep
        if source[col].isna().any():
            raise ValueError(f"{col} has unparsable values.")

    sleep_start = sleep["sleep_bedtime_start_dt"]
    sleep_end = sleep["sleep_bedtime_end_dt"]
    sleep["sleep_start_hour"] = sleep_start.dt.hour + sleep_start.dt.minute / 60 + sleep_start.dt.second / 3600
    sleep["sleep_end_hour"] = sleep_end.dt.hour + sleep_end.dt.minute / 60 + sleep_end.dt.second / 3600
    sleep["sleep_time_calculated"] = (sleep_end - sleep_start).dt.total_seconds() / 3600
    sleep["_sleep_duration_seconds"] = (sleep_end - sleep_start).dt.total_seconds()

    sleep = (
        sleep.sort_values(["EMAIL", "date", "_sleep_duration_seconds"], ascending=[True, True, False])
        .drop_duplicates(["EMAIL", "date"], keep="first")
        .reset_index(drop=True)
    )

    if activity.duplicated(["EMAIL", "date"]).any():
        dup = activity.loc[activity.duplicated(["EMAIL", "date"], keep=False), ["EMAIL", "date"]].head()
        raise ValueError(f"Activity has duplicate EMAIL/date rows. Examples:\n{dup}")

    daily = activity.merge(sleep, on=["EMAIL", "date"], how="inner", suffixes=("", "_sleep"))
    daily = daily.merge(label, on="EMAIL", how="left")
    if daily["label"].isna().any():
        raise ValueError("Some merged rows have missing labels.")

    daily = add_sequence_features(daily)
    daily = drop_unmodelable_columns(daily)
    daily = drop_single_value_columns(daily)
    daily = daily.replace([np.inf, -np.inf], np.nan)
    return daily


def drop_unmodelable_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        "activity_class_5min",
        "activity_met_1min",
        "sleep_hr_5min",
        "sleep_hypnogram_5min",
        "sleep_rmssd_5min",
        "CONVERT(activity_class_5min USING utf8)",
        "CONVERT(activity_met_1min USING utf8)",
        "CONVERT(sleep_hr_5min USING utf8)",
        "CONVERT(sleep_hypnogram_5min USING utf8)",
        "CONVERT(sleep_rmssd_5min USING utf8)",
        "activity_day_start",
        "activity_day_end",
        "activity_day_start_dt",
        "activity_day_end_dt",
        "sleep_bedtime_start",
        "sleep_bedtime_end",
        "sleep_bedtime_start_dt",
        "sleep_bedtime_end_dt",
        "_sleep_duration_seconds",
    ]
    return df.drop(columns=drop_cols, errors="ignore")


def drop_single_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    protected = {"EMAIL", "date", "DIAG_NM", "original_label", "label"}
    drop_cols = [
        col for col in df.columns
        if col not in protected and df[col].dropna().nunique() <= 1
    ]
    return df.drop(columns=drop_cols)


def make_patient_table(daily_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = daily_df.select_dtypes(include=[np.number]).columns
    exclude = {"original_label", "label", "fold"}
    features_to_agg = [c for c in numeric_cols if c not in exclude]

    df_mean = daily_df.groupby("EMAIL")[features_to_agg].mean().reset_index()

    df_std = daily_df.groupby("EMAIL")[features_to_agg].std().reset_index()
    std_rename = {col: col + "_std" for col in features_to_agg}
    df_std = df_std.rename(columns=std_rename)

    df_combined = df_mean.merge(df_std, on="EMAIL", how="left")
    df_combined = df_combined.fillna(0)

    labels = daily_df[["EMAIL", "DIAG_NM", "original_label", "label"]].drop_duplicates()
    return df_combined.merge(labels, on="EMAIL", how="left")


def main() -> None:
    print("[1/3] Loading raw CSVs...")
    train_activity = read_csv_flexible(TRAIN_ACTIVITY)
    val_activity = read_csv_flexible(VAL_ACTIVITY)
    train_sleep = read_csv_flexible(TRAIN_SLEEP)
    val_sleep = read_csv_flexible(VAL_SLEEP)
    train_label = preprocess_label(read_csv_flexible(TRAIN_LABEL))
    val_label = preprocess_label(read_csv_flexible(VAL_LABEL))

    print("[2/3] Building daily tables...")
    train_daily = make_daily_table(train_activity, train_sleep, train_label)
    val_daily = make_daily_table(val_activity, val_sleep, val_label)

    print("[3/3] Aggregating to patient level...")
    all_daily = pd.concat([train_daily, val_daily], ignore_index=True)
    patient_df = make_patient_table(all_daily)

    out_path = OUT_DIR / "patient_level_all_v2.csv"
    patient_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("Done.")
    print("patient_level_all:", patient_df.shape)
    print(patient_df["DIAG_NM"].value_counts())
    print("nia+219@rowan.kr in data:", (patient_df["EMAIL"] == "nia+219@rowan.kr").any())
    row = patient_df.loc[patient_df["EMAIL"] == "nia+219@rowan.kr"]
    if len(row):
        print(row[["EMAIL", "DIAG_NM", "original_label", "label"]])


if __name__ == "__main__":
    main()
