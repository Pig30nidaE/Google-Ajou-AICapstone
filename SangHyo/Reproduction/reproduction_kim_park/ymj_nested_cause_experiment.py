"""Targeted audit: did nesting itself lower the reported YMJ performance?

This script intentionally does not reproduce the broad A/H conditions.  It runs
the four smallest comparisons needed to separate the relevant mechanisms for
the paper's primary Exp.11 feature set:

  B strict_tuned       strict inner-CV selection of C, class weight, and k
  C strict_fixed       strict fold-local RFE with C=1, no class weight, k=30
  F no_rfe_fixed       fold-local preprocessing, no RFE, same fixed LR
  G global_rfe_fixed   full-label RFE once, then fold-local model preprocessing

All conditions share identical repeated outer splits.  Existing evidence files
are never read as inputs and existing outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


HERE = Path(__file__).resolve().parent


def discover_data_root() -> Path:
    """Find a sibling/cwd/repository Data folder without machine-specific paths."""
    candidates = [HERE / "Data", Path.cwd() / "Data"]
    candidates.extend(parent / "Data" for parent in HERE.parents)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (HERE / "Data").resolve()


DEFAULT_DATA = discover_data_root()

MIN_NIGHTS = 35
MISSING_THRESHOLD = 0.20
N_OUTER = 10
N_INNER = 5
THRESHOLD = 0.5
BASE_SEED = 42
FIXED_C = 1.0
FIXED_CW = None
FIXED_K = 30
STEP = 5
C_GRID = (0.1, 0.3, 1.0, 3.0)
CW_GRID = (None, "balanced")
K_GRID = (20, 30, 50)
CONDITIONS = ("B_strict_tuned", "C_strict_fixed", "F_no_rfe_fixed", "G_global_rfe_fixed")

FAMILY_SUFFIXES = {
    "M": ("_mean",),
    "EM": ("_median", "_trimmed_mean", "_mode"),
    "Dist": ("_min", "_max", "_mad", "_kurtosis", "_range"),
    "Disp": ("_sd", "_cv", "_iqr"),
    "TS": ("_stv", "_tbv", "_rcv", "_mr", "_tbcr"),
}
EXP11_FAMILIES = ("EM", "Dist", "TS")
ALL_SUFFIXES = sorted(
    {suffix for suffixes in FAMILY_SUFFIXES.values() for suffix in suffixes},
    key=len,
    reverse=True,
)

HYP_COL = "CONVERT(sleep_hypnogram_5min USING utf8)"
HR_COL = "CONVERT(sleep_hr_5min USING utf8)"
WINDOW = 7
TIME_BINS = 4
EPS = 1e-8
DAILY_METRICS = (
    "TST",
    "Sleep_duration",
    "SE",
    "WASO",
    "N3_ratio",
    "REM_ratio",
    "N1_plus_N2_ratio",
    "NREM_ratio",
    "NREM_proportion",
    "Sleep_midpoint_time",
    "Sleep_bedtime_start_num",
    "Sleep_bedtime_end_num",
    "Sleep_hr_average",
    "Sleep_hr_lowest",
    "Sleep_RMSSD",
    "Sleep_breath_average",
    "HR_drop_ratio",
    "HR_drop_per_hour_nrem",
    "Daily_sleep_count",
    "SRI",
)


class ProgressBar:
    """Small tqdm wrapper used by the nested search loops."""

    def __init__(self, total: int):
        self.bar = tqdm(
            total=max(total, 1),
            desc="YMJ nested audit",
            unit="step",
            dynamic_ncols=True,
            smoothing=0.1,
        )

    def update(self, label: str) -> None:
        self.bar.set_postfix_str(label, refresh=False)
        self.bar.update(1)

    def message(self, text: str) -> None:
        tqdm.write(text)

    def close(self) -> None:
        self.bar.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Standalone YMJ Exp.11 nested-CV cause experiment (B/C/F/G)."
    )
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--mode", choices=("smoke", "full"), default="full")
    p.add_argument("--repeats", type=int, default=None, help="Override mode default (smoke=1, full=10).")
    p.add_argument("--bootstrap", type=int, default=None, help="Override mode default (smoke=200, full=2000).")
    p.add_argument("--seed", type=int, default=BASE_SEED)
    return p.parse_args()


def read_csv_flexible(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode {path}")


def preprocess_label(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "SAMPLE_EMAIL" in frame.columns:
        frame = frame.rename(columns={"SAMPLE_EMAIL": "EMAIL"})
    frame["original_label"] = frame["DIAG_NM"].map(
        {"CN": 0, "MCI": 1, "Dem": 2, "Dementia": 2}
    )
    return frame[["EMAIL", "DIAG_NM", "original_label"]].drop_duplicates("EMAIL")


def parse_sequence(value, *, drop_zero: bool = False) -> np.ndarray:
    if pd.isna(value):
        return np.asarray([], dtype=float)
    parsed = []
    for token in str(value).split("/"):
        token = token.strip()
        if not token or token == "...":
            continue
        try:
            parsed.append(float(token))
        except ValueError:
            continue
    array = np.asarray(parsed, dtype=float)
    array = array[array != -1]
    if drop_zero:
        array = array[array != 0]
    return array


def compute_daily_row(row) -> dict:
    total = row["sleep_total"]
    deep, light, rem, awake = (
        row["sleep_deep"],
        row["sleep_light"],
        row["sleep_rem"],
        row["sleep_awake"],
    )
    denominator = total if pd.notna(total) and total > 0 else np.nan
    nrem_rem = deep + light + rem
    result = {
        "TST": total,
        "Sleep_duration": row["sleep_duration"],
        "SE": row["sleep_efficiency"],
        "WASO": awake,
        "N3_ratio": deep / denominator,
        "REM_ratio": rem / denominator,
        "N1_plus_N2_ratio": light / denominator,
        "NREM_ratio": (deep + light) / denominator,
        "NREM_proportion": (deep + light) / nrem_rem if pd.notna(nrem_rem) and nrem_rem > 0 else np.nan,
        "Sleep_midpoint_time": row["sleep_midpoint_time"],
        "Sleep_bedtime_start_num": row["_start_hour"],
        "Sleep_bedtime_end_num": row["_end_hour"],
        "Sleep_hr_average": row["sleep_hr_average"],
        "Sleep_hr_lowest": row["sleep_hr_lowest"],
        "Sleep_RMSSD": row["sleep_rmssd"],
        "Sleep_breath_average": row["sleep_breath_average"],
    }
    heart_rate = parse_sequence(row[HR_COL], drop_zero=True)
    if len(heart_rate) >= 4:
        baseline = float(np.mean(heart_rate[:3]))
        drop = baseline - float(np.min(heart_rate))
        result["HR_drop_ratio"] = drop / baseline if baseline > 0 else np.nan
    else:
        drop = np.nan
        result["HR_drop_ratio"] = np.nan
    hypnogram = parse_sequence(row[HYP_COL], drop_zero=True)
    nrem_hours = (
        float(np.sum((hypnogram == 1) | (hypnogram == 2))) * 5.0 / 60.0
        if len(hypnogram)
        else np.nan
    )
    result["HR_drop_per_hour_nrem"] = (
        drop / nrem_hours if pd.notna(drop) and pd.notna(nrem_hours) and nrem_hours > 0 else np.nan
    )
    return result


def build_sleep_wake_grid(group: pd.DataFrame):
    starts, ends = group["_start_dt"].dropna(), group["_end_dt"].dropna()
    if starts.empty or ends.empty:
        return None, None
    start = starts.min().floor("D")
    end = ends.max().ceil("D")
    n_epochs = int((end - start).total_seconds() // 300)
    if n_epochs <= 0:
        return None, None
    grid = np.zeros(n_epochs, dtype=np.int8)
    for left, right in zip(group["_start_dt"], group["_end_dt"]):
        if pd.isna(left) or pd.isna(right) or right <= left:
            continue
        i0 = max(int((left - start).total_seconds() // 300), 0)
        i1 = min(int((right - start).total_seconds() // 300), n_epochs)
        if i1 > i0:
            grid[i0:i1] = 1
    return grid, start


def daily_sri(grid: np.ndarray) -> np.ndarray:
    per_day = 288
    n_days = len(grid) // per_day
    if n_days < 2:
        return np.asarray([], dtype=float)
    matrix = grid[: n_days * per_day].reshape(n_days, per_day)
    return np.asarray(
        [-100.0 + 200.0 * float(np.mean(matrix[i] == matrix[i + 1])) for i in range(n_days - 1)]
    )


def build_daily_table(sleep: pd.DataFrame) -> pd.DataFrame:
    sleep = sleep.copy()
    numeric_columns = (
        "sleep_total",
        "sleep_duration",
        "sleep_efficiency",
        "sleep_awake",
        "sleep_light",
        "sleep_deep",
        "sleep_rem",
        "sleep_midpoint_time",
        "sleep_hr_average",
        "sleep_hr_lowest",
        "sleep_rmssd",
        "sleep_breath_average",
        "sleep_period_id",
        "sleep_is_longest",
    )
    for column in numeric_columns:
        sleep[column] = pd.to_numeric(sleep[column], errors="coerce")
    sleep["_start_dt"] = pd.to_datetime(sleep["sleep_bedtime_start"], errors="coerce", utc=True)
    sleep["_end_dt"] = pd.to_datetime(sleep["sleep_bedtime_end"], errors="coerce", utc=True)
    before = len(sleep)
    sleep = sleep.dropna(subset=["_start_dt", "_end_dt"]).copy()
    print(f"timestamp filter: {before} -> {len(sleep)} rows", flush=True)
    sleep["_start_dt"] = sleep["_start_dt"].dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    sleep["_end_dt"] = sleep["_end_dt"].dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    sleep["_start_hour"] = (
        sleep["_start_dt"].dt.hour
        + sleep["_start_dt"].dt.minute / 60
        + sleep["_start_dt"].dt.second / 3600
    )
    sleep["_end_hour"] = (
        sleep["_end_dt"].dt.hour
        + sleep["_end_dt"].dt.minute / 60
        + sleep["_end_dt"].dt.second / 3600
    )
    sleep["_start_hour"] = np.where(sleep["_start_hour"] < 12, sleep["_start_hour"] + 24, sleep["_start_hour"])
    sleep["_date"] = sleep["_end_dt"].dt.date

    rows = []
    for email, group in sleep.groupby("EMAIL"):
        group = group.sort_values("_start_dt")
        counts = group.groupby("_date").size()
        episodes = group.copy()
        episodes["_duration_seconds"] = (
            episodes["_end_dt"] - episodes["_start_dt"]
        ).dt.total_seconds()
        main = (
            episodes.sort_values(["_date", "_duration_seconds"], ascending=[True, False])
            .drop_duplicates("_date", keep="first")
        )
        grid, first_day = build_sleep_wake_grid(group)
        sri_values = daily_sri(grid) if grid is not None else np.asarray([])
        sri_by_date = {}
        if len(sri_values) and first_day is not None:
            sri_by_date = {
                (first_day + pd.Timedelta(days=i)).date(): float(value)
                for i, value in enumerate(sri_values)
            }
        for _, row in main.iterrows():
            values = compute_daily_row(row)
            values.update(
                {
                    "EMAIL": email,
                    "date": row["_date"],
                    "Daily_sleep_count": float(counts.get(row["_date"], 1)),
                    "SRI": sri_by_date.get(row["_date"], np.nan),
                }
            )
            rows.append(values)
    daily = pd.DataFrame(rows)
    daily[list(DAILY_METRICS)] = daily[list(DAILY_METRICS)].replace([np.inf, -np.inf], np.nan)
    return daily


def short_term_variability(values: np.ndarray, window: int = WINDOW) -> float:
    if len(values) < window:
        return np.nan
    return float(
        np.mean([np.std(values[i : i + window], ddof=1) for i in range(len(values) - window + 1)])
    )


def rolling_cv(values: np.ndarray, window: int = WINDOW) -> float:
    if len(values) < window:
        return np.nan
    return float(
        np.mean(
            [
                np.std(values[i : i + window], ddof=1)
                / (abs(np.mean(values[i : i + window])) + EPS)
                for i in range(len(values) - window + 1)
            ]
        )
    )


def bin_means(values: np.ndarray, bins: int = TIME_BINS):
    if len(values) < bins:
        return None
    width = len(values) // bins
    if width == 0:
        return None
    return values[: width * bins].reshape(bins, width).mean(axis=1)


def summarize(values: np.ndarray, prefix: str) -> dict:
    values = values[~np.isnan(values)]
    keys = (
        "mean", "median", "trimmed_mean", "mode", "min", "max", "mad", "kurtosis",
        "range", "sd", "cv", "iqr", "stv", "tbv", "rcv", "mr", "tbcr",
    )
    if len(values) < 3:
        return {f"{prefix}_{key}": np.nan for key in keys}
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    rounded = np.round(values, 1)
    unique, counts = np.unique(rounded, return_counts=True)
    binned = bin_means(values)
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_trimmed_mean": float(stats.trim_mean(values, 0.1)),
        f"{prefix}_mode": float(unique[np.argmax(counts)]),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_mad": float(np.mean(np.abs(values - mean))),
        f"{prefix}_kurtosis": float(stats.kurtosis(values)) if len(values) > 3 else np.nan,
        f"{prefix}_range": float(np.max(values) - np.min(values)),
        f"{prefix}_sd": sd,
        f"{prefix}_cv": sd / (abs(mean) + EPS),
        f"{prefix}_iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        f"{prefix}_stv": short_term_variability(values),
        f"{prefix}_tbv": float(np.var(binned, ddof=1)) if binned is not None else np.nan,
        f"{prefix}_rcv": rolling_cv(values),
        f"{prefix}_mr": float(np.mean(np.abs(np.diff(values)))) if len(values) >= 2 else np.nan,
        f"{prefix}_tbcr": float(np.mean(np.abs(np.diff(binned)))) if binned is not None else np.nan,
    }


def aggregate_patients(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for email, group in daily.groupby("EMAIL"):
        group = group.sort_values("date")
        features = {"EMAIL": email, "n_nights": len(group)}
        for metric in DAILY_METRICS:
            features.update(summarize(group[metric].to_numpy(dtype=float), metric))
        rows.append(features)
    return pd.DataFrame(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def lr(C: float, class_weight) -> LogisticRegression:
    return LogisticRegression(
        penalty="l2",
        C=C,
        class_weight=class_weight,
        solver="lbfgs",
        max_iter=5000,
        random_state=BASE_SEED,
    )


def resolve_exp11(columns: list[str]) -> list[str]:
    wanted = {s for family in EXP11_FAMILIES for s in FAMILY_SUFFIXES[family]}
    return [
        col
        for col in columns
        if (suffix := next((s for s in ALL_SUFFIXES if col.endswith(s)), None)) in wanted
    ]


def feature_family(name: str) -> str:
    suffix = next((s for s in ALL_SUFFIXES if name.endswith(s)), None)
    if suffix is None:
        return "unknown"
    return next(family for family, suffixes in FAMILY_SUFFIXES.items() if suffix in suffixes)


def input_paths(data_root: Path) -> dict[str, Path]:
    return {
        "train_sleep": data_root / "1.Training/SourceData/2.Sleep/train_sleep.csv",
        "val_sleep": data_root / "2.Validation/SourceData/2.Sleep/val_sleep.csv",
        "train_label": data_root / "1.Training/LabelingData/3.CognitiveFunction/training_label.csv",
        "val_label": data_root / "2.Validation/LabelingData/3.CognitiveFunction/val_label.csv",
    }


def build_exp11(data_root: Path) -> tuple[pd.DataFrame, np.ndarray, list[str], dict]:
    paths = input_paths(data_root)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")

    sleep = pd.concat(
        [read_csv_flexible(paths["train_sleep"]), read_csv_flexible(paths["val_sleep"])],
        ignore_index=True,
    )
    daily = build_daily_table(sleep)
    patients = aggregate_patients(daily)
    labels = pd.concat(
        [
            preprocess_label(read_csv_flexible(paths["train_label"])),
            preprocess_label(read_csv_flexible(paths["val_label"])),
        ],
        ignore_index=True,
    ).drop_duplicates("EMAIL")

    cohort = patients.merge(labels, on="EMAIL", how="inner")
    cohort = cohort[cohort["n_nights"] >= MIN_NIGHTS]
    cohort = cohort[cohort["original_label"].isin([0, 1])].sort_values("EMAIL").reset_index(drop=True)
    y = cohort["original_label"].astype(int).to_numpy()
    meta = {"EMAIL", "n_nights", "DIAG_NM", "original_label", "mci_label"}
    candidates = resolve_exp11([c for c in cohort.columns if c not in meta])
    X = cohort[candidates].copy()
    selected_ids = set(cohort["EMAIL"])
    selected_raw_episodes = int(sleep["EMAIL"].isin(selected_ids).sum())

    info = {
        "raw_sleep_rows": int(len(sleep)),
        "selected_raw_episode_rows": selected_raw_episodes,
        "daily_rows_all_diagnoses": int(len(daily)),
        "n_subjects": int(len(cohort)),
        "class_counts": {"CN": int((y == 0).sum()), "MCI": int((y == 1).sum())},
        "n_exp11_candidates_before_fold_filter": int(len(candidates)),
        "nights": {
            "sum": int(cohort["n_nights"].sum()),
            "mean": float(cohort["n_nights"].mean()),
            "min": int(cohort["n_nights"].min()),
            "max": int(cohort["n_nights"].max()),
        },
        "input_hashes": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
    }
    return X, y, candidates, info


def fit_preprocessor(X_train: pd.DataFrame, X_other: pd.DataFrame):
    missing_rate = X_train.isna().mean()
    kept = missing_rate[missing_rate <= MISSING_THRESHOLD].index.tolist()
    if not kept:
        raise RuntimeError("Fold-local missingness filter removed every feature")
    imputer = SimpleImputer(strategy="median")
    train_i = imputer.fit_transform(X_train[kept])
    other_i = imputer.transform(X_other[kept])
    scaler = StandardScaler()
    train_s = scaler.fit_transform(train_i)
    other_s = scaler.transform(other_i)
    return train_s, other_s, kept, imputer, scaler


def fit_named_preprocessor(X_train: pd.DataFrame, X_other: pd.DataFrame, names: list[str]):
    """Fit preprocessing on outer training data while preserving a fixed feature set."""
    all_missing = [name for name in names if X_train[name].notna().sum() == 0]
    if all_missing:
        raise RuntimeError(f"Globally selected features are empty in an outer training fold: {all_missing}")
    imputer = SimpleImputer(strategy="median")
    train_i = imputer.fit_transform(X_train[names])
    other_i = imputer.transform(X_other[names])
    scaler = StandardScaler()
    return scaler.fit_transform(train_i), scaler.transform(other_i)


def direct_rfe(X: np.ndarray, y: np.ndarray, names: list[str], C: float, cw, k: int):
    k_eff = min(k, X.shape[1])
    selector = RFE(lr(C, cw), n_features_to_select=k_eff, step=STEP)
    selector.fit(X, y)
    indices = np.flatnonzero(selector.support_)
    selected = [names[i] for i in indices]
    return selector, indices, selected


def evaluate_metrics(y: np.ndarray, probability: np.ndarray) -> dict:
    prediction = (probability >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "accuracy": float(accuracy_score(y, prediction)),
        "sensitivity": float(recall_score(y, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def fit_fold_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    *,
    C: float,
    cw,
    k: int | None,
):
    Xtr, Xte, names, _, _ = fit_preprocessor(X_train, X_test)
    if k is not None:
        _, idx, selected = direct_rfe(Xtr, y_train, names, C, cw, k)
        Xtr, Xte = Xtr[:, idx], Xte[:, idx]
    else:
        selected = names
    model = lr(C, cw)
    model.fit(Xtr, y_train)
    probability = model.predict_proba(Xte)[:, 1]
    coefficients = {name: float(value) for name, value in zip(selected, model.coef_[0])}
    return probability, selected, coefficients, len(names)


def choose_strict_config(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
    progress: ProgressBar,
    repeat: int,
    outer_fold: int,
) -> tuple[float, object, int, float]:
    splitter = StratifiedKFold(N_INNER, shuffle=True, random_state=seed)
    folds = []
    for inner_train, inner_valid in splitter.split(X_train, y_train):
        Xtr, Xva, names, _, _ = fit_preprocessor(X_train.iloc[inner_train], X_train.iloc[inner_valid])
        folds.append((Xtr, Xva, y_train[inner_train], y_train[inner_valid], names))

    best = None
    for C, cw, k in itertools.product(C_GRID, CW_GRID, K_GRID):
        scores = []
        for Xtr, Xva, ytr, yva, names in folds:
            _, idx, _ = direct_rfe(Xtr, ytr, names, C, cw, k)
            model = lr(C, cw)
            model.fit(Xtr[:, idx], ytr)
            scores.append(roc_auc_score(yva, model.predict_proba(Xva[:, idx])[:, 1]))
        score = float(np.mean(scores))
        candidate = (score, C, cw, k)
        if best is None or score > best[0]:
            best = candidate
        progress.update(
            f"repeat {repeat + 1}, fold {outer_fold + 1}: C={C}, cw={cw}, k={k}"
        )
    assert best is not None
    return best[1], best[2], best[3], best[0]


def fit_global_selector(X: pd.DataFrame, y: np.ndarray):
    dummy = X.iloc[:1].copy()
    Xall, _, names, _, _ = fit_preprocessor(X, dummy)
    _, _, selected = direct_rfe(Xall, y, names, FIXED_C, FIXED_CW, FIXED_K)
    return selected


def run_experiment(X: pd.DataFrame, y: np.ndarray, repeats: int, base_seed: int):
    n = len(y)
    probabilities = {condition: np.full((repeats, n), np.nan) for condition in CONDITIONS}
    repeat_metrics = {condition: [] for condition in CONDITIONS}
    fold_metrics = {condition: [] for condition in CONDITIONS}
    selections = {condition: [] for condition in CONDITIONS}
    coefficients = {condition: [] for condition in CONDITIONS}
    picks = []
    fold_assignments = []

    configurations_per_fold = len(C_GRID) * len(CW_GRID) * len(K_GRID)
    progress = ProgressBar(1 + repeats * N_OUTER * (configurations_per_fold + 1))
    global_selected = fit_global_selector(X, y)
    progress.update("global RFE feature set complete")

    for repeat in range(repeats):
        outer_seed = base_seed + repeat
        outer = StratifiedKFold(N_OUTER, shuffle=True, random_state=outer_seed)
        for fold, (train_idx, test_idx) in enumerate(outer.split(X, y)):
            Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
            ytr, yte = y[train_idx], y[test_idx]
            fold_assignments.append(
                {"repeat": repeat, "fold": fold, "train_rows": train_idx.tolist(), "test_rows": test_idx.tolist()}
            )

            C, cw, k, inner_auc = choose_strict_config(
                Xtr,
                ytr,
                seed=10_000 + outer_seed * 100 + fold,
                progress=progress,
                repeat=repeat,
                outer_fold=fold,
            )
            picks.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "C": C,
                    "class_weight": str(cw),
                    "k": k,
                    "inner_auc": inner_auc,
                }
            )

            condition_specs = {
                "B_strict_tuned": (C, cw, k),
                "C_strict_fixed": (FIXED_C, FIXED_CW, FIXED_K),
                "F_no_rfe_fixed": (FIXED_C, FIXED_CW, None),
            }
            for condition, (model_C, model_cw, model_k) in condition_specs.items():
                prob, selected, coef, n_after_filter = fit_fold_model(
                    Xtr, ytr, Xte, C=model_C, cw=model_cw, k=model_k
                )
                probabilities[condition][repeat, test_idx] = prob
                selections[condition].append(selected)
                coefficients[condition].append(coef)
                fm = evaluate_metrics(yte, prob)
                fm.update(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "n_after_missing_filter": n_after_filter,
                        "n_selected": len(selected),
                    }
                )
                fold_metrics[condition].append(fm)

            # G: feature names are selected once with all labels, but every outer
            # model re-fits imputation/scaling on its own training subjects.
            Gtr, Gte = fit_named_preprocessor(Xtr, Xte, global_selected)
            global_model = lr(FIXED_C, FIXED_CW)
            global_model.fit(Gtr, ytr)
            prob = global_model.predict_proba(Gte)[:, 1]
            coef = {
                name: float(value)
                for name, value in zip(global_selected, global_model.coef_[0])
            }
            n_after_filter = len(global_selected)
            condition = "G_global_rfe_fixed"
            probabilities[condition][repeat, test_idx] = prob
            selections[condition].append(list(global_selected))
            coefficients[condition].append(coef)
            fm = evaluate_metrics(yte, prob)
            fm.update(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "n_after_missing_filter": n_after_filter,
                    "n_selected": len(global_selected),
                }
            )
            fold_metrics[condition].append(fm)
            progress.update(f"repeat {repeat + 1}, fold {fold + 1}: outer models complete")

        for condition in CONDITIONS:
            if np.isnan(probabilities[condition][repeat]).any():
                raise RuntimeError(f"Incomplete OOF predictions: {condition}, repeat {repeat}")
            value = evaluate_metrics(y, probabilities[condition][repeat])
            value["repeat"] = repeat
            repeat_metrics[condition].append(value)

        progress.message(
            f"repeat {repeat + 1}/{repeats}: "
            + ", ".join(
                f"{condition}={repeat_metrics[condition][-1]['roc_auc']:.4f}" for condition in CONDITIONS
            )
        )

    progress.close()

    return {
        "probabilities": probabilities,
        "repeat_metrics": repeat_metrics,
        "fold_metrics": fold_metrics,
        "selections": selections,
        "coefficients": coefficients,
        "picks": picks,
        "fold_assignments": fold_assignments,
        "global_selected": global_selected,
    }


def summarize_metrics(repeat_metrics: dict) -> dict:
    metric_names = (
        "roc_auc",
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "balanced_accuracy",
    )
    output = {}
    for condition, rows in repeat_metrics.items():
        output[condition] = {}
        for metric in metric_names:
            values = np.asarray([row[metric] for row in rows], dtype=float)
            output[condition][metric] = {
                "mean": float(values.mean()),
                "sd_across_splits": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
            }
    return output


def jaccard_summary(feature_sets: list[list[str]]) -> dict:
    sets = [set(value) for value in feature_sets]
    values = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            values.append(len(sets[i] & sets[j]) / len(union) if union else 1.0)
    if not values:
        return {"mean": 1.0, "median": 1.0, "min": 1.0, "max": 1.0, "n_pairs": 0}
    arr = np.asarray(values)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_pairs": int(len(arr)),
    }


def summarize_features(selections: dict) -> dict:
    output = {}
    for condition, sets in selections.items():
        counts = Counter(feature for selected in sets for feature in selected)
        family_counts = Counter(feature_family(feature) for selected in sets for feature in selected)
        output[condition] = {
            "n_fitted_models": len(sets),
            "jaccard": jaccard_summary(sets),
            "feature_counts": dict(counts.most_common()),
            "family_selection_counts": dict(sorted(family_counts.items())),
        }
    return output


def summarize_coefficients(coefficients: dict) -> dict:
    output = {}
    for condition, model_coefficients in coefficients.items():
        values = defaultdict(list)
        for coef in model_coefficients:
            for feature, value in coef.items():
                values[feature].append(value)
        rows = {}
        total_models = len(model_coefficients)
        for feature, vals in values.items():
            arr = np.asarray(vals)
            rows[feature] = {
                "selected_models": int(len(arr)),
                "selection_frequency": float(len(arr) / total_models),
                "mean": float(arr.mean()),
                "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "positive_fraction": float((arr > 0).mean()),
                "negative_fraction": float((arr < 0).mean()),
            }
        output[condition] = dict(
            sorted(rows.items(), key=lambda item: (-item[1]["selection_frequency"], item[0]))
        )
    return output


def paired_subject_bootstrap(y, probabilities, n_boot: int, seed: int) -> dict:
    metric_names = (
        "roc_auc",
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "balanced_accuracy",
    )
    comparisons = {
        "G_minus_C_global_selection_effect": ("G_global_rfe_fixed", "C_strict_fixed"),
        "B_minus_C_nested_tuning_effect": ("B_strict_tuned", "C_strict_fixed"),
        "F_minus_C_no_rfe_effect": ("F_no_rfe_fixed", "C_strict_fixed"),
    }
    rng = np.random.default_rng(seed)
    condition_draws = {condition: {metric: [] for metric in metric_names} for condition in CONDITIONS}
    difference_draws = {name: {metric: [] for metric in metric_names} for name in comparisons}
    accepted = 0
    while accepted < n_boot:
        idx = rng.integers(0, len(y), len(y))
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        draw_values = {}
        for condition in CONDITIONS:
            per_repeat = [evaluate_metrics(yy, row[idx]) for row in probabilities[condition]]
            draw_values[condition] = {
                metric: float(np.mean([result[metric] for result in per_repeat]))
                for metric in metric_names
            }
            for metric in metric_names:
                condition_draws[condition][metric].append(draw_values[condition][metric])
        for name, (left, right) in comparisons.items():
            for metric in metric_names:
                difference_draws[name][metric].append(draw_values[left][metric] - draw_values[right][metric])
        accepted += 1

    def describe(draws):
        result = {}
        for key, values in draws.items():
            arr = np.asarray(values)
            result[key] = {
                "bootstrap_mean": float(arr.mean()),
                "ci95_percentile": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
            }
        return result

    return {
        "method": (
            "Subjects resampled synchronously across all conditions; metrics computed for each repeat, "
            "then averaged across repeats. Percentile intervals use paired subject bootstrap draws."
        ),
        "n_bootstrap": n_boot,
        "conditions": {condition: describe(values) for condition, values in condition_draws.items()},
        "differences": {name: describe(values) for name, values in difference_draws.items()},
    }


def hyperparameter_summary(picks: list[dict]) -> dict:
    return {
        "C": dict(sorted(Counter(str(row["C"]) for row in picks).items())),
        "class_weight": dict(sorted(Counter(row["class_weight"] for row in picks).items())),
        "k": dict(sorted(Counter(str(row["k"]) for row in picks).items())),
        "joint": {
            "|".join(map(str, key)): count
            for key, count in Counter(
                (row["C"], row["class_weight"], row["k"]) for row in picks
            ).most_common()
        },
    }


def environment_info() -> dict:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "tqdm"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=HERE, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except Exception:
        commit, dirty = None, None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
        "git_commit": commit,
        "git_dirty_at_run": dirty,
    }


def main() -> None:
    args = parse_args()
    repeats = args.repeats if args.repeats is not None else (1 if args.mode == "smoke" else 10)
    n_bootstrap = args.bootstrap if args.bootstrap is not None else (200 if args.mode == "smoke" else 2000)
    if repeats < 1 or n_bootstrap < 1:
        raise ValueError("--repeats and --bootstrap must be positive")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        args.output.resolve()
        if args.output is not None
        else (HERE / "results" / f"ymj_nested_cause_{args.mode}_{timestamp}").resolve()
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    public = output / "public"
    private = output / "private"
    public.mkdir(parents=True)
    private.mkdir(parents=True)

    started = time.time()
    X, y, candidates, cohort_info = build_exp11(args.data_root.resolve())
    if len(y) != 162 or int((y == 0).sum()) != 111 or int((y == 1).sum()) != 51:
        raise RuntimeError(
            "Cohort guard failed: expected n=162 (CN=111, MCI=51), got "
            f"n={len(y)} (CN={(y == 0).sum()}, MCI={(y == 1).sum()})"
        )
    if len(candidates) != 260:
        raise RuntimeError(f"Feature guard failed: expected 260 pre-filter Exp.11 candidates, got {len(candidates)}")
    if cohort_info["selected_raw_episode_rows"] != 11398 or cohort_info["nights"]["sum"] != 11387:
        raise RuntimeError(
            "Observation-count guard failed: expected 11,398 selected raw episodes and 11,387 "
            f"unique end-date nights, got {cohort_info['selected_raw_episode_rows']} and "
            f"{cohort_info['nights']['sum']}"
        )
    print(
        f"Exp.11 cohort n={len(y)} (CN={(y == 0).sum()}, MCI={(y == 1).sum()}), "
        f"candidate features before fold filtering={len(candidates)}",
        flush=True,
    )

    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "YMJ Exp.11 targeted nested-cause audit",
        "mode": args.mode,
        "conditions": list(CONDITIONS),
        "repeats": repeats,
        "outer_folds": N_OUTER,
        "inner_folds": N_INNER,
        "base_seed": args.seed,
        "threshold": THRESHOLD,
        "minimum_nights": MIN_NIGHTS,
        "fold_local_missing_threshold": MISSING_THRESHOLD,
        "rfe_step": STEP,
        "fixed": {"C": FIXED_C, "class_weight": FIXED_CW, "k": FIXED_K},
        "grid": {"C": C_GRID, "class_weight": CW_GRID, "k": K_GRID},
        "bootstrap": n_bootstrap,
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
    }
    json_dump(public / "config.json", config)
    json_dump(public / "environment.json", environment_info())
    json_dump(public / "input_manifest.json", cohort_info)

    result = run_experiment(X, y, repeats, args.seed)
    aggregate = summarize_metrics(result["repeat_metrics"])
    bootstrap = paired_subject_bootstrap(y, result["probabilities"], n_bootstrap, args.seed + 90_000)

    json_dump(public / "aggregate_metrics.json", aggregate)
    json_dump(public / "repeat_metrics.json", result["repeat_metrics"])
    json_dump(public / "outer_fold_metrics.json", result["fold_metrics"])
    json_dump(public / "hyperparameter_counts.json", hyperparameter_summary(result["picks"]))
    json_dump(public / "selected_features_by_fold.json", result["selections"])
    json_dump(public / "feature_stability.json", summarize_features(result["selections"]))
    json_dump(public / "coefficient_stability.json", summarize_coefficients(result["coefficients"]))
    json_dump(public / "paired_subject_bootstrap.json", bootstrap)
    json_dump(public / "global_selected_features.json", result["global_selected"])

    results_to_attach = {
        "question": "Did nested evaluation itself lower performance, or did it remove optimistic global feature selection?",
        "primary_comparisons": {
            "G_minus_C": "Global full-label RFE versus strict fold-local RFE, with C/class_weight/k fixed.",
            "B_minus_C": "Strict nested hyperparameter selection versus a strict fixed configuration.",
            "F_minus_C": "No RFE versus strict fold-local RFE under the same fixed LR configuration.",
        },
        "aggregate_metrics": aggregate,
        "paired_subject_bootstrap_differences": bootstrap["differences"],
        "strict_tuned_hyperparameters": hyperparameter_summary(result["picks"]),
        "feature_jaccard": {
            condition: summarize_features(result["selections"])[condition]["jaccard"]
            for condition in CONDITIONS
        },
    }
    json_dump(public / "RESULTS_TO_ATTACH.json", results_to_attach)
    summary_lines = [
        "# YMJ Exp.11 Nested-CV Cause Experiment",
        "",
        f"- Mode: {args.mode}",
        f"- Repeated outer CV: {N_OUTER}-fold × {repeats}",
        f"- Inner CV for B: {N_INNER}-fold",
        f"- Subjects: {len(y)} (CN={int((y == 0).sum())}, MCI={int((y == 1).sum())})",
        "",
        "## Mean repeated-OOF metrics",
        "",
        "| Condition | ROC-AUC | Accuracy | Sensitivity | Specificity | F1 | Balanced accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        values = aggregate[condition]
        summary_lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                condition,
                values["roc_auc"]["mean"],
                values["accuracy"]["mean"],
                values["sensitivity"]["mean"],
                values["specificity"]["mean"],
                values["f1"]["mean"],
                values["balanced_accuracy"]["mean"],
            )
        )
    summary_lines.extend(
        [
            "",
            "## Paired subject-bootstrap AUC differences",
            "",
            "| Contrast | Mean difference | 95% percentile CI |",
            "|---|---:|---:|",
        ]
    )
    for contrast, values in bootstrap["differences"].items():
        auc = values["roc_auc"]
        lo, hi = auc["ci95_percentile"]
        summary_lines.append(
            f"| {contrast} | {auc['bootstrap_mean']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |"
        )
    summary_lines.extend(
        [
            "",
            "Interpretation signs:",
            "",
            "- G−C > 0: using all labels for feature selection creates optimistic performance.",
            "- B−C < 0: inner-CV tuning is less stable than the preregistered fixed setting.",
            "- F−C > 0: fold-local RFE harms performance relative to regularized all-feature LR.",
            "",
            "Subject-level OOF predictions and fold assignments are stored only in `private/` and are not included in the public ZIP.",
        ]
    )
    (public / "RUN_SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    np.savez_compressed(
        private / "subject_level_oof_predictions.npz",
        y=y,
        **{condition: values for condition, values in result["probabilities"].items()},
    )
    json_dump(private / "fold_assignments.json", result["fold_assignments"])

    runtime = {
        "elapsed_seconds": time.time() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
    }
    json_dump(public / "runtime.json", runtime)
    archive = shutil.make_archive(str(output / "PUBLIC_RESULTS_TO_ATTACH"), "zip", root_dir=public)
    print(f"Public attachment bundle: {archive}", flush=True)
    print(json.dumps({"aggregate": aggregate, "runtime": runtime}, indent=2), flush=True)


if __name__ == "__main__":
    main()
