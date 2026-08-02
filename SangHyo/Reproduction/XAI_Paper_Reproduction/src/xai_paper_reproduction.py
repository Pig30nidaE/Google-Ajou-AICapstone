from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PAPER_MODEL_METRICS = {
    "LightGBM": {
        "accuracy": 0.8262,
        "roc_auc": 0.9010,
        "precision_macro": 0.8276,
        "recall_macro": 0.7904,
        "f1_macro": 0.8025,
    },
    "Random forest": {
        "accuracy": 0.8055,
        "roc_auc": 0.8835,
        "precision_macro": 0.8325,
        "recall_macro": 0.7491,
        "f1_macro": 0.7659,
    },
    "Decision tree": {
        "accuracy": 0.7041,
        "roc_auc": 0.6806,
        "precision_macro": 0.6808,
        "recall_macro": 0.6806,
        "f1_macro": 0.6807,
    },
    "K-Nearest Neighbor": {
        "accuracy": 0.6572,
        "roc_auc": 0.6595,
        "precision_macro": 0.6229,
        "recall_macro": 0.6111,
        "f1_macro": 0.6136,
    },
    "Multi-Layer Perceptron": {
        "accuracy": 0.5953,
        "roc_auc": 0.6348,
        "precision_macro": 0.6188,
        "recall_macro": 0.5634,
        "f1_macro": 0.5142,
    },
    "Support vector machine": {
        "accuracy": 0.6393,
        "roc_auc": 0.6249,
        "precision_macro": 0.6609,
        "recall_macro": 0.5083,
        "f1_macro": 0.4114,
    },
    "Logistic regression": {
        "accuracy": 0.6457,
        "roc_auc": 0.6067,
        "precision_macro": 0.6113,
        "recall_macro": 0.5331,
        "f1_macro": 0.4830,
    },
}

PAPER_FINAL_PARAMS = {
    "min_child_samples": 41,
    "num_leaves": 330,
    "n_estimators": 1000,
    "learning_rate": 0.08,
}

PAPER_SELECTED_FEATURE_COUNT = 40


@dataclass(frozen=True)
class ProjectPaths:
    repo_root: Path
    reproduction_root: Path
    raw_dir: Path
    output_dir: Path

    @classmethod
    def from_script(
        cls,
        script_path: str | Path,
        raw_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> "ProjectPaths":
        script_path = Path(script_path).resolve()
        reproduction_root = script_path.parents[1]
        repo_root = reproduction_root.parents[1]
        raw = Path(raw_dir).expanduser() if raw_dir else repo_root / "128.치매 고위험군 라이프로그"
        out = Path(output_dir).expanduser() if output_dir else reproduction_root / "outputs"
        return cls(
            repo_root=repo_root,
            reproduction_root=reproduction_root,
            raw_dir=raw.resolve(),
            output_dir=out.resolve(),
        )


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_flexible(path: Path, **kwargs) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"Unable to decode {path}")


def normalize_path_text(value: str | Path) -> str:
    return unicodedata.normalize("NFC", str(value))


def find_one(root: Path, filename: str, must_contain: tuple[str, ...] = ()) -> Path:
    required_parts = tuple(normalize_path_text(part) for part in must_contain)
    candidates = [
        p
        for p in root.rglob(filename)
        if all(part in normalize_path_text(p) for part in required_parts)
    ]
    if not candidates:
        raise FileNotFoundError(f"Cannot find {filename} under {root} with filters {must_contain}")
    candidates = sorted(candidates, key=lambda p: len(str(p)))
    return candidates[0]


def load_raw_frames(raw_dir: Path) -> dict[str, pd.DataFrame]:
    files = {
        "train_activity": find_one(raw_dir, "train_activity.csv"),
        "train_sleep": find_one(raw_dir, "train_sleep.csv"),
        "train_label": find_one(raw_dir, "training_label.csv", ("라벨링데이터", "1.걸음걸이")),
        "val_activity": find_one(raw_dir, "val_activity.csv"),
        "val_sleep": find_one(raw_dir, "val_sleep.csv"),
        "val_label": find_one(raw_dir, "val_label.csv", ("라벨링데이터", "1.걸음걸이")),
    }
    return {name: read_csv_flexible(path) for name, path in files.items()}


def preprocess_label(label_df: pd.DataFrame, split: str) -> pd.DataFrame:
    label = label_df.copy()
    if "SAMPLE_EMAIL" in label.columns:
        label = label.rename(columns={"SAMPLE_EMAIL": "patient_id"})
    elif "EMAIL" in label.columns:
        label = label.rename(columns={"EMAIL": "patient_id"})
    else:
        raise ValueError("Label file must contain SAMPLE_EMAIL or EMAIL")

    if "DIAG_NM" not in label.columns:
        raise ValueError("Label file must contain DIAG_NM")

    label["diagnosis"] = label["DIAG_NM"].astype(str)
    label["binary_class"] = label["diagnosis"].map({"CN": 0, "MCI": 1, "Dem": 1, "DEM": 1, "Dementia": 1})
    if label["binary_class"].isna().any():
        raise ValueError(f"Unmapped diagnosis values: {label['diagnosis'].value_counts(dropna=False).to_dict()}")
    label["split"] = split
    return label[["patient_id", "diagnosis", "binary_class", "split"]].drop_duplicates("patient_id")


def timestamp_hour(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    return dt.dt.hour + dt.dt.minute / 60.0 + dt.dt.second / 3600.0


def timestamp_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date.astype(str)


def parse_slash_sequence(value: Any) -> np.ndarray:
    if pd.isna(value):
        return np.array([], dtype=float)
    if not isinstance(value, str):
        value = str(value)
    if value.strip() in {"", "..."}:
        return np.array([], dtype=float)
    out = []
    for token in value.split("/"):
        token = token.strip()
        if not token or token == "...":
            continue
        try:
            out.append(float(token))
        except ValueError:
            continue
    return np.asarray(out, dtype=float)


def numeric_stats(values: np.ndarray, prefix: str, *, drop_zero: bool = False) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr != -1]
    if drop_zero:
        arr = arr[arr != 0]
    keys = ["mean", "std", "var", "min", "max", "median", "q25", "q75", "iqr", "count"]
    if len(arr) == 0:
        return {f"{prefix}_{k}": np.nan for k in keys}
    q25, q75 = np.percentile(arr, [25, 75])
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
        f"{prefix}_count": float(len(arr)),
    }


def categorical_counts(values: np.ndarray, prefix: str, labels: list[int]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr != -1]
    total = len(arr)
    out: dict[str, float] = {f"{prefix}_valid_count": float(total)}
    for label in labels:
        count = float(np.sum(arr == label))
        out[f"{prefix}_count_{label}"] = count
        out[f"{prefix}_ratio_{label}"] = count / total if total else np.nan
    out[f"{prefix}_transition_count"] = float(np.sum(arr[1:] != arr[:-1])) if total > 1 else 0.0
    return out


def clean_column_name(name: str) -> str:
    name = name.replace("CONVERT(", "").replace(" USING utf8)", "")
    name = re.sub(r"[^0-9A-Za-z가-힣_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def build_activity_daily(activity: pd.DataFrame) -> pd.DataFrame:
    df = activity.copy()
    df = df.rename(columns={"EMAIL": "patient_id"})
    df["sample_date"] = timestamp_date(df["activity_day_start"])
    df["activity_day_start_hour"] = timestamp_hour(df["activity_day_start"])
    df["activity_day_end_hour"] = timestamp_hour(df["activity_day_end"])

    seq_activity_class = "CONVERT(activity_class_5min USING utf8)"
    seq_activity_met = "CONVERT(activity_met_1min USING utf8)"

    seq_features = []
    for _, row in df.iterrows():
        feats = {}
        if seq_activity_class in df.columns:
            feats.update(categorical_counts(parse_slash_sequence(row[seq_activity_class]), "activity_class_5min", [0, 1, 2, 3, 4, 5]))
        if seq_activity_met in df.columns:
            feats.update(numeric_stats(parse_slash_sequence(row[seq_activity_met]), "activity_met_1min"))
        seq_features.append(feats)
    seq_df = pd.DataFrame(seq_features)

    drop_cols = [
        "activity_day_start",
        "activity_day_end",
        "activity_class_5min",
        "activity_met_1min",
        "CONVERT(activity_class_5min USING utf8)",
        "CONVERT(activity_met_1min USING utf8)",
    ]
    keep = df.drop(columns=[c for c in drop_cols if c in df.columns])
    out = pd.concat([keep.reset_index(drop=True), seq_df.reset_index(drop=True)], axis=1)
    return normalize_feature_frame(out)


def build_sleep_daily(sleep: pd.DataFrame) -> pd.DataFrame:
    df = sleep.copy()
    df = df.rename(columns={"EMAIL": "patient_id"})
    start = pd.to_datetime(df["sleep_bedtime_start"], errors="coerce")
    end = pd.to_datetime(df["sleep_bedtime_end"], errors="coerce")
    df["sample_date"] = end.dt.date.astype(str)
    df["sleep_bedtime_start_hour"] = timestamp_hour(df["sleep_bedtime_start"])
    df["sleep_bedtime_end_hour"] = timestamp_hour(df["sleep_bedtime_end"])
    df["sleep_time_from_timestamp"] = (end - start).dt.total_seconds()
    df["_sleep_duration_seconds"] = df["sleep_time_from_timestamp"]

    # When multiple sleep rows exist for one day, use the longest main sleep row.
    df = (
        df.sort_values(["patient_id", "sample_date", "_sleep_duration_seconds"], ascending=[True, True, False])
        .drop_duplicates(["patient_id", "sample_date"], keep="first")
        .reset_index(drop=True)
    )

    seq_hypnogram = "CONVERT(sleep_hypnogram_5min USING utf8)"
    seq_features = []
    for _, row in df.iterrows():
        feats = {}
        if seq_hypnogram in df.columns:
            feats.update(categorical_counts(parse_slash_sequence(row[seq_hypnogram]), "sleep_hypnogram_5min", [1, 2, 3, 4]))
        seq_features.append(feats)
    seq_df = pd.DataFrame(seq_features)

    # Paper states 5-minute heart-rate logs with unresolvable missingness were removed.
    drop_cols = [
        "sleep_bedtime_start",
        "sleep_bedtime_end",
        "sleep_hr_5min",
        "sleep_hypnogram_5min",
        "sleep_rmssd_5min",
        "CONVERT(sleep_hr_5min USING utf8)",
        "CONVERT(sleep_hypnogram_5min USING utf8)",
        "CONVERT(sleep_rmssd_5min USING utf8)",
        "_sleep_duration_seconds",
    ]
    keep = df.drop(columns=[c for c in drop_cols if c in df.columns])
    out = pd.concat([keep.reset_index(drop=True), seq_df.reset_index(drop=True)], axis=1)
    return normalize_feature_frame(out)


def normalize_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        if col in {"patient_id", "sample_date", "split", "diagnosis", "binary_class"}:
            continue
        renamed[col] = clean_column_name(col)
    return df.rename(columns=renamed)


def make_daily_binary_dataset(raw_dir: Path, *, merge_policy: str = "left_activity") -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    raw = load_raw_frames(raw_dir)
    train_label = preprocess_label(raw["train_label"], "train")
    val_label = preprocess_label(raw["val_label"], "val")
    labels = pd.concat([train_label, val_label], ignore_index=True)

    activity = pd.concat(
        [
            build_activity_daily(raw["train_activity"]).assign(split="train"),
            build_activity_daily(raw["val_activity"]).assign(split="val"),
        ],
        ignore_index=True,
    )
    sleep = pd.concat(
        [
            build_sleep_daily(raw["train_sleep"]).assign(split="train"),
            build_sleep_daily(raw["val_sleep"]).assign(split="val"),
        ],
        ignore_index=True,
    )

    merge_cols = ["patient_id", "sample_date", "split"]
    if merge_policy == "left_activity":
        daily = activity.merge(sleep, on=merge_cols, how="left", suffixes=("", "_sleepdup"))
    elif merge_policy == "inner":
        daily = activity.merge(sleep, on=merge_cols, how="inner", suffixes=("", "_sleepdup"))
    else:
        raise ValueError("merge_policy must be left_activity or inner")

    duplicate_cols = [c for c in daily.columns if c.endswith("_sleepdup")]
    daily = daily.drop(columns=duplicate_cols)
    daily = daily.merge(labels, on=["patient_id", "split"], how="left")

    if daily["binary_class"].isna().any():
        raise ValueError("Some daily rows did not receive labels")

    daily["binary_class"] = daily["binary_class"].astype(int)
    daily = daily.sort_values(["split", "patient_id", "sample_date"]).reset_index(drop=True)

    non_features = {"patient_id", "sample_date", "split", "diagnosis", "binary_class"}
    feature_cols = [c for c in daily.columns if c not in non_features]
    for col in feature_cols:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")

    feature_cols = remove_unusable_features(daily, feature_cols)
    daily = daily[["patient_id", "sample_date", "split", "diagnosis", "binary_class", *feature_cols]]

    summary = {
        "rows": int(len(daily)),
        "subjects": int(daily["patient_id"].nunique()),
        "class_counts": {str(k): int(v) for k, v in daily["binary_class"].value_counts().sort_index().items()},
        "split_counts": {str(k): int(v) for k, v in daily["split"].value_counts().items()},
        "feature_count": int(len(feature_cols)),
        "merge_policy": merge_policy,
        "paper_target_rows": 12183,
        "paper_target_class_counts": {"0": 7737, "1": 4446},
    }
    return daily, feature_cols, summary


def remove_unusable_features(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    kept = []
    for col in feature_cols:
        s = df[col]
        if s.notna().sum() == 0:
            continue
        nunique = s.dropna().nunique()
        if nunique <= 1:
            continue
        kept.append(col)
    return kept


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_cv(y: pd.Series, groups: pd.Series | None, *, n_splits: int, random_state: int, grouped: bool):
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    if grouped:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(
            np.zeros(len(y)), y, groups
        )
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(
        np.zeros(len(y)), y
    )


def import_lightgbm():
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError("LightGBM is required. Install with: pip install lightgbm") from exc
    return LGBMClassifier


def make_median_imputer():
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median")
    try:
        imputer.set_output(transform="pandas")
    except Exception:
        pass
    return imputer


def model_registry(random_state: int = 42) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    LGBMClassifier = import_lightgbm()
    scaled = lambda estimator: Pipeline(
        [
            ("imputer", make_median_imputer()),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )
    tree_pipe = lambda estimator: Pipeline(
        [
            ("imputer", make_median_imputer()),
            ("model", estimator),
        ]
    )
    return {
        "Logistic regression": scaled(
            LogisticRegression(max_iter=3000, solver="lbfgs", random_state=random_state)
        ),
        "Decision tree": tree_pipe(
            DecisionTreeClassifier(random_state=random_state)
        ),
        "K-Nearest Neighbor": scaled(
            KNeighborsClassifier(n_neighbors=5)
        ),
        "Support vector machine": scaled(
            SVC(kernel="rbf", probability=True, random_state=random_state)
        ),
        "Multi-Layer Perceptron": scaled(
            MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, early_stopping=True, random_state=random_state)
        ),
        "Random forest": tree_pipe(
            RandomForestClassifier(n_estimators=500, random_state=random_state, n_jobs=-1)
        ),
        "LightGBM": tree_pipe(
            LGBMClassifier(random_state=random_state, n_jobs=-1, verbosity=-1)
        ),
    }


def lgbm_pipeline(params: dict[str, Any] | None = None, random_state: int = 42) -> Pipeline:
    from sklearn.pipeline import Pipeline

    LGBMClassifier = import_lightgbm()
    params = dict(params or {})
    params.setdefault("random_state", random_state)
    params.setdefault("n_jobs", -1)
    params.setdefault("verbosity", -1)
    return Pipeline(
        [
            ("imputer", make_median_imputer()),
            ("model", LGBMClassifier(**params)),
        ]
    )


def evaluate_cv_model(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    groups: pd.Series | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    grouped: bool = False,
) -> dict[str, Any]:
    from sklearn.base import clone

    y = pd.Series(y).astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)
    groups = groups.reset_index(drop=True) if groups is not None else None

    pred = np.zeros(len(y), dtype=int)
    prob = np.zeros(len(y), dtype=float)
    fold_metrics = []

    for fold, (train_idx, valid_idx) in enumerate(make_cv(y, groups, n_splits=n_splits, random_state=random_state, grouped=grouped)):
        model = clone(estimator)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_prob = positive_probability(model, X.iloc[valid_idx])
        fold_pred = (fold_prob >= 0.5).astype(int)
        pred[valid_idx] = fold_pred
        prob[valid_idx] = fold_prob
        fold_metrics.append(metrics_dict(y.iloc[valid_idx].to_numpy(), fold_pred, fold_prob, prefix={"fold": fold}))

    overall = metrics_dict(y.to_numpy(), pred, prob)
    overall["fold_metrics"] = fold_metrics
    overall["oof_prediction"] = pred.tolist()
    overall["oof_probability"] = prob.tolist()
    return overall


def positive_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
    elif isinstance(model, Pipeline) and hasattr(model[-1], "predict_proba"):
        proba = model.predict_proba(X)
    else:
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    proba = np.asarray(proba)
    if proba.ndim == 1:
        return proba
    return proba[:, 1]


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, prefix: dict[str, Any] | None = None) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    out = dict(prefix or {})
    out.update(
        {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision_positive": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "recall_positive": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "f1_positive": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        }
    )
    return out


def compare_against_paper(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in metrics_df.iterrows():
        model = row["model"]
        paper = PAPER_MODEL_METRICS.get(model)
        if not paper:
            continue
        merged = {"model": model}
        for key, value in paper.items():
            merged[f"paper_{key}"] = value
            merged[f"repro_{key}"] = float(row[key])
            merged[f"delta_{key}"] = float(row[key]) - value
        rows.append(merged)
    return pd.DataFrame(rows)


def compute_oof_shap_importance(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    params: dict[str, Any] | None = None,
    groups: pd.Series | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    grouped: bool = False,
    sample_per_fold: int | None = None,
) -> pd.DataFrame:
    import shap

    y = pd.Series(y).astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)
    groups = groups.reset_index(drop=True) if groups is not None else None
    feature_names = list(X.columns)
    shap_accum = []

    for fold, (train_idx, valid_idx) in enumerate(make_cv(y, groups, n_splits=n_splits, random_state=random_state, grouped=grouped)):
        pipe = lgbm_pipeline(params=params, random_state=random_state + fold)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        Xt_valid = transformed_features(pipe, X.iloc[valid_idx], feature_names)
        if sample_per_fold and len(Xt_valid) > sample_per_fold:
            Xt_valid = Xt_valid.sample(sample_per_fold, random_state=random_state + fold)
        model = pipe.named_steps["model"]
        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(Xt_valid)
        shap_pos = positive_class_shap(raw)
        shap_accum.append(np.abs(shap_pos))

    all_abs = np.vstack(shap_accum)
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.mean(all_abs, axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance["rank"] = np.arange(1, len(importance) + 1)
    return importance[["rank", "feature", "mean_abs_shap"]]


def transformed_features(pipe: Pipeline, X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    Xt = pipe.named_steps["imputer"].transform(X)
    return pd.DataFrame(Xt, columns=feature_names, index=X.index)


def positive_class_shap(raw: Any) -> np.ndarray:
    if isinstance(raw, list):
        if len(raw) == 1:
            return np.asarray(raw[0])
        return np.asarray(raw[1])
    arr = np.asarray(raw)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        return arr[:, :, 1]
    if arr.ndim == 3 and arr.shape[0] == 2:
        return arr[1]
    return arr


def run_forward_selection(
    X: pd.DataFrame,
    y: pd.Series,
    ranked_features: list[str],
    *,
    params: dict[str, Any] | None = None,
    max_features: int = 80,
    n_splits: int = 5,
    random_state: int = 42,
    groups: pd.Series | None = None,
    grouped: bool = False,
) -> pd.DataFrame:
    rows = []
    max_features = min(max_features, len(ranked_features))
    for k in range(1, max_features + 1):
        feats = ranked_features[:k]
        metrics = evaluate_cv_model(
            lgbm_pipeline(params=params, random_state=random_state),
            X[feats],
            y,
            groups=groups,
            n_splits=n_splits,
            random_state=random_state,
            grouped=grouped,
        )
        rows.append(
            {
                "n_features": k,
                "roc_auc": metrics["roc_auc"],
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
            }
        )
    out = pd.DataFrame(rows)
    out["paper_selected_feature_count"] = PAPER_SELECTED_FEATURE_COUNT
    return out


def plot_forward_selection(metrics: pd.DataFrame, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    ensure_dir(path.parent)
    plt.figure(figsize=(10, 5))
    plt.plot(metrics["n_features"], metrics["roc_auc"], marker="o", linewidth=1)
    plt.axvline(PAPER_SELECTED_FEATURE_COUNT, color="tab:red", linestyle="--", label="Paper top 40")
    best = metrics.sort_values("roc_auc", ascending=False).iloc[0]
    plt.scatter([best["n_features"]], [best["roc_auc"]], color="black", zorder=3, label=f"Best k={int(best['n_features'])}")
    plt.xlabel("Number of SHAP-ranked features")
    plt.ylabel("5-fold ROC-AUC")
    plt.title("Forward Feature Selection by SHAP Importance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def grid_search_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    feature_cols: list[str],
    grid: list[dict[str, Any]],
    n_splits: int = 5,
    random_state: int = 42,
    groups: pd.Series | None = None,
    grouped: bool = False,
) -> pd.DataFrame:
    rows = []
    for i, params in enumerate(grid):
        metrics = evaluate_cv_model(
            lgbm_pipeline(params=params, random_state=random_state),
            X[feature_cols],
            y,
            groups=groups,
            n_splits=n_splits,
            random_state=random_state,
            grouped=grouped,
        )
        row = {"grid_id": i, **params}
        for key in ["accuracy", "roc_auc", "precision_macro", "recall_macro", "f1_macro"]:
            row[key] = metrics[key]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def default_lgbm_grid() -> list[dict[str, Any]]:
    grid = []
    for num_leaves in [300, 320, 330, 340]:
        for min_child_samples in [31, 41, 51]:
            for learning_rate in [0.05, 0.08, 0.1]:
                for n_estimators in [600, 1000]:
                    grid.append(
                        {
                            "num_leaves": num_leaves,
                            "min_child_samples": min_child_samples,
                            "learning_rate": learning_rate,
                            "n_estimators": n_estimators,
                        }
                    )
    return grid


def final_shap_and_drs(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    *,
    feature_cols: list[str],
    params: dict[str, Any],
    output_dir: Path,
    random_state: int = 42,
    max_drs_rows: int | None = None,
) -> dict[str, Any]:
    import joblib
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    from scipy import stats

    ensure_dir(output_dir)
    X = X.reset_index(drop=True)
    y = pd.Series(y).astype(int).reset_index(drop=True)
    meta = meta.reset_index(drop=True)

    pipe = lgbm_pipeline(params=params, random_state=random_state)
    pipe.fit(X[feature_cols], y)
    joblib.dump(pipe, output_dir / "final_lgbm_pipeline.joblib")

    model = pipe.named_steps["model"]
    full_prob = pipe.predict_proba(X[feature_cols])[:, 1]
    full_pred = (full_prob >= 0.5).astype(int)
    cv_like = metrics_dict(y.to_numpy(dtype=int), full_pred, full_prob)

    eval_idx = np.arange(len(y))
    sample_note = {"sampled": False, "requested_max_rows": max_drs_rows, "used_rows": int(len(eval_idx))}
    if max_drs_rows is not None and len(eval_idx) > max_drs_rows:
        if max_drs_rows < y.nunique():
            raise ValueError("max_drs_rows must be at least the number of target classes.")
        from sklearn.model_selection import train_test_split

        eval_idx, _ = train_test_split(
            np.arange(len(y)),
            train_size=max_drs_rows,
            stratify=y,
            random_state=random_state,
        )
        eval_idx = np.sort(eval_idx)
        sample_note = {"sampled": True, "requested_max_rows": int(max_drs_rows), "used_rows": int(len(eval_idx))}

    X_eval = X.iloc[eval_idx].reset_index(drop=True)
    meta_eval = meta.iloc[eval_idx].reset_index(drop=True)
    prob = full_prob[eval_idx]
    pred = full_pred[eval_idx]
    Xt = transformed_features(pipe, X_eval[feature_cols], feature_cols)

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(Xt)
    shap_pos = positive_class_shap(raw)
    drs = np.maximum(shap_pos, 0).sum(axis=1)

    shap_df = pd.DataFrame(shap_pos, columns=feature_cols)
    shap_df.insert(0, "row_id", eval_idx)
    shap_df.to_csv(output_dir / "shap_values_positive_class.csv", index=False, encoding="utf-8-sig")

    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "mean_abs_shap": np.abs(shap_pos).mean(axis=0),
            "mean_signed_shap": shap_pos.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(output_dir / "shap_importance_positive.csv", index=False, encoding="utf-8-sig")

    risk = meta_eval.copy()
    risk.insert(0, "row_id", eval_idx)
    risk["predicted_probability"] = prob
    risk["predicted_class"] = pred
    risk["dementia_risk_score"] = drs
    risk.to_csv(output_dir / "dementia_risk_scores.csv", index=False, encoding="utf-8-sig")

    summary = risk.groupby("binary_class")["dementia_risk_score"].agg(["count", "min", "max", "mean", "std"]).reset_index()
    summary.to_csv(output_dir / "dementia_risk_score_summary.csv", index=False, encoding="utf-8-sig")

    cn_mean = risk.loc[risk["binary_class"] == 0, "dementia_risk_score"].mean()
    impaired = risk.loc[risk["binary_class"] == 1, "dementia_risk_score"]
    t_res = stats.ttest_1samp(impaired, popmean=cn_mean, alternative="greater")
    subject_summary = (
        risk.groupby(["patient_id", "binary_class"], as_index=False)["dementia_risk_score"]
        .mean()
        .rename(columns={"dementia_risk_score": "subject_mean_drs"})
    )
    subject_cn_mean = subject_summary.loc[subject_summary["binary_class"] == 0, "subject_mean_drs"].mean()
    subject_imp = subject_summary.loc[subject_summary["binary_class"] == 1, "subject_mean_drs"]
    subject_t = stats.ttest_1samp(subject_imp, popmean=subject_cn_mean, alternative="greater")

    result = {
        "params": params,
        "training_set_metrics_at_0_5": cv_like,
        "drs_row_sample": sample_note,
        "daily_drs_summary": summary.to_dict(orient="records"),
        "daily_one_sided_t_test": {
            "cn_mean": float(cn_mean),
            "impaired_mean": float(impaired.mean()),
            "t_statistic": float(t_res.statistic),
            "p_value": float(t_res.pvalue),
            "alternative": "impaired mean > CN mean",
        },
        "subject_one_sided_t_test": {
            "cn_subject_mean": float(subject_cn_mean),
            "impaired_subject_mean": float(subject_imp.mean()),
            "t_statistic": float(subject_t.statistic),
            "p_value": float(subject_t.pvalue),
            "alternative": "impaired subject mean > CN subject mean",
        },
        "top_features": importance.head(20).to_dict(orient="records"),
    }
    save_json(result, output_dir / "dementia_risk_score_summary.json")

    plot_risk_histogram(risk, output_dir / "dementia_risk_score_histogram.png")
    plot_roc(y.to_numpy(dtype=int), full_prob, output_dir / "final_training_roc.png")
    shap.summary_plot(shap_pos, Xt, feature_names=feature_cols, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_positive.png", dpi=180, bbox_inches="tight")
    plt.close()
    return result


def plot_risk_histogram(risk: pd.DataFrame, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    ensure_dir(path.parent)
    plt.figure(figsize=(9, 5))
    for label, name, color in [(0, "CN", "tab:blue"), (1, "MCI/Dem", "tab:orange")]:
        values = risk.loc[risk["binary_class"] == label, "dementia_risk_score"]
        plt.hist(values, bins=40, alpha=0.55, label=name, color=color, density=True)
    plt.xlabel("Dementia Risk Score")
    plt.ylabel("Density")
    plt.title("Dementia Risk Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    path = Path(path)
    ensure_dir(path.parent)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("LightGBM ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_model_comparison(metrics_df: pd.DataFrame, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    ensure_dir(path.parent)
    ordered = metrics_df.sort_values("roc_auc", ascending=True)
    plt.figure(figsize=(9, 5))
    plt.barh(ordered["model"], ordered["roc_auc"], color="tab:green")
    plt.xlabel("5-fold ROC-AUC")
    plt.title("Prediction Model Comparison")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def load_dataset_outputs(output_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    data_path = output_dir / "data" / "daily_binary_lifelog.csv"
    feature_path = output_dir / "data" / "feature_columns.json"
    if not data_path.exists() or not feature_path.exists():
        raise FileNotFoundError("Run 01_preprocess_daily_binary.py first.")
    df = pd.read_csv(data_path, low_memory=False)
    features = load_json(feature_path)
    return df, features
