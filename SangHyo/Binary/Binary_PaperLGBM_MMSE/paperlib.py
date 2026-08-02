"""Leakage-free reproduction of the KIIE 2025 lifelog LightGBM study.

The paper (천희웅 외, 2025) reports ROC-AUC 0.90 (tuned 0.949) with LightGBM, but
it evaluates at the **day level** with random 5-fold cross-validation over 12,183
subject-days.  Because one subject contributes 35-122 days, a random day split
puts the same subject in both training and validation folds — textbook subject
leakage that inflates ROC-AUC.  The paper itself notes it built a "하루 단위"
(day-level), not subject-level, model.

This module keeps the paper's rich feature engineering (Figure 2: intraday MET
statistics, activity-class counts, sleep-stage counts, plus daily scalars) but
evaluates **without leakage**:

* primary deliverable  : features aggregated to one row per subject, LightGBM,
  repeated subject-level Stratified K-fold OOF ROC-AUC + a frozen validation.
* leakage diagnostic   : the same day-level features scored with (a) random
  K-fold (paper-style, leaky → ~0.9) and (b) GroupKFold by subject (honest),
  to quantify exactly how much the leakage inflates ROC-AUC.

Honest expectation: leakage-free subject-level ROC-AUC is ~0.71 (the ceiling
found across every prior experiment); the 0.9 target is only reachable with the
leaky day-level split, so it is reported strictly as a diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from SangHyo.Binary.Binary_Wearable_SequenceFusion_Google import data as D

# Daily scalar features (paper Figure 2), physiological/behavioural only.  Pure
# identifiers (sleep_period_id) are dropped even though the paper listed them.
ACTIVITY_SCALARS = [
    "activity_average_met", "activity_cal_active", "activity_cal_total",
    "activity_daily_movement", "activity_high", "activity_inactive",
    "activity_inactivity_alerts", "activity_low", "activity_medium",
    "activity_met_min_high", "activity_met_min_inactive", "activity_met_min_low",
    "activity_met_min_medium", "activity_non_wear", "activity_rest",
    "activity_score", "activity_score_meet_daily_targets",
    "activity_score_move_every_hour", "activity_score_recovery_time",
    "activity_score_stay_active", "activity_score_training_frequency",
    "activity_score_training_volume", "activity_steps", "activity_total",
]
SLEEP_SCALARS = [
    "sleep_awake", "sleep_breath_average", "sleep_deep", "sleep_duration",
    "sleep_efficiency", "sleep_hr_average", "sleep_hr_lowest", "sleep_light",
    "sleep_midpoint_at_delta", "sleep_onset_latency", "sleep_rem", "sleep_restless",
    "sleep_rmssd", "sleep_score", "sleep_score_alignment", "sleep_score_deep",
    "sleep_score_disturbances", "sleep_score_efficiency", "sleep_score_latency",
    "sleep_score_rem", "sleep_score_total", "sleep_temperature_delta", "sleep_total",
]

# ---- MMSE (only used by the MMSE variant) --------------------------------- #
MMSE_DOMAINS = {
    "orient_time": ["Q01", "Q02", "Q03", "Q04", "Q05"],
    "orient_place": ["Q06", "Q07", "Q08", "Q09", "Q10"],
    "attention": ["Q12_1", "Q12_2", "Q12_3", "Q12_4", "Q12_5"],
    "recall": ["Q13_1", "Q13_2", "Q13_3"],
    "language": ["Q14_1", "Q14_2", "Q15", "Q16_1", "Q16_2", "Q16_3", "Q17", "Q18", "Q19"],
}
MMSE_KEY_ITEMS = ["Q13_2", "Q13_3", "Q12_5", "Q03", "Q09"]
MMSE_FORBIDDEN = frozenset({"DIAG_NM", "DIAG_SEQ", "DOCTOR_NM", "MMSE_NUM", "MMSE_KIND"})
_MMSE_FILE = {"train": "train_mmse.csv", "val": "val_mmse.csv"}


def _hour(value) -> float:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return np.nan
    local = ts.tz_convert(D.LOCAL_TIMEZONE)
    return float(local.hour + local.minute / 60.0)


def _array_stats(arr: np.ndarray, prefix: str) -> dict:
    finite = np.asarray(arr, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    keys = ["std", "variance", "kurtosis", "skewness", "mean", "median", "min",
            "max", "autocorrelation", "quantile_25", "quantile_50", "quantile_75"]
    out = {f"{prefix}_{k}": np.nan for k in keys}
    if len(finite) == 0:
        return out
    out[f"{prefix}_mean"] = float(np.mean(finite))
    out[f"{prefix}_median"] = float(np.median(finite))
    out[f"{prefix}_min"] = float(np.min(finite))
    out[f"{prefix}_max"] = float(np.max(finite))
    if len(finite) >= 2:
        out[f"{prefix}_std"] = float(np.std(finite))
        out[f"{prefix}_variance"] = float(np.var(finite))
        q25, q50, q75 = np.quantile(finite, [0.25, 0.5, 0.75])
        out[f"{prefix}_quantile_25"] = float(q25)
        out[f"{prefix}_quantile_50"] = float(q50)
        out[f"{prefix}_quantile_75"] = float(q75)
    if len(finite) >= 4 and np.std(finite) > 0:
        out[f"{prefix}_kurtosis"] = float(sp_stats.kurtosis(finite))
        out[f"{prefix}_skewness"] = float(sp_stats.skew(finite))
    if len(finite) >= 3 and np.std(finite[:-1]) > 0 and np.std(finite[1:]) > 0:
        out[f"{prefix}_autocorrelation"] = float(np.corrcoef(finite[:-1], finite[1:])[0, 1])
    return out


def _mmse_features(root: Path, split_key: str) -> pd.DataFrame:
    frame = D._read_csv(root / "SourceData" / "3.CognitiveFunction" / _MMSE_FILE[split_key])
    id_col = next((c for c in ("SAMPLE_EMAIL", "EMAIL") if c in frame.columns))
    bad = [c for c in frame.columns if c in MMSE_FORBIDDEN]
    frame = frame.drop(columns=bad).copy()
    frame["_subject_id"] = frame[id_col].astype(str).str.strip()
    g = frame.drop_duplicates("_subject_id").set_index("_subject_id")
    out = pd.DataFrame(index=g.index)
    out["mmse_TOTAL"] = g["TOTAL"].astype(float)
    for domain, cols in MMSE_DOMAINS.items():
        out[f"mmse_{domain}"] = g[cols].astype(float).sum(axis=1)
    for item in MMSE_KEY_ITEMS:
        out[f"mmse_{item}"] = g[item].astype(float)
    return out.reset_index()


def build_daily_frame(data_root: str | Path, split: str, include_mmse: bool) -> pd.DataFrame:
    """One row per subject-day with the paper's rich feature set."""

    split_key = D._normalise_split(split)
    root = D.resolve_split_root(data_root, split_key)
    aligned, _audit = D._prepare_sources(root, split_key)
    labels = D.load_binary_labels(root)

    rows = []
    for record in aligned.to_dict("records"):
        sid = str(record["_subject_id"])
        feats: dict[str, float] = {}
        for col in ACTIVITY_SCALARS + SLEEP_SCALARS:
            feats[col] = D._numeric(record.get(col))
        feats["sleep_bedtime_end_hour"] = _hour(record.get("sleep_bedtime_end"))
        feats["sleep_bedtime_start_hour"] = _hour(record.get("sleep_bedtime_start"))
        met = D._sequence_from_row(record, D.ACTIVITY_MET_BLOB, "activity_met_1min")
        feats.update(_array_stats(met, "activity_met_1min"))
        cls = D._sequence_from_row(record, D.ACTIVITY_CLASS_BLOB, "activity_class_5min")
        for code in (1, 2, 3, 4):
            feats[f"activity_class_5min_count_{code}"] = int(np.sum(cls == code))
        hyp = D._sequence_from_row(record, D.SLEEP_STAGE_BLOB, "sleep_hypnogram_5min")
        for code in (1, 2, 3, 4):
            feats[f"sleep_hypnogram_5min_count_{code}"] = int(np.sum(hyp == code))
        feats["_subject_id"] = sid
        feats["y"] = int(labels.loc[sid])
        rows.append(feats)

    df = pd.DataFrame(rows)
    if include_mmse:
        df = df.merge(_mmse_features(root, split_key), on="_subject_id", how="left")
    return df


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("_subject_id", "y")]


def aggregate_to_subject(df: pd.DataFrame, include_mmse: bool):
    """Aggregate day-level features to one leakage-safe row per subject."""

    wearable = [c for c in _feature_columns(df) if not c.startswith("mmse_")]
    grouped = df.groupby("_subject_id", sort=True)
    agg = grouped[wearable].agg(["mean", "std"])
    agg.columns = [f"{col}__{stat}" for col, stat in agg.columns]
    if include_mmse:
        mmse_cols = [c for c in df.columns if c.startswith("mmse_")]
        agg = agg.join(grouped[mmse_cols].first())
    subjects = list(agg.index)
    y = grouped["y"].first().reindex(subjects).to_numpy(np.int64)
    return subjects, agg.to_numpy(np.float64), list(agg.columns), y


def _lgbm(params: dict):
    import lightgbm as lgb
    return lgb.LGBMClassifier(**params)


def _subject_hp() -> dict:
    # Conservative HP for ~141 subject rows (the paper's num_leaves=330 was tuned
    # for 12k day rows and would not split meaningfully here).
    return dict(n_estimators=400, num_leaves=15, min_child_samples=8,
                learning_rate=0.03, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.7, reg_lambda=1.0, class_weight="balanced",
                random_state=0, n_jobs=-1, verbose=-1)


def _paper_hp() -> dict:
    # The paper's tuned LightGBM HP (for the day-level, leaky diagnostic only).
    return dict(n_estimators=1000, num_leaves=330, min_child_samples=41,
                learning_rate=0.08, class_weight="balanced",
                random_state=0, n_jobs=-1, verbose=-1)


def _impute_fit(x):
    med = np.nanmedian(x, axis=0)
    return np.where(np.isfinite(med), med, 0.0)


def _impute(x, med):
    return np.where(np.isfinite(x), x, med)


def subject_cv(X, y, *, repeats, folds, seed) -> dict:
    """Repeated subject-level Stratified K-fold OOF (leakage-free)."""

    n = len(y)
    prob_sum = np.zeros(n)
    seen = np.zeros(n)
    for r in range(repeats):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + r)
        for tr, te in skf.split(np.arange(n), y):
            med = _impute_fit(X[tr])
            model = _lgbm(_subject_hp()).fit(_impute(X[tr], med), y[tr])
            prob_sum[te] += model.predict_proba(_impute(X[te], med))[:, 1]
            seen[te] += 1
    prob = prob_sum / seen
    return {"oof_prob": prob, "roc_auc": float(roc_auc_score(y, prob))}


def leakage_diagnostic(df: pd.DataFrame, *, folds: int, seed: int) -> dict:
    """Day-level ROC-AUC under random K-fold (leaky) vs GroupKFold (honest)."""

    cols = _feature_columns(df)
    X = df[cols].to_numpy(np.float64)
    y = df["y"].to_numpy(np.int64)
    groups = df["_subject_id"].to_numpy()

    def day_oof(splits):
        prob = np.zeros(len(y))
        for tr, te in splits:
            med = _impute_fit(X[tr])
            model = _lgbm(_paper_hp()).fit(_impute(X[tr], med), y[tr])
            prob[te] = model.predict_proba(_impute(X[te], med))[:, 1]
        return prob

    random_splits = list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y))
    random_prob = day_oof(random_splits)
    group_splits = list(GroupKFold(folds).split(X, y, groups))
    group_prob = day_oof(group_splits)

    # subject-level AUC from the honest GroupKFold day predictions
    subj = pd.DataFrame({"g": groups, "p": group_prob, "y": y}).groupby("g").agg(
        p=("p", "mean"), y=("y", "first"))
    return {
        "day_random_kfold_roc_auc_LEAKY": float(roc_auc_score(y, random_prob)),
        "day_groupkfold_roc_auc_honest": float(roc_auc_score(y, group_prob)),
        "subject_from_groupkfold_roc_auc_honest": float(roc_auc_score(subj["y"], subj["p"])),
        "note": ("day_random_kfold is the paper-style split and is inflated by "
                 "subject leakage; the GroupKFold/subject numbers are honest."),
    }


@dataclass
class RunConfig:
    training_root: str
    validation_root: str
    output_dir: str
    include_mmse: bool
    experiment_name: str
    run_mode: str = "full"
    repeats: int = 5
    folds: int = 5
    seed: int = 20260724
    run_leakage_diagnostic: bool = True
    evaluate_validation: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _hash(s: str) -> str:
    return hashlib.sha256(str(s).encode()).hexdigest()[:16]


def run_experiment(config: RunConfig) -> dict:
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    train_daily = build_daily_frame(config.training_root, "train", config.include_mmse)
    subjects, X, feat_names, y = aggregate_to_subject(train_daily, config.include_mmse)

    eda = {
        "experiment": config.experiment_name, "include_mmse": config.include_mmse,
        "n_subjects": len(subjects), "n_subject_days": int(len(train_daily)),
        "class_counts": {"CN": int((y == 0).sum()), "MCI_DEM": int((y == 1).sum())},
        "n_features_subject_level": len(feat_names),
    }
    _write_json(output / "eda" / "eda_summary.json", eda)

    cv = subject_cv(X, y, repeats=config.repeats, folds=config.folds, seed=config.seed)
    (output / "training").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"subject_hash": [_hash(s) for s in subjects], "y_true": y,
                 "oof_prob": cv["oof_prob"]}).to_csv(
        output / "training" / "oof_predictions_hashed.csv", index=False)

    diagnostic = None
    if config.run_leakage_diagnostic:
        diagnostic = leakage_diagnostic(train_daily, folds=config.folds, seed=config.seed)

    validation = None
    if config.evaluate_validation:
        validation = _validate(config, X, y, feat_names)

    report = {
        "experiment": config.experiment_name, "run_mode": config.run_mode,
        "include_mmse": config.include_mmse,
        "started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
        "primary_metric": "leakage-free subject-level ROC-AUC",
        "target": {"roc_auc": 0.90, "accuracy": 0.80},
        "leakage_free_subject_oof_roc_auc": cv["roc_auc"],
        "leakage_diagnostic": diagnostic,
        "validation": validation,
        "honesty_note": (
            "The 0.90 target is only reachable with the paper's day-level random "
            "K-fold, which has subject leakage (see leakage_diagnostic). The honest, "
            "leakage-free subject-level ROC-AUC is the primary metric above."),
    }
    _write_json(output / "training" / "FINAL_REPORT.json", report)
    _write_json(output / "training" / "TRAINING_COMPLETE.json",
                {"status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat()})
    return report


def _validate(config, X_train, y_train, feat_names) -> dict:
    val_daily = build_daily_frame(config.validation_root, "val", config.include_mmse)
    v_subjects, X_val, v_names, _ = aggregate_to_subject(val_daily, config.include_mmse)
    if v_names != feat_names:
        # align columns by name (validation may miss a feature); reindex
        vidx = {n: i for i, n in enumerate(v_names)}
        X_val = np.column_stack([X_val[:, vidx[n]] if n in vidx else np.full(len(v_subjects), np.nan)
                                 for n in feat_names])
    med = _impute_fit(X_train)
    model = _lgbm(_subject_hp()).fit(_impute(X_train, med), y_train)
    prob = model.predict_proba(_impute(X_val, med))[:, 1]

    y_val = D.load_binary_labels(D.resolve_split_root(config.validation_root, "val"))
    y_val = y_val.loc[[str(s) for s in v_subjects]].to_numpy(np.int64)
    auc = float(roc_auc_score(y_val, prob))
    # accuracy at 0.5 and at the balanced-accuracy-optimal threshold on train would
    # need a threshold; report 0.5 and the all-CN baseline for context.
    pred = (prob >= 0.5).astype(int)
    acc = float(np.mean(pred == y_val))
    return {"n_subjects": len(v_subjects), "roc_auc": auc, "accuracy_threshold_0.5": acc,
            "all_cn_accuracy": float(np.mean(y_val == 0)),
            "historical_benchmark_note": "33-subject reused benchmark, not a fresh test."}


__all__ = ["RunConfig", "run_experiment", "build_daily_frame", "aggregate_to_subject"]
