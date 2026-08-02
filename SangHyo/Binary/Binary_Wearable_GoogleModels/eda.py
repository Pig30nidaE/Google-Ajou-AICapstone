"""Training-only EDA for wearable CN vs MCI/DEM classification.

The EDA reports raw collection coverage, duplicate events, and observation
periods, but those protocol variables are never joined to the model feature
table.  Validation sources and Validation labels are deliberately out of scope.
Only aggregate outputs are persisted; subject identifiers are not written.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # Package import in tests / notebooks.
    from .data import (
        BinaryDataset,
        CLASS_DISPLAY_NAMES,
        CLASS_NAMES,
        assert_official_split_contract,
        assert_wearable_feature_contract,
        build_binary_dataset,
        discover_wearable_split_files,
        feature_family,
        load_consistent_label_copies,
    )
except ImportError:  # Direct ``python eda.py`` execution.
    from SangHyo.Binary.Binary_Wearable_GoogleModels.data import (  # type: ignore
        BinaryDataset,
        CLASS_DISPLAY_NAMES,
        CLASS_NAMES,
        assert_official_split_contract,
        assert_wearable_feature_contract,
        build_binary_dataset,
        discover_wearable_split_files,
        feature_family,
        load_consistent_label_copies,
    )


def _jsonable(value: Any) -> Any:
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.ndarray, pd.Index, pd.Series)):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _assert_training_only_path(path: Path, *, role: str) -> Path:
    resolved = path.expanduser().resolve()
    if any("validation" in part.lower() for part in resolved.parts):
        raise ValueError(f"Training-only EDA rejected a Validation {role}: {resolved}")
    return resolved


def _schema(frame: pd.DataFrame) -> dict[str, Any]:
    duplicate_columns = frame.columns[frame.columns.duplicated()].astype(str).tolist()
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "column_names": [str(column) for column in frame.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "duplicate_column_names": duplicate_columns,
    }


def _normalize_source_ids(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    if "EMAIL" not in frame.columns:
        raise ValueError(f"{source} source is missing EMAIL")
    result = frame.copy()
    result["EMAIL"] = result["EMAIL"].astype("string").str.strip()
    if result["EMAIL"].isna().any() or result["EMAIL"].eq("").any():
        raise ValueError(f"{source} source has an empty subject identifier")
    return result


def _kst_timestamp(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_convert(
        "Asia/Seoul"
    )


def _quantile_summary(values: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    numeric = numeric.dropna()
    if numeric.empty:
        return {
            "n": 0,
            "min": float("nan"),
            "q25": float("nan"),
            "median": float("nan"),
            "q75": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": int(len(numeric)),
        "min": float(numeric.min()),
        "q25": float(numeric.quantile(0.25)),
        "median": float(numeric.median()),
        "q75": float(numeric.quantile(0.75)),
        "max": float(numeric.max()),
    }


def _date_audit(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    modality: str,
) -> tuple[pd.DataFrame, dict[str, Any], set[tuple[str, object]]]:
    if timestamp_column not in frame.columns:
        raise ValueError(f"{modality} source is missing {timestamp_column}")
    timestamps = _kst_timestamp(frame[timestamp_column])
    event_dates = timestamps.dt.date
    keys = pd.DataFrame({"subject_id": frame["EMAIL"], "event_date": event_dates})
    valid_keys = keys.dropna(subset=["event_date"])
    key_counts = valid_keys.groupby(
        ["subject_id", "event_date"], sort=False, observed=True
    ).size()
    duplicate_groups = key_counts[key_counts > 1]

    per_subject = (
        pd.DataFrame(
            {
                "subject_id": frame["EMAIL"],
                f"{modality}_timestamp": timestamps,
                f"{modality}_date": event_dates,
            }
        )
        .groupby("subject_id", sort=True)
        .agg(
            **{
                f"{modality}_rows": (f"{modality}_date", "size"),
                f"{modality}_days": (f"{modality}_date", "nunique"),
                f"{modality}_first": (f"{modality}_timestamp", "min"),
                f"{modality}_last": (f"{modality}_timestamp", "max"),
            }
        )
    )
    span = (
        per_subject[f"{modality}_last"].dt.normalize()
        - per_subject[f"{modality}_first"].dt.normalize()
    ).dt.days
    per_subject[f"{modality}_span_days"] = span + 1

    finite_timestamps = timestamps.dropna()
    audit = {
        "timestamp_column": timestamp_column,
        "rows": int(len(frame)),
        "subjects": int(frame["EMAIL"].nunique()),
        "timestamp_parse_failures": int(timestamps.isna().sum()),
        "date_min": None
        if finite_timestamps.empty
        else finite_timestamps.min().isoformat(),
        "date_max": None
        if finite_timestamps.empty
        else finite_timestamps.max().isoformat(),
        "duplicate_subject_date_groups": int(len(duplicate_groups)),
        "duplicate_subject_date_extra_rows": int((duplicate_groups - 1).sum()),
        "reported_for_audit_only_not_model_features": True,
    }
    key_set = set(
        zip(
            valid_keys["subject_id"].astype(str),
            valid_keys["event_date"],
        )
    )
    return per_subject, audit, key_set


def _coverage_by_class(
    coverage: pd.DataFrame,
    binary_target: pd.Series,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    target = binary_target.reindex(coverage.index)
    for class_id, class_name in enumerate(CLASS_NAMES):
        selected = coverage.loc[target == class_id]
        result[class_name] = {
            column: _quantile_summary(selected[column])
            for column in (
                "activity_rows",
                "activity_days",
                "activity_span_days",
                "sleep_rows",
                "sleep_days",
                "sleep_span_days",
            )
        }
    return result


def _training_raw_audit(
    training_root: Path,
) -> tuple[dict[str, Any], pd.Series]:
    files = discover_wearable_split_files(training_root, require_labels=True)
    source_paths = [files.activity, files.sleep, *files.labels]
    for source_path in source_paths:
        checked = _assert_training_only_path(Path(source_path), role="source file")
        try:
            checked.relative_to(training_root)
        except ValueError as exc:
            raise ValueError(f"Training source escaped its root: {checked}") from exc

    # All diagnosis copies are Training labels.  The audited loader normalizes
    # label aliases and raises immediately on any subject/target disagreement.
    labels = load_consistent_label_copies(files.labels)
    copies = [load_consistent_label_copies([path]) for path in files.labels]
    copies_equal = all(
        labels.index.equals(copy.index) and labels.equals(copy) for copy in copies
    )
    if not copies_equal:
        raise AssertionError("Training diagnosis-label copies differ")

    activity = _normalize_source_ids(
        pd.read_csv(files.activity, low_memory=False), source="activity"
    )
    sleep = _normalize_source_ids(
        pd.read_csv(files.sleep, low_memory=False), source="sleep"
    )
    activity_subjects = set(activity["EMAIL"].astype(str))
    sleep_subjects = set(sleep["EMAIL"].astype(str))
    label_subjects = set(labels.index.astype(str))
    if not (activity_subjects == sleep_subjects == label_subjects):
        raise AssertionError("Training wearable and label subject sets differ")

    activity_coverage, activity_audit, activity_keys = _date_audit(
        activity,
        timestamp_column="activity_day_start",
        modality="activity",
    )
    sleep_coverage, sleep_audit, sleep_keys = _date_audit(
        sleep,
        timestamp_column="sleep_bedtime_end",
        modality="sleep",
    )
    coverage = activity_coverage.join(sleep_coverage, how="outer")
    binary_target = labels.map(lambda value: 0 if value == "CN" else 1).astype(
        np.int64
    )

    label_schemas = []
    for path in files.labels:
        header = pd.read_csv(path, nrows=0)
        label_schemas.append(
            {
                "file": str(path),
                "column_names": [str(column) for column in header.columns],
                "columns": int(len(header.columns)),
            }
        )
    original_counts = {
        name: int((labels == name).sum()) for name in ("CN", "MCI", "DEM")
    }
    binary_counts = {
        class_name: int((binary_target == class_id).sum())
        for class_id, class_name in enumerate(CLASS_NAMES)
    }

    sleep_period_duplicates: int | None = None
    if "sleep_period_id" in sleep.columns:
        period_values = sleep["sleep_period_id"].astype("string")
        period_valid = period_values.notna() & period_values.str.strip().ne("")
        sleep_period_duplicates = int(
            sleep.loc[period_valid]
            .assign(_period=period_values.loc[period_valid].str.strip())
            .duplicated(["EMAIL", "_period"])
            .sum()
        )

    audit = {
        "training_root": str(training_root),
        "subjects": int(len(labels)),
        "original_diagnosis_counts": original_counts,
        "binary_class_counts": binary_counts,
        "raw_schema": {
            "activity": _schema(activity),
            "sleep": _schema(sleep),
            "label_copies": label_schemas,
        },
        "label_copy_consistency": {
            "copies_checked": int(len(files.labels)),
            "same_subject_order_and_normalized_labels": bool(copies_equal),
            "schemas_identical": len(
                {tuple(item["column_names"]) for item in label_schemas}
            )
            == 1,
        },
        "source_subject_sets": {
            "activity_equals_sleep_equals_labels": True,
            "identifiers_persisted": False,
        },
        "activity_collection": activity_audit,
        "sleep_collection": {
            **sleep_audit,
            "duplicate_subject_period_id_extra_rows": sleep_period_duplicates,
        },
        "cross_modality_subject_dates": {
            "matched": int(len(activity_keys & sleep_keys)),
            "activity_only": int(len(activity_keys - sleep_keys)),
            "sleep_only": int(len(sleep_keys - activity_keys)),
            "reported_for_audit_only_not_model_features": True,
        },
        "coverage_by_binary_class": _coverage_by_class(coverage, binary_target),
        "collection_coverage_joined_to_model_features": False,
        "duplicate_or_period_fields_joined_to_model_features": False,
        "mmse_source_resolved": False,
        "mmse_source_opened": False,
        "mmse_source_used": False,
        "validation_source_opened": False,
        "validation_label_opened": False,
    }
    return audit, labels


def _feature_quality(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    numeric = X.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    rows: list[dict[str, Any]] = []
    for feature in numeric.columns:
        series = numeric[feature]
        finite_values = series.dropna()
        rows.append(
            {
                "feature": str(feature),
                "family": feature_family(str(feature)),
                "missing_fraction": float(series.isna().mean()),
                "missing_fraction_cn": float(series.loc[y == 0].isna().mean()),
                "missing_fraction_mci_dem": float(
                    series.loc[y == 1].isna().mean()
                ),
                "finite_count": int(series.notna().sum()),
                "unique_nonmissing": int(series.nunique(dropna=True)),
                "all_missing": bool(finite_values.empty),
                "zero_variance": bool(series.nunique(dropna=True) <= 1),
                "q01": float(finite_values.quantile(0.01))
                if not finite_values.empty
                else float("nan"),
                "median": float(finite_values.median())
                if not finite_values.empty
                else float("nan"),
                "q99": float(finite_values.quantile(0.99))
                if not finite_values.empty
                else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["missing_fraction", "zero_variance", "feature"],
        ascending=[False, False, True],
        kind="mergesort",
    )


def _auc_and_delta(
    cn_values: np.ndarray,
    impaired_values: np.ndarray,
) -> tuple[float, float, float]:
    cn = np.asarray(cn_values, dtype=float)
    impaired = np.asarray(impaired_values, dtype=float)
    cn = cn[np.isfinite(cn)]
    impaired = impaired[np.isfinite(impaired)]
    if not len(cn) or not len(impaired):
        return float("nan"), float("nan"), float("nan")
    combined = np.concatenate([cn, impaired])
    ranks = pd.Series(combined).rank(method="average").to_numpy(dtype=float)
    n_cn = len(cn)
    n_impaired = len(impaired)
    positive_rank_sum = float(ranks[n_cn:].sum())
    auc = (
        positive_rank_sum - n_impaired * (n_impaired + 1) / 2.0
    ) / (n_impaired * n_cn)
    auc = float(np.clip(auc, 0.0, 1.0))
    direction_free_auc = float(max(auc, 1.0 - auc))
    # With average tie ranks, 2*AUC-1 equals (greater-lesser)/(n1*n0).
    cliffs_delta = float(2.0 * auc - 1.0)
    return auc, direction_free_auc, cliffs_delta


def _univariate_effects(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in X.columns:
        values = pd.to_numeric(X[feature], errors="coerce").to_numpy(dtype=float)
        cn = values[y == 0]
        impaired = values[y == 1]
        auc, direction_free_auc, delta = _auc_and_delta(cn, impaired)
        finite_cn = cn[np.isfinite(cn)]
        finite_impaired = impaired[np.isfinite(impaired)]
        rows.append(
            {
                "feature": str(feature),
                "family": feature_family(str(feature)),
                "n_cn": int(len(finite_cn)),
                "n_mci_dem": int(len(finite_impaired)),
                "cn_median": float(np.median(finite_cn))
                if len(finite_cn)
                else float("nan"),
                "mci_dem_median": float(np.median(finite_impaired))
                if len(finite_impaired)
                else float("nan"),
                "auc_mci_dem_positive": auc,
                "direction_free_univariate_auc": direction_free_auc,
                "cliffs_delta_mci_dem_minus_cn": delta,
                "abs_cliffs_delta": abs(delta) if np.isfinite(delta) else float("nan"),
                "higher_values_in": "MCI_DEM"
                if np.isfinite(delta) and delta > 0
                else "CN"
                if np.isfinite(delta) and delta < 0
                else "TIE_OR_UNDEFINED",
                "missing_fraction": float(np.mean(~np.isfinite(values))),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["direction_free_univariate_auc", "abs_cliffs_delta", "feature"],
        ascending=[False, False, True],
        kind="mergesort",
        na_position="last",
    )


def _plot_class_counts(y: np.ndarray, output: Path) -> None:
    counts = [int(np.sum(y == class_id)) for class_id in range(2)]
    labels = [CLASS_DISPLAY_NAMES[name] for name in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(labels, counts, color=["#4C78A8", "#E45756"])
    ax.set_title("Training subjects: binary target")
    ax.set_ylabel("Subjects")
    ax.set_ylim(0, max(counts) * 1.16)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count + 1,
            str(count),
            ha="center",
        )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_top_effects(effects: pd.DataFrame, output: Path, *, limit: int = 20) -> None:
    shown = effects.dropna(
        subset=["cliffs_delta_mci_dem_minus_cn"]
    ).head(limit).copy()
    shown = shown.sort_values("cliffs_delta_mci_dem_minus_cn")
    labels = (
        shown["feature"]
        .str.replace("__", " / ", regex=False)
        .str.slice(0, 84)
        .tolist()
    )
    colors = np.where(
        shown["cliffs_delta_mci_dem_minus_cn"] >= 0, "#E45756", "#4C78A8"
    )
    fig_height = max(5.5, 0.36 * max(1, len(shown)))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    positions = np.arange(len(shown))
    ax.barh(
        positions,
        shown["cliffs_delta_mci_dem_minus_cn"],
        color=colors,
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Cliff's delta (MCI + DEM minus CN)")
    ax.set_title("Largest direction-free univariate effects (Training only)")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_table(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    limit: int,
) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for record in frame.head(limit)[columns].to_dict(orient="records"):
        values = []
        for column in columns:
            value = record[column]
            if isinstance(value, (float, np.floating)):
                rendered = "" if not np.isfinite(value) else f"{value:.3f}"
            else:
                rendered = str(value)
            values.append(rendered.replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _coverage_row(raw_audit: dict[str, Any], class_name: str) -> str:
    coverage = raw_audit["coverage_by_binary_class"][class_name]
    activity = coverage["activity_days"]
    sleep = coverage["sleep_days"]
    return (
        f"| {CLASS_DISPLAY_NAMES[class_name]} | "
        f"{activity['median']:.0f} [{activity['q25']:.0f}–{activity['q75']:.0f}] | "
        f"{sleep['median']:.0f} [{sleep['q25']:.0f}–{sleep['q75']:.0f}] |"
    )


def _build_report(
    raw_audit: dict[str, Any],
    dataset: BinaryDataset,
    quality: pd.DataFrame,
    effects: pd.DataFrame,
) -> str:
    activity = raw_audit["activity_collection"]
    sleep = raw_audit["sleep_collection"]
    cross = raw_audit["cross_modality_subject_dates"]
    original = raw_audit["original_diagnosis_counts"]
    binary = raw_audit["binary_class_counts"]
    usable = int((~quality["all_missing"] & ~quality["zero_variance"]).sum())
    high_missing = int((quality["missing_fraction"] > 0.35).sum())
    top = effects.copy()
    top["feature"] = top["feature"].str.replace("__", " → ", regex=False)
    lines = [
        "# CN vs MCI + DEM — Training-only EDA",
        "",
        "이 보고서는 **Training 데이터만** 사용했습니다. Validation 원천 파일과 라벨은 열지 않았고, MMSE 원천은 경로를 찾거나 읽지 않았습니다.",
        "",
        "## 핵심 요약",
        "",
        f"- 총 {raw_audit['subjects']}명: CN {binary['CN']}명, MCI + DEM {binary['MCI_DEM']}명입니다. 원래 진단은 CN {original['CN']}명, MCI {original['MCI']}명, DEM {original['DEM']}명입니다.",
        f"- Activity {activity['rows']:,}행과 Sleep {sleep['rows']:,}행에서 7/14/28개 최근 관측 이벤트 요약 {dataset.X.shape[1]:,}개를 만들었습니다.",
        f"- 라벨 사본 {raw_audit['label_copy_consistency']['copies_checked']}개는 subject와 정규화된 진단이 모두 같습니다.",
        f"- 유효·비상수 특징은 {usable:,}개이고, 결측률 35% 초과 특징은 {high_missing:,}개입니다. 결측 대치·스케일링·특징 선택은 EDA가 아니라 각 학습 fold 안에서만 수행해야 합니다.",
        "- 아래 AUC와 효과크기는 방향을 고르기 위한 Training 연관성일 뿐이며, 이 순위를 전체 Training에서 고정해 학습에 넘기지 않습니다.",
        "",
        "## 원시 스키마와 수집 감사",
        "",
        f"- Activity 스키마: {raw_audit['raw_schema']['activity']['columns']}열, 기간 {activity['date_min']} ~ {activity['date_max']}",
        f"- Sleep 스키마: {raw_audit['raw_schema']['sleep']['columns']}열, 기간 {sleep['date_min']} ~ {sleep['date_max']}",
        f"- subject-date 중복 추가 행: Activity {activity['duplicate_subject_date_extra_rows']:,}, Sleep {sleep['duplicate_subject_date_extra_rows']:,}",
        f"- 같은 subject-date가 양쪽 modality에 모두 있는 경우 {cross['matched']:,}건, Activity만 {cross['activity_only']:,}건, Sleep만 {cross['sleep_only']:,}건입니다.",
        "",
        "수집량·중복·관측기간은 데이터 품질을 이해하기 위해서만 보고합니다. 진단군마다 장비 착용 또는 수집 프로토콜이 다를 수 있으므로 모델 특징에는 넣지 않았습니다.",
        "",
        "| class | Activity 관측일 중앙값 [Q1–Q3] | Sleep 관측일 중앙값 [Q1–Q3] |",
        "| --- | ---: | ---: |",
        _coverage_row(raw_audit, "CN"),
        _coverage_row(raw_audit, "MCI_DEM"),
        "",
        "## 단변량 CN 분리 신호",
        "",
        "`direction_free_univariate_auc`는 값이 커지는 방향과 작아지는 방향 중 더 큰 AUC입니다. Cliff's delta는 양수면 MCI + DEM 쪽 값이 더 크고, 음수면 CN 쪽이 더 큽니다.",
        "",
        _markdown_table(
            top,
            [
                "feature",
                "family",
                "direction_free_univariate_auc",
                "cliffs_delta_mci_dem_minus_cn",
                "cn_median",
                "mci_dem_median",
            ],
            limit=15,
        ),
        "",
        "## 전처리 및 검증 원칙",
        "",
        "1. 사람 단위 한 행을 유지해 같은 사람의 이벤트가 서로 다른 fold로 새지 않게 합니다.",
        "2. Activity와 Sleep의 최근 7/14/28 관측 이벤트만 사용하며, 마지막 Activity 예측시점 이후의 Sleep은 제외합니다.",
        "3. ID, 진단, MMSE, 절대 날짜, 관측일 수, calendar gap, mask, non-wear 및 period ID는 모델 특징으로 금지합니다.",
        "4. 결측 대치, clipping, scaling, feature selection은 교차검증의 Training fold에만 적합합니다.",
        "5. Accuracy 0.90 목표는 내부 OOF와 별도의 최종 Validation으로 확인하되, class imbalance 때문에 ROC-AUC·balanced accuracy·민감도도 함께 봅니다.",
        "",
        "## 산출물",
        "",
        "- `feature_quality.csv`: 특징별 결측률·고유값·분위수",
        "- `top_effects.csv`: 방향무관 단변량 AUC와 Cliff's delta 전체 순위",
        "- `class_counts.png`, `top_effects.png`: class 분포와 상위 연관성",
        "- `eda_audit.json`: 원시 스키마, 라벨 일치, coverage/중복/기간, 누수 방지 감사",
        "",
        "이 결과는 임상 진단 기준이나 인과관계를 뜻하지 않습니다.",
    ]
    return "\n".join(lines) + "\n"


def run_eda(
    training_root: Path,
    output_dir: Path,
    dataset: BinaryDataset | None = None,
) -> dict[str, Any]:
    """Run aggregate EDA using Training only and return its JSON audit."""

    root = _assert_training_only_path(Path(training_root), role="root")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    raw_audit, labels = _training_raw_audit(root)
    if dataset is None:
        dataset = build_binary_dataset(root, require_labels=True)
    dataset_root_value = dataset.audit.get("split_root")
    if dataset_root_value is None:
        raise AssertionError("Supplied dataset has no auditable split_root")
    dataset_root = _assert_training_only_path(
        Path(dataset_root_value), role="dataset root"
    )
    if dataset_root != root:
        raise AssertionError("Supplied dataset was not built from training_root")
    if dataset.y is None:
        raise AssertionError("Training-only EDA requires Training labels")
    y = np.asarray(dataset.y, dtype=np.int64)
    if set(np.unique(y).tolist()) != {0, 1}:
        raise AssertionError("Training EDA requires both binary classes {0, 1}")
    if len(dataset.subject_ids) != len(dataset.X) or len(dataset.X) != len(y):
        raise AssertionError("Training dataset rows, IDs, and labels are misaligned")
    assert_wearable_feature_contract(dataset.X.columns)
    official_contract = assert_official_split_contract(dataset, "training")

    subject_ids = pd.Index(dataset.subject_ids.astype(str), name="subject_id")
    aligned_labels = labels.reindex(subject_ids)
    if aligned_labels.isna().any():
        raise AssertionError("Engineered subject is absent from Training labels")
    expected_y = (aligned_labels.to_numpy(dtype=object) != "CN").astype(np.int64)
    if not np.array_equal(y, expected_y):
        raise AssertionError("Binary target is not aligned with subject_ids")

    quality = _feature_quality(dataset.X, y)
    effects = _univariate_effects(dataset.X, y)
    quality.to_csv(output / "feature_quality.csv", index=False)
    effects.to_csv(output / "top_effects.csv", index=False)
    _plot_class_counts(y, output / "class_counts.png")
    _plot_top_effects(effects, output / "top_effects.png")

    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Training only; aggregate EDA; no model fitting",
        "official_training_contract": official_contract,
        "raw_training": raw_audit,
        "engineered_dataset": dataset.audit,
        "feature_quality_summary": {
            "features": int(len(quality)),
            "all_missing": int(quality["all_missing"].sum()),
            "zero_variance": int(quality["zero_variance"].sum()),
            "missing_fraction_over_0_35": int(
                (quality["missing_fraction"] > 0.35).sum()
            ),
            "maximum_missing_fraction": float(quality["missing_fraction"].max()),
        },
        "univariate_effect_summary": {
            "features_ranked": int(len(effects)),
            "direction_free_auc_at_least_0_70": int(
                (effects["direction_free_univariate_auc"] >= 0.70).sum()
            ),
            "absolute_cliffs_delta_at_least_0_33": int(
                (effects["abs_cliffs_delta"] >= 0.33).sum()
            ),
            "ranking_consumed_by_training_code": False,
        },
        "privacy": {
            "subject_level_rows_persisted": False,
            "raw_or_hashed_subject_identifiers_persisted": False,
            "aggregate_outputs_only": True,
        },
        "leakage_guards": {
            "validation_source_opened": False,
            "validation_label_opened": False,
            "mmse_source_resolved": False,
            "mmse_source_opened": False,
            "mmse_source_used": False,
            "coverage_or_period_audit_used_as_model_features": False,
        },
        "outputs": [
            "feature_quality.csv",
            "top_effects.csv",
            "class_counts.png",
            "top_effects.png",
            "EDA_REPORT_KO.md",
            "eda_audit.json",
        ],
    }
    report = _build_report(raw_audit, dataset, quality, effects)
    (output / "EDA_REPORT_KO.md").write_text(report, encoding="utf-8")
    _write_json(output / "eda_audit.json", audit)
    return audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_eda(args.training_root, args.output_dir)

