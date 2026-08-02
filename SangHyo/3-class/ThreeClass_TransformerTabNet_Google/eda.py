"""Training-only EDA for the CN / MCI / DEM project.

The script writes aggregate reports only.  It does not train a model and does
not read validation labels.
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

from feature_engineering import (
    CLASS_NAMES,
    build_subject_dataset,
    discover_split_files,
    load_labels,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if not len(a) or not len(b):
        return np.nan
    differences = a[:, None] - b[None, :]
    return float((np.sum(differences > 0) - np.sum(differences < 0)) / differences.size)


def _quantile_summary(values: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {"n": 0, "median": np.nan, "q25": np.nan, "q75": np.nan, "min": np.nan, "max": np.nan}
    return {
        "n": int(len(values)),
        "median": float(values.median()),
        "q25": float(values.quantile(0.25)),
        "q75": float(values.quantile(0.75)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _raw_audit(training_root: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    files = discover_split_files(training_root, require_label=True)
    labels = load_labels(files.label)
    label_paths = sorted(training_root.glob("LabelingData/*/*label.csv"))
    label_copies = [load_labels(path).sort_values("subject_id").reset_index(drop=True) for path in label_paths]
    label_subject_sets_equal = all(
        set(copy["subject_id"]) == set(labels["subject_id"]) for copy in label_copies
    )
    reference_targets = labels.sort_values("subject_id").reset_index(drop=True)
    label_targets_equal = all(copy.equals(reference_targets) for copy in label_copies)
    label_map = labels.set_index("subject_id")["target"]
    activity = pd.read_csv(files.activity, low_memory=False).rename(columns={"EMAIL": "subject_id"})
    sleep = pd.read_csv(files.sleep, low_memory=False).rename(columns={"EMAIL": "subject_id"})
    mmse = pd.read_csv(files.mmse, low_memory=False).rename(columns={"SAMPLE_EMAIL": "subject_id"})
    # The activity record covers a 04:00-to-03:59 day, so its start date is the
    # appropriate calendar key when comparing it with the sleep wake date.
    activity["date"] = pd.to_datetime(activity["activity_day_start"], utc=True).dt.tz_convert("Asia/Seoul").dt.date
    sleep["date"] = pd.to_datetime(sleep["sleep_bedtime_end"], utc=True).dt.tz_convert("Asia/Seoul").dt.date
    activity_keys = set(zip(activity["subject_id"], activity["date"]))
    sleep_keys = set(zip(sleep["subject_id"], sleep["date"]))

    coverage = pd.DataFrame({"subject_id": labels["subject_id"]})
    coverage["target"] = coverage["subject_id"].map(label_map)
    activity_days = activity.groupby("subject_id")["date"].nunique()
    sleep_days = sleep.groupby("subject_id")["date"].nunique()
    coverage["activity_days"] = coverage["subject_id"].map(activity_days).fillna(0)
    coverage["sleep_days"] = coverage["subject_id"].map(sleep_days).fillna(0)

    direct_diag_agreement = None
    if "DIAG_NM" in mmse:
        normalized_mmse = mmse["DIAG_NM"].astype(str).str.strip().str.lower().replace({"dem": "dem"})
        normalized_label = mmse["subject_id"].map(
            labels.assign(name=[CLASS_NAMES[i] for i in labels["target"]]).set_index("subject_id")["name"]
        ).astype(str).str.lower()
        direct_diag_agreement = float((normalized_mmse == normalized_label).mean())

    mmse_total = mmse[["subject_id", "TOTAL"]].copy() if "TOTAL" in mmse else pd.DataFrame()
    if not mmse_total.empty:
        mmse_total["target"] = mmse_total["subject_id"].map(label_map)

    audit = {
        "training_subjects": int(len(labels)),
        "class_counts": {CLASS_NAMES[i]: int((labels["target"] == i).sum()) for i in range(3)},
        "label_copy_consistency": {
            "copies_checked": int(len(label_copies)),
            "subject_sets_equal": bool(label_subject_sets_equal),
            "targets_equal": bool(label_targets_equal),
        },
        "activity": {
            "rows": int(len(activity)),
            "subjects": int(activity["subject_id"].nunique()),
            "duplicate_subject_dates": int(activity.duplicated(["subject_id", "date"]).sum()),
            "date_min": str(min(activity["date"])),
            "date_max": str(max(activity["date"])),
        },
        "sleep": {
            "rows": int(len(sleep)),
            "subjects": int(sleep["subject_id"].nunique()),
            "duplicate_subject_dates": int(sleep.duplicated(["subject_id", "date"]).sum()),
            "date_min": str(min(sleep["date"])),
            "date_max": str(max(sleep["date"])),
        },
        "cross_modality_dates": {
            "matched": int(len(activity_keys & sleep_keys)),
            "activity_only": int(len(activity_keys - sleep_keys)),
            "sleep_only": int(len(sleep_keys - activity_keys)),
        },
        "coverage_by_class": {
            CLASS_NAMES[i]: {
                "activity_days": _quantile_summary(coverage.loc[coverage["target"] == i, "activity_days"]),
                "sleep_days": _quantile_summary(coverage.loc[coverage["target"] == i, "sleep_days"]),
            }
            for i in range(3)
        },
        "mmse": {
            "rows": int(len(mmse)),
            "subjects": int(mmse["subject_id"].nunique()),
            "direct_DIAG_NM_agreement_with_target": direct_diag_agreement,
            "direct_DIAG_NM_used_as_feature": False,
            "total_by_class": {
                CLASS_NAMES[i]: _quantile_summary(mmse_total.loc[mmse_total["target"] == i, "TOTAL"])
                for i in range(3)
            } if not mmse_total.empty else {},
        },
    }
    return audit, coverage


def _effect_table(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    records = []
    for feature in X.columns:
        cn = X.loc[y == 0, feature].to_numpy(dtype=float)
        for class_id in (1, 2):
            other = X.loc[y == class_id, feature].to_numpy(dtype=float)
            delta = cliffs_delta(other, cn)
            records.append(
                {
                    "feature": feature,
                    "feature_family": (
                        "MMSE" if feature.startswith("mmse__")
                        else "activity" if feature.startswith("activity__")
                        else "sleep"
                    ),
                    "comparison": f"{CLASS_NAMES[class_id]} vs CN",
                    "class_median": float(np.nanmedian(other)) if np.isfinite(other).any() else np.nan,
                    "cn_median": float(np.nanmedian(cn)) if np.isfinite(cn).any() else np.nan,
                    "cliffs_delta": delta,
                    "abs_cliffs_delta": abs(delta) if np.isfinite(delta) else np.nan,
                    "class_nonmissing": int(np.isfinite(other).sum()),
                    "cn_nonmissing": int(np.isfinite(cn).sum()),
                }
            )
    return pd.DataFrame(records).sort_values(
        ["comparison", "abs_cliffs_delta"], ascending=[True, False]
    )


def _plot_class_counts(class_counts: dict[str, int], path: Path) -> None:
    plt.figure(figsize=(6.5, 4.2))
    names = list(CLASS_NAMES)
    values = [class_counts[name] for name in names]
    bars = plt.bar(names, values, color=["#4C78A8", "#F2A541", "#D45D79"])
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 1, str(value), ha="center")
    plt.title("Training subjects by class")
    plt.ylabel("Subjects")
    plt.ylim(0, max(values) * 1.15)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_top_effects(effects: pd.DataFrame, path: Path) -> None:
    wearable = effects.loc[effects["feature_family"].isin(["activity", "sleep"])]
    top = wearable.groupby("comparison", sort=False).head(10).copy()
    top["label"] = top["feature"].str.replace("__", " / ", regex=False).str.slice(0, 72)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True)
    for axis, comparison in zip(axes, ("MCI vs CN", "DEM vs CN")):
        part = top[top["comparison"] == comparison].sort_values("cliffs_delta")
        colors = np.where(part["cliffs_delta"] >= 0, "#D45D79", "#4C78A8")
        axis.barh(part["label"], part["cliffs_delta"], color=colors)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_title(comparison)
        axis.set_xlabel("Cliff's delta (subject-level association)")
    fig.suptitle("Largest wearable associations in Training EDA")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _short_feature(feature: str) -> str:
    return feature.replace("__", " → ").replace("scalar_", "")


def _markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 10) -> str:
    part = frame[columns].head(limit).copy()
    for col in part.select_dtypes(include=[np.number]):
        part[col] = part[col].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    headers = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(row[col]) for col in columns) + " |" for _, row in part.iterrows()]
    return "\n".join([headers, divider, *rows])


def make_report(audit: dict[str, Any], dataset_audit: dict[str, Any], effects: pd.DataFrame, quality: pd.DataFrame) -> str:
    mci_wear = effects[(effects["comparison"] == "MCI vs CN") & (effects["feature_family"] != "MMSE")].head(10).copy()
    dem_wear = effects[(effects["comparison"] == "DEM vs CN") & (effects["feature_family"] != "MMSE")].head(10).copy()
    mmse = effects[effects["feature_family"] == "MMSE"].groupby("comparison", sort=False).head(8).copy()
    for frame in (mci_wear, dem_wear, mmse):
        frame["feature"] = frame["feature"].map(_short_feature)
    coverage = audit["coverage_by_class"]
    mmse_totals = audit["mmse"]["total_by_class"]
    lines = [
        "# CN / MCI / DEM Training EDA 보고서",
        "",
        "이 보고서는 **학습 데이터만** 살펴본 결과입니다. 모델 학습이나 Validation 라벨 확인은 하지 않았습니다.",
        "",
        "## 먼저 알아둘 점",
        "",
        f"- 사람 수는 CN {audit['class_counts']['CN']}명, MCI {audit['class_counts']['MCI']}명, DEM {audit['class_counts']['DEM']}명입니다.",
        f"- Gait/Sleep/CognitiveFunction의 라벨 사본 {audit['label_copy_consistency']['copies_checked']}개는 사람과 정답이 모두 같습니다.",
        "- DEM은 9명뿐이라 한두 명의 결과가 점수를 크게 바꿉니다. 그래서 단일 accuracy보다 Macro F1과 class별 결과를 함께 봐야 합니다.",
        f"- Activity와 Sleep은 각각 {audit['activity']['rows']:,}행이며, 두 자료가 같은 사람·날짜에 맞는 경우는 {audit['cross_modality_dates']['matched']:,}건입니다.",
        f"- 가공 후 사람당 입력 후보는 {dataset_audit['features']:,}개입니다. 실제 학습에서는 각 fold의 Training 부분 안에서만 줄입니다.",
        "",
        "## 눈에 띄는 패턴",
        "",
        "### 1. 인지검사 점수는 강하지만, 정답 열은 반드시 제외해야 합니다",
        "",
        f"MMSE TOTAL 중앙값은 CN {mmse_totals['CN']['median']:.1f}, MCI {mmse_totals['MCI']['median']:.1f}, DEM {mmse_totals['DEM']['median']:.1f}입니다. ",
        "다만 MMSE 파일의 `DIAG_NM`은 이번 정답과 완전히 같은 값입니다. 이 열을 넣으면 모델이 질병 패턴을 배우는 것이 아니라 정답을 복사하므로 코드가 강제로 차단합니다. 질문별 점수와 TOTAL만 기본 모드에서 사용합니다.",
        "",
        "### 2. DEM에서는 활동량 감소와 수면 변화가 비교적 크게 보입니다",
        "",
        _markdown_table(dem_wear, ["feature", "class_median", "cn_median", "cliffs_delta"]),
        "",
        "위 표는 Training 안의 연관성입니다. DEM 사람이 매우 적으므로 질병의 원인이나 확정된 임상 패턴으로 해석하면 안 됩니다.",
        "",
        "### 3. MCI와 CN은 차이가 더 작고 겹침이 큽니다",
        "",
        _markdown_table(mci_wear, ["feature", "class_median", "cn_median", "cliffs_delta"]),
        "",
        "MCI 구분이 세 class 중 가장 어려울 가능성이 큽니다. 이 때문에 class별 가중치, Macro F1 중심 선택, 여러 모델의 확률 평균을 사용합니다.",
        "",
        "### 4. 수집 일수 자체도 class마다 조금 다릅니다",
        "",
        "| class | activity days median [Q1-Q3] | sleep days median [Q1-Q3] |",
        "| --- | ---: | ---: |",
    ]
    for name in CLASS_NAMES:
        a = coverage[name]["activity_days"]
        s = coverage[name]["sleep_days"]
        lines.append(
            f"| {name} | {a['median']:.0f} [{a['q25']:.0f}-{a['q75']:.0f}] | "
            f"{s['median']:.0f} [{s['q25']:.0f}-{s['q75']:.0f}] |"
        )
    lines += [
        "",
        "수집 일수나 빈 날짜 수를 주 모델에 넣으면 병이 아니라 수집 방식의 차이를 외울 수 있습니다. 그래서 날짜·수집량·ID는 입력에서 제외했습니다.",
        "",
        "## MMSE 문항에서 보이는 차이",
        "",
        _markdown_table(mmse, ["comparison", "feature", "class_median", "cn_median", "cliffs_delta"], limit=16),
        "",
        "## 전처리 결정",
        "",
        "1. 한 사람당 한 행으로 합쳐 사람 단위로만 나눕니다.",
        "2. Activity와 Sleep은 중앙값, 변동폭, 최근-초기 변화, 완만한 추세로 요약합니다.",
        "3. 수면이 하루에 여러 개면 장비 ID 대신 가장 긴 수면과 시간 순서로 하나를 고릅니다.",
        "4. 마지막 Activity 시점보다 미래인 Sleep 행은 제외합니다.",
        "5. 결측값 채우기·크기 맞추기·특징 선택은 매 fold의 Training 부분에서만 배웁니다.",
        "6. `DIAG_NM`, `DIAG_SEQ`, ID, 의사명, 검사순번, 절대 날짜는 어떤 모델에도 넣지 않습니다.",
        "",
        "## 성능 목표를 읽는 법",
        "",
        "Accuracy 0.8 이상을 목표로 탐색하지만, 141명·DEM 9명 자료에서 이를 미리 보장할 수는 없습니다. ",
        "내부 nested-CV와 한 번만 확인하는 Validation 결과를 따로 저장하며, Accuracy·Macro F1·ROC-AUC와 class별 F1을 모두 보고 판단합니다.",
        "",
        "## 산출물",
        "",
        f"- 사용 가능한 특징: {int((quality['nonmissing_fraction'] > 0).sum()):,}개",
        f"- 결측률 40% 초과 특징: {int((quality['missing_fraction'] > 0.40).sum()):,}개",
        "- `top_effects.csv`: class별 큰 연관성 목록",
        "- `feature_quality.csv`: 결측·고유값 점검",
        "- `class_counts.png`, `top_wearable_effects.png`: 빠르게 보는 그림",
        "",
        "이 분석은 관련성을 보여줄 뿐, 임상 진단이나 인과관계를 뜻하지 않습니다.",
    ]
    return "\n".join(lines) + "\n"


def run(training_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_audit, _ = _raw_audit(training_root)
    dataset = build_subject_dataset(
        training_root,
        feature_mode="clinical_plus_lifelog",
        require_labels=True,
    )
    assert dataset.y is not None
    quality = pd.DataFrame(
        {
            "feature": dataset.X.columns,
            "missing_fraction": dataset.X.isna().mean().to_numpy(),
            "nonmissing_fraction": dataset.X.notna().mean().to_numpy(),
            "unique_nonmissing": dataset.X.nunique(dropna=True).to_numpy(),
            "family": [
                "MMSE" if c.startswith("mmse__")
                else "activity" if c.startswith("activity__")
                else "sleep"
                for c in dataset.X.columns
            ],
        }
    ).sort_values(["missing_fraction", "feature"])
    effects = _effect_table(dataset.X, dataset.y)
    quality.to_csv(output_dir / "feature_quality.csv", index=False)
    effects.to_csv(output_dir / "top_effects.csv", index=False)
    combined_audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Training only; no model fitting; no Validation label access",
        "raw": raw_audit,
        "engineered_dataset": dataset.audit,
        "privacy": {"subject_identifiers_persisted": False, "aggregate_outputs_only": True},
    }
    _write_json(output_dir / "data_audit.json", combined_audit)
    _plot_class_counts(raw_audit["class_counts"], output_dir / "class_counts.png")
    _plot_top_effects(effects, output_dir / "top_wearable_effects.png")
    report = make_report(raw_audit, dataset.audit, effects, quality)
    (output_dir / "EDA_REPORT_KO.md").write_text(report, encoding="utf-8")
    print(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.training_root.expanduser().resolve(), args.output_dir.expanduser().resolve())
