"""Read-only input audit and explicitly authorized, row-local derivations."""
from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

DOMAIN_FEATURES = [
    "sleep_score_alignment", "sleep_hr_5min_max_std", "sleep_awake_std",
    "sleep_breath_average", "activity_score_std", "activity_class_3_count_std",
    "activity_met_min_low_std", "sleep_restless_std", "circadian_IV",
    "circadian_IS", "circadian_RA", "sleep_wake_bouts_avg", "HR_drop_ratio",
    "Circadian_Strain",
]
TARGET = "original_label"
GROUP = "EMAIL"
DERIVATIONS = {
    "HR_drop_ratio": {
        "sources": ["sleep_hr_average", "sleep_hr_lowest"],
        "formula": "(sleep_hr_average - sleep_hr_lowest) / (sleep_hr_average + 1e-5)",
    },
    "Circadian_Strain": {
        "sources": ["circadian_IV", "circadian_IS"],
        "formula": "circadian_IV / (circadian_IS + 1e-5)",
    },
}


def forbidden_reason(column):
    name = column.upper()
    if column == TARGET or name in {"LABEL", "TARGET", "STAGE1_LABEL", "STAGE2_LABEL"}:
        return "target_or_label"
    if "MMSE" in name or "TOTAL" == name or re.match(r"^Q\d+(?:_|$)", name):
        return "cognitive_score"
    if "DIAG" in name or "DOCTOR" in name:
        return "diagnostic_information"
    if ("EMAIL" in name or name in {"ID", "SUBJECT", "SUBJECT_ID", "PATIENT_ID", "DATE", "FOLD"}
            or name.endswith("_ID")):
        return "identifier_or_split_metadata"
    return None


def inspect_input(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        rows = list(reader)
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("Input must have unique column names")
    if not rows or not {TARGET, GROUP}.issubset(columns):
        raise ValueError("Nonempty input requires original_label and EMAIL")
    groups = [row[GROUP].strip() for row in rows]
    if any(not group for group in groups):
        raise ValueError("EMAIL must be nonmissing and nonblank")
    labels = []
    subject_labels = {}
    for row, group in zip(rows, groups):
        value = float(row[TARGET])
        if value not in (0, 1, 2):
            raise ValueError("original_label must be exactly 0, 1 or 2")
        label = int(value)
        if group in subject_labels and subject_labels[group] != label:
            raise ValueError("A subject has conflicting target labels")
        subject_labels[group] = label
        labels.append(label)
    counts = Counter(subject_labels.values())
    if set(counts) != {0, 1, 2} or min(counts.values()) < 5:
        raise ValueError("Five-fold evaluation requires at least five subjects per class")
    for feature in DOMAIN_FEATURES:
        if forbidden_reason(feature):
            raise ValueError(f"Forbidden domain feature: {feature}")
    excluded = [{"column": c, "reason": forbidden_reason(c) or "not_in_fixed_domain_features"}
                for c in columns if forbidden_reason(c) or c not in DOMAIN_FEATURES]
    zero_columns = ["circadian_IS", "circadian_IV", "circadian_RA", "sleep_wake_bouts_avg"]
    zero_count = (sum(all(float(row[c]) == 0 for c in zero_columns) for row in rows)
                  if set(zero_columns).issubset(columns) else None)
    return {
        "input_path": str(path.resolve()),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "n_rows": len(rows), "n_columns": len(columns), "n_subjects": len(subject_labels),
        "one_row_per_subject": len(rows) == len(subject_labels),
        "target_column": TARGET, "group_column": GROUP,
        "class_names": {0: "CN", 1: "MCI", 2: "Dementia"},
        "row_class_counts": dict(sorted(Counter(labels).items())),
        "subject_class_counts": dict(sorted(counts.items())),
        "missing_domain_features": [f for f in DOMAIN_FEATURES if f not in columns],
        "excluded_columns": excluded, "fixed_features": DOMAIN_FEATURES,
        "rows_with_four_circadian_values_zero": zero_count,
    }


def load_dataset(path, allow_derived=False):
    import pandas as pd

    audit = inspect_input(path)
    missing = audit["missing_domain_features"]
    unresolved = [f for f in missing if not allow_derived or f not in DERIVATIONS]
    if unresolved:
        raise ValueError(f"Missing fixed features; no substitutions allowed: {unresolved}")
    df = pd.read_csv(path, dtype={GROUP: "string"})
    source_columns = []
    for feature in DERIVATIONS:
        spec = DERIVATIONS[feature]
        absent = [c for c in spec["sources"] if c not in df]
        if absent:
            raise ValueError(f"Missing derivation sources for {feature}: {absent}")
        source_columns.extend(spec["sources"])
    # No derivation or fitted preprocessing before splitting. Do not return labels/IDs in X.
    raw_columns = list(dict.fromkeys([c for c in DOMAIN_FEATURES if c not in DERIVATIONS] + source_columns))
    X = df[raw_columns].apply(pd.to_numeric, errors="raise")
    audit["derived_features"] = DERIVATIONS
    audit["derivation_scope"] = "inside_each_outer_fold_preprocessing_without_fit"
    audit["missing_cells_per_raw_column"] = X.isna().sum().to_dict()
    groups = df[GROUP].str.strip().to_numpy(dtype=str)
    return X, df[TARGET].to_numpy(dtype=int), groups, audit


def derive_wearable_features(raw, fold, role):
    """Pure row-local formulas; no target, statistics or fit. Return audit before imputation."""
    import numpy as np
    import pandas as pd

    sources = list(dict.fromkeys(c for spec in DERIVATIONS.values() for c in spec["sources"]))
    required = list(dict.fromkeys([c for c in DOMAIN_FEATURES if c not in DERIVATIONS] + sources))
    absent = [c for c in required if c not in raw]
    if absent:
        raise ValueError(f"Missing wearable source features: {absent}")
    # A strict allowlist ensures that even an accidental caller cannot feed metadata to the model.
    values = raw[required].apply(pd.to_numeric, errors="raise").copy()
    rows = []
    for source in sources:
        col = values[source].to_numpy(dtype=float)
        rows.append({"fold": fold, "role": role, "kind": "source", "feature": source,
                     "n_rows": len(col), "missing_before_cleaning": int(np.isnan(col).sum()),
                     "inf_before_cleaning": int(np.isinf(col).sum()),
                     "invalid_denominator_count": 0,
                     "missing_after_cleaning": int((~np.isfinite(col)).sum()), "inf_after_cleaning": 0})
    for feature, spec in DERIVATIONS.items():
        first, second = [values[c].to_numpy(dtype=float) for c in spec["sources"]]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            denominator = (first if feature == "HR_drop_ratio" else second) + 1e-5
            numerator = first - second if feature == "HR_drop_ratio" else first
            result = numerator / denominator
        bad_denominator = (denominator == 0) | ~np.isfinite(denominator)
        invalid = bad_denominator | ~np.isfinite(first) | ~np.isfinite(second) | ~np.isfinite(result)
        cleaned = np.where(invalid, np.nan, result)
        rows.append({"fold": fold, "role": role, "kind": "derived", "feature": feature,
                     "n_rows": len(result), "missing_before_cleaning": int(np.isnan(result).sum()),
                     "inf_before_cleaning": int(np.isinf(result).sum()),
                     "invalid_denominator_count": int(bad_denominator.sum()),
                     "missing_after_cleaning": int(np.isnan(cleaned).sum()),
                     "inf_after_cleaning": int(np.isinf(cleaned).sum())})
        values[feature] = cleaned
    return values[DOMAIN_FEATURES].replace([np.inf, -np.inf], np.nan), rows
