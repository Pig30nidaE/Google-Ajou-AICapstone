#!/usr/bin/env python3
"""Execute the deterministic Training-only feature contract without ML packages.

The output is aggregate-only: no raw or hashed subject identifiers and no
subject-level rows are written.  Validation and MMSE paths are never resolved.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import performance_lab_core as plc


LAB_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = LAB_ROOT.parent


def array_audit(values: np.ndarray) -> dict[str, int | float | list[int]]:
    array = np.asarray(values)
    finite = np.isfinite(array)
    return {
        "shape": list(array.shape),
        "finite_values": int(finite.sum()),
        "nonfinite_values": int((~finite).sum()),
        "nonfinite_fraction": float((~finite).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-root",
        type=Path,
        default=PROJECT_ROOT / "Data" / "1.Training",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=LAB_ROOT / "artifacts" / "local_eda" / "feature_contract_smoke.json",
    )
    args = parser.parse_args()

    training = plc.load_training_dataset(args.training_root)
    bundle = plc.build_feature_bundle(
        training.activity,
        training.sleep,
        training.subject_ids,
    )
    plc.validate_feature_bundle(bundle, expected_subjects=plc.EXPECTED_SUBJECTS)

    labels = training.labels.reindex(training.subject_ids).to_numpy(dtype=int)
    class_counts = np.bincount(labels, minlength=3)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Data/1.Training deterministic feature construction only",
        "subjects": int(len(bundle.subject_ids)),
        "class_counts": {
            plc.CLASS_NAMES[index]: int(class_counts[index]) for index in range(3)
        },
        "raw_rows": {
            "activity": int(len(training.activity)),
            "sleep": int(len(training.sleep)),
        },
        "event_summary": {
            **array_audit(bundle.event_summary.to_numpy(dtype=float)),
            "feature_count": int(bundle.event_summary.shape[1]),
            "feature_name_contract_sha256": plc.stable_json_hash(
                list(bundle.event_summary.columns)
            ),
        },
        "activity_event_sequence": {
            **array_audit(bundle.activity_sequence),
            "feature_count": len(bundle.activity_sequence_features),
            "feature_name_contract_sha256": plc.stable_json_hash(
                bundle.activity_sequence_features
            ),
        },
        "sleep_event_sequence": {
            **array_audit(bundle.sleep_sequence),
            "feature_count": len(bundle.sleep_sequence_features),
            "feature_name_contract_sha256": plc.stable_json_hash(
                bundle.sleep_sequence_features
            ),
        },
        "legacy_comparator": {
            **array_audit(bundle.legacy_values),
            "feature_count": len(bundle.legacy_features),
            "selection_eligible": False,
        },
        "coverage_negative_control": {
            **array_audit(bundle.coverage.to_numpy(dtype=float)),
            "feature_count": int(bundle.coverage.shape[1]),
            "selection_eligible": False,
        },
        "diagnostics": bundle.diagnostics,
        "contract_checks": {
            "feature_bundle_validated": True,
            "primary_forbidden_feature_check_passed": True,
            "subject_order_aligned_in_memory": True,
            "validation_source_or_labels_opened": False,
            "mmse_opened": False,
            "subject_level_rows_persisted": False,
            "raw_or_hashed_subject_ids_persisted": False,
        },
    }
    plc.write_json(args.output, report)
    print(
        {
            "output": str(args.output),
            "subjects": report["subjects"],
            "event_summary_shape": report["event_summary"]["shape"],
            "activity_sequence_shape": report["activity_event_sequence"]["shape"],
            "sleep_sequence_shape": report["sleep_event_sequence"]["shape"],
            "legacy_shape": report["legacy_comparator"]["shape"],
        }
    )


if __name__ == "__main__":
    main()
