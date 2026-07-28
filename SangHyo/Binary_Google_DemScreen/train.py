"""Dementia screening (CN+MCI vs Dem) evaluated as a 2x2 sensitivity design.

Four arms are always run and always reported together, so no single favourable
combination can be presented as "the" result:

                      full cohort (174)      quality-filtered
    wearable only           HEADLINE              sensitivity
    wearable + MMSE        reference              sensitivity

* **wearable only** is the headline because it answers the question that is
  actually interesting -- can a wrist device flag dementia *without* a cognitive
  test.  **+MMSE** is reported as a reference ceiling, flagged as partly
  circular: MMSE is the instrument clinicians use to make this diagnosis, so
  predicting the diagnosis from it is closer to re-reading the label than to
  screening.
* **quality-filtered** applies the label-blind rules in ``data.QUALITY_RULES``.
  The gap between the two cohort columns is the sensitivity to exclusions, which
  is reported rather than resolved by picking the better number.

Context the report carries by construction: the same MMSE TOTAL feature scores
~0.95 on this task and ~0.73 on the CN vs MCI+Dem task used by every other
binary folder here.  That number is computed and written into the report so the
two tasks are never silently compared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from SangHyo.Binary_Google_MaxAUC_Tuned.engine import safe_auc

from .data import Cohort, DIAG_ORDER, QUALITY_RULES, hash_subject_id, load_cohort, quality_flags
from .engine import bootstrap_auc_ci, repeated_nested_cv, univariate_table
from .learners import ALL_KINDS, SMOTE_AVAILABLE

EXPERIMENT_NAME = "Binary_Google_DemScreen"


def _log(message: str) -> None:
    print(message, flush=True)


@dataclass
class RunConfig:
    data_root: str
    output_dir: str
    run_mode: str = "full"
    kinds: tuple[str, ...] = ALL_KINDS
    repeats: int = 20
    outer_k: int = 5
    inner_k: int = 4
    auc_gate: float = 0.55
    search: bool = False
    smote: bool = False
    seed: int = 20260728
    extra: dict[str, Any] = field(default_factory=dict)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
                    encoding="utf-8")


def _task_difficulty_context(cohort: Cohort) -> dict:
    """Why this folder's numbers look higher than the CN vs MCI+Dem folders."""

    index = {n: i for i, n in enumerate(cohort.feature_names)}
    total = cohort.X[:, index["mmse_TOTAL"]] if "mmse_TOTAL" in index else None
    if total is None:
        return {}
    severity = cohort.severity
    definitions = {
        "this_folder_CN+MCI_vs_Dem": (severity == 2).astype(int),
        "other_folders_CN_vs_MCI+Dem": (severity >= 1).astype(int),
    }
    out = {}
    for name, y in definitions.items():
        auc = safe_auc(y, -total)
        out[name] = {"mmse_total_only_roc_auc": round(max(auc, 1 - auc), 4),
                     "n_positive": int(y.sum()), "n_negative": int((y == 0).sum())}
    mask = severity <= 1
    auc = safe_auc((severity[mask] == 1).astype(int), -total[mask])
    out["hard_boundary_CN_vs_MCI"] = {"mmse_total_only_roc_auc": round(max(auc, 1 - auc), 4),
                                      "n_positive": int((severity == 1).sum()),
                                      "n_negative": int((severity == 0).sum())}
    out["mmse_total_mean_by_group"] = {
        d: round(float(np.nanmean(total[cohort.diagnosis == d])), 2) for d in DIAG_ORDER}
    out["note"] = ("Same single feature, different question: this folder's task is "
                   "far more separable. Higher numbers here are a property of the "
                   "label definition, not evidence of a better model.")
    return out


def _run_arm(cohort: Cohort, name: str, config: RunConfig) -> dict:
    _log(f"  [{name}] {cohort.n_subjects} subjects x {cohort.n_features} features, "
         f"Dem {int(cohort.y.sum())}")
    result = repeated_nested_cv(cohort.X, cohort.y, config.kinds, repeats=config.repeats,
                                outer_k=config.outer_k, inner_k=config.inner_k,
                                auc_gate=config.auc_gate, search=config.search,
                                seed=config.seed, log=_log)
    result["bootstrap_ci"] = bootstrap_auc_ci(cohort.y, np.asarray(result["mean_oof_prob"]),
                                              seed=config.seed)
    result["n_subjects"] = int(cohort.n_subjects)
    result["n_features"] = int(cohort.n_features)
    result["n_positive"] = int(cohort.y.sum())
    result["top_univariate"] = univariate_table(cohort.X, cohort.y, cohort.feature_names)
    picks = result.pop("univariate_pick_counts", {})
    result["univariate_most_picked_features"] = {
        cohort.feature_names[int(j)]: int(c) for j, c in picks.items()
        if int(j) < cohort.n_features}
    ci = result["bootstrap_ci"]
    _log(f"    -> ROC-AUC {result['roc_auc']['mean']:.4f} "
         f"(+/-{result['roc_auc']['std']:.4f} across repeats), "
         f"bootstrap 95% CI [{ci['lo']:.3f}, {ci['hi']:.3f}]")
    return result


def run_experiment(config: RunConfig) -> dict:
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    wall = time.monotonic()

    _log(f"[{EXPERIMENT_NAME}] mode={config.run_mode}  repeats={config.repeats}  "
         f"search={config.search}  smote={config.smote and SMOTE_AVAILABLE}")

    cohort = load_cohort(config.data_root)
    flags = quality_flags(cohort)
    _log(f"  cohort: {cohort.n_subjects} subjects "
         f"({', '.join(f'{d} {int((cohort.diagnosis == d).sum())}' for d in DIAG_ORDER)})")
    _log(f"  label-blind quality flags: {len(flags)} subject(s)")
    for subject, reasons in flags.items():
        _log(f"    - {hash_subject_id(subject)}: {reasons[0]}")

    keep = np.array([s not in flags for s in cohort.subject_ids])
    arms = {
        "wearable_only__full": cohort.wearable_only(),
        "wearable_plus_mmse__full": cohort,
        "wearable_only__filtered": cohort.wearable_only().subset(keep),
        "wearable_plus_mmse__filtered": cohort.subset(keep),
    }

    results = {}
    for name, arm_cohort in arms.items():
        results[name] = _run_arm(arm_cohort, name, config)

    headline = results["wearable_only__full"]
    (output / "training").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "subject_hash": [hash_subject_id(s) for s in cohort.subject_ids],
        "diagnosis": cohort.diagnosis,
        "y_dementia": cohort.y,
        "mean_oof_prob_wearable_only": headline["mean_oof_prob"],
        "quality_flagged": ~keep,
    }).to_csv(output / "training" / "oof_predictions_hashed.csv", index=False)

    report = {
        "experiment": EXPERIMENT_NAME,
        "task": "CN+MCI (0) vs Dem (1) — dementia screening",
        "run_mode": config.run_mode,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.monotonic() - wall,
        "primary_metric": "ROC-AUC (threshold-free), repeated nested CV",
        "headline_arm": "wearable_only__full",
        "headline_roc_auc": headline["roc_auc"]["mean"],
        "headline_bootstrap_95ci": [headline["bootstrap_ci"]["lo"], headline["bootstrap_ci"]["hi"]],
        "cohort": {
            "n_subjects": int(cohort.n_subjects),
            "counts": {d: int((cohort.diagnosis == d).sum()) for d in DIAG_ORDER},
            "n_positive_dementia": int(cohort.y.sum()),
            "quality_flagged": {hash_subject_id(s): r for s, r in flags.items()},
            "quality_rules": {k: v[2] for k, v in QUALITY_RULES.items()},
        },
        "task_difficulty_context": _task_difficulty_context(cohort),
        "arms": results,
        "config": {
            "kinds": list(config.kinds), "repeats": config.repeats, "outer_k": config.outer_k,
            "inner_k": config.inner_k, "auc_gate": config.auc_gate, "search": config.search,
            "smote": config.smote, "smote_available": SMOTE_AVAILABLE, "seed": config.seed,
        },
        "limitations": [
            "양성(Dem) 12명. 어떤 점추정치도 부트스트랩 CI 없이는 읽으면 안 됨.",
            "train+validation을 합쳐 교차검증하므로 별도 홀드아웃 테스트셋이 없음.",
            "+MMSE 결과는 참고용. MMSE는 이 진단을 내리는 검사 도구 자체라 부분적으로 순환적.",
            "품질 필터는 라벨을 보지 않는 규칙으로 사전 정의했고, 두 결과를 모두 보고함 "
            "(성능이 좋아지는 쪽을 고르지 않음).",
        ],
        "honesty_note": (
            "Feature subset, blend weights and the operating threshold are all chosen "
            "inside the outer-training part of each fold. ROC-AUC is threshold-free and "
            "is the headline; threshold-dependent metrics come from a nested threshold."
        ),
    }
    _write_json(output / "training" / "FINAL_REPORT.json", report)
    _write_json(output / "training" / "TRAINING_COMPLETE.json",
                {"status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
                 "headline_roc_auc": headline["roc_auc"]["mean"]})

    ci = headline["bootstrap_ci"]
    _log(f"\nHeadline (wearable only, full cohort): ROC-AUC {headline['roc_auc']['mean']:.4f} "
         f"[95% CI {ci['lo']:.3f}-{ci['hi']:.3f}]")
    _log(f"Reference (+MMSE, full cohort):        ROC-AUC "
         f"{results['wearable_plus_mmse__full']['roc_auc']['mean']:.4f}")
    return report


__all__ = ["EXPERIMENT_NAME", "RunConfig", "run_experiment"]
