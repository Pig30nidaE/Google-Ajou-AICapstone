"""실험 A — ``paper_reported_reconstruction``.

논문이 서술한 절차를 그대로 재구성한다. **검증방법이 적절하다고 전제하지 않는다.**

논문 절차는 설계상 누수를 포함하므로(행 단위 분할, 전체 데이터 전처리) 감사기를
``observe`` 모드로 돌린다. 위반을 막는 대신 **정량 측정해 보고**하는 것이 산출물이다.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..audit.leakage import LeakageAuditor
from ..augmentation.generators import augment_train_fold
from ..data.loader import LifelogData
from ..data.paper_reference import (
    DERIVED_SPLIT_FACTS,
    RECONSTRUCTED_EVAL_SETS,
    SECTION51_AFTER_OUTLIER,
    TABLE6_WIDE_DEEP,
)
from ..evaluation.aggregate import aggregate_to_subject
from ..evaluation.metrics import compute_metrics
from ..models.registry import PAPER_MODEL_ORDER, fit_classifier
from ..preprocessing.pipeline import FoldPreprocessor
from ..splits.row_level import compare_with_table5, paper_row_split
from ..utils.io import RunPaths, save_json, save_provenance, save_table

log = logging.getLogger(__name__)

__all__ = ["run_experiment_a", "plan_experiment_a"]


def _apply_outlier_before_split(data: LifelogData, cfg: dict, *, seed: int):
    """논문 §5.1의 순서 — split **이전에** 전체 데이터에서 이상치 처리기를 적합한다.

    이것이 I-7의 누수다. 의도적으로 재현하며 감사기가 관측 결과를 기록한다.

    Returns:
        (이상치 처리된 데이터, 처리기 설명, 삭제 행 수).
    """
    from ..preprocessing.outliers import make_outlier_handler

    handler = make_outlier_handler(cfg.get("outlier"), seed=seed)
    handler.fit(data.X, data.y)
    res = handler.transform(data.X, data.y)
    return data.with_features(res.X).take(res.keep_mask), handler.describe(), int(res.n_dropped)


def plan_experiment_a(data: LifelogData, cfg: dict, *, seed: int) -> dict:
    """--dry-run: 학습 없이 절차와 규모만 확인한다.

    실제 실행과 동일하게 **이상치 제거를 split 이전에** 수행하므로,
    표 5 대조가 실제 실행 결과를 예측한다.
    """
    before = data.class_counts(by="record")
    outlier_desc, n_dropped = {"method": "none"}, 0
    if (cfg.get("preprocessing") or {}).get("fit_scope", "all_data") == "all_data":
        data, outlier_desc, n_dropped = _apply_outlier_before_split(data, cfg, seed=seed)
    after = data.class_counts(by="record")

    split = paper_row_split(
        data,
        ratios=tuple(cfg.get("split", {}).get("ratios", (0.8, 0.1, 0.1))),
        seed=seed,
    )
    train_sub = set(data.subject[split.train_idx])
    test_sub = set(data.subject[split.test_idx])
    valid_sub = set(data.subject[split.valid_idx])
    n_syn = int(((cfg.get("augmentation") or {}).get("vae") or {}).get("n_synthetic") or 0)

    dem_train = int((data.y[split.train_idx] == 2).sum())
    return {
        "experiment": "A",
        "split_unit": "row (논문 절차 재현)",
        "outlier_method": outlier_desc,
        "rows_before_outlier": before,
        "rows_after_outlier": after,
        "n_rows_dropped_by_outlier": n_dropped,
        "paper_after_outlier": dict(SECTION51_AFTER_OUTLIER),
        "outlier_matches_paper": after == dict(SECTION51_AFTER_OUTLIER),
        "n_train": len(split.train_idx),
        "n_valid": len(split.valid_idx),
        "n_test": len(split.test_idx),
        "n_train_subjects": len(train_sub),
        "n_test_subjects": len(test_sub),
        "subject_overlap_train_test": len(train_sub & test_sub),
        "subject_overlap_train_valid": len(train_sub & valid_sub),
        "preprocessing_fit_scope": (cfg.get("preprocessing") or {}).get("fit_scope", "all_data"),
        "scaler_scope": (cfg.get("preprocessing") or {}).get("scaler_scope"),
        "vae_fit_scope": ((cfg.get("augmentation") or {}).get("vae") or {}).get("fit_scope"),
        "dem_train_rows_real": dem_train,
        "expected_synthetic_rows": n_syn,
        "dem_train_rows_after_augmentation": dem_train + n_syn,
        "paper_table5_dem_train": 4412,
        "paper_derived_synthetic": DERIVED_SPLIT_FACTS["dem_synthetic_rows"],
        "table5_comparison": compare_with_table5(data, split, n_syn).to_dict(orient="records"),
        "note": (
            "행 단위 분할이므로 피험자 중복이 발생한다. 이는 논문 절차의 재현이며 "
            "실험 B·C가 통제하려는 대상이다 (report_inconsistencies.md I-6)."
        ),
    }


def run_experiment_a(
    data: LifelogData,
    cfg: dict,
    *,
    out_root: str,
    label: str,
    seed: int = 42,
    models: tuple[str, ...] = PAPER_MODEL_ORDER,
    augmentations: tuple[str, ...] = ("none", "vae"),
) -> dict:
    """논문 보고 방법 재구성 실행.

    사용자 요구 7: 증강 전과 증강 후를 **같은 split**에서 비교한다.
    (논문은 그러지 않았다 — report_inconsistencies.md I-4. 그 사실을 결과에 각주로 남긴다.)
    """
    paths = RunPaths(out_root, f"A_{label}")
    auditor = LeakageAuditor(mode="observe", name=f"A_{label}")

    # ---- (1) 전처리를 split **이전에** 수행한다 = 논문 §5.1의 순서 (누수, 의도적)
    fit_scope = (cfg.get("preprocessing") or {}).get("fit_scope", "all_data")
    # fit_scope=all_data이면 이상치 처리는 아래에서 이미 한 번만 적용한다. 이후 fold
    # preprocessor에 같은 IsolationForest를 다시 fit하면 imputer만 두 번째 subset을
    # 보게 되는 비일관 경로가 생기므로 outlier=no-op으로 명시한다.
    global_outlier_applied = fit_scope == "all_data"
    pre_cfg = {
        "outlier": {"method": "none"} if global_outlier_applied else cfg.get("outlier"),
        "preprocessing": cfg.get("preprocessing"),
    }

    row_counts_before = data.class_counts(by="record")
    global_outlier_fit_subjects = None
    global_outlier_fit_row_ids = None
    global_outlier_fit_n_rows = 0
    if fit_scope == "all_data":
        # 감사기에 신고하려면 fold가 먼저 등록되어야 하므로(그것이 감사기의 요점이다),
        # 논문 순서를 재현하는 이 단계는 직접 수행하고 관측만 기록한다.
        n_before = data.n
        global_outlier_fit_subjects = data.subject.copy()
        global_outlier_fit_row_ids = data.row_id.copy()
        global_outlier_fit_n_rows = data.n
        data, outlier_desc, n_dropped = _apply_outlier_before_split(data, cfg, seed=seed)
        auditor.observations.append(
            {
                "kind": "outlier_fit_on_all_data",
                "detail": outlier_desc,
                "n_rows_seen": n_before,
                "n_dropped": n_dropped,
                "note": "논문 §5.1의 순서. split 이전이므로 test·valid가 임계값 결정에 사용되었다 (I-7).",
            }
        )
    row_counts_after = data.class_counts(by="record")
    outlier_table = pd.DataFrame(
        [
            {
                "class": c,
                "before": row_counts_before.get(c, 0),
                "after_reproduction": row_counts_after.get(c, 0),
                "paper_section51": SECTION51_AFTER_OUTLIER[c],
                "diff_vs_paper": row_counts_after.get(c, 0) - SECTION51_AFTER_OUTLIER[c],
            }
            for c in ("CN", "MCI", "Dem")
        ]
    )
    outlier_table.loc[len(outlier_table)] = {
        "class": "TOTAL",
        "before": sum(row_counts_before.values()),
        "after_reproduction": sum(row_counts_after.values()),
        "paper_section51": sum(SECTION51_AFTER_OUTLIER.values()),
        "diff_vs_paper": sum(row_counts_after.values()) - sum(SECTION51_AFTER_OUTLIER.values()),
    }
    save_table(outlier_table, paths("row_counts.csv"))

    # ---- (2) 행 단위 8:1:1 분할 (피험자 누수 의도적)
    split = paper_row_split(
        data, ratios=tuple(cfg.get("split", {}).get("ratios", (0.8, 0.1, 0.1))), seed=seed
    )
    fold_id = split.fold_id
    auditor.register_split(
        fold_id,
        train_subjects=data.subject[split.train_idx],
        eval_subjects=data.subject[split.test_idx],
        train_row_ids=data.row_id[split.train_idx],
        eval_row_ids=data.row_id[split.test_idx],
        validation_subjects=data.subject[split.valid_idx],
        validation_row_ids=data.row_id[split.valid_idx],
        require_disjoint_subjects=False,   # 행 단위 분할이므로 중복을 측정만 한다
    )
    if global_outlier_applied:
        # 실제 fit은 split 전에 일어났지만 경계를 등록한 뒤 역사적 이벤트로 신고한다.
        # row ID 검사를 함께 해야 행 단위 split의 피험자 중복 때문에 누수가 가려지지 않는다.
        auditor.record_fit(
            "global_outlier_detector",
            fold_id,
            subjects=global_outlier_fit_subjects,
            row_ids=global_outlier_fit_row_ids,
            n_rows=global_outlier_fit_n_rows,
            occurred_before_split=True,
        )

    train0 = data.take(split.train_idx)
    valid0 = data.take(split.valid_idx)
    test0 = data.take(split.test_idx)

    results: dict[tuple[str, str], dict] = {}
    # 논문은 기록 단위로만 평가했다. 교차 실험 비교(실험 B·C는 피험자 단위)를 위해
    # 같은 예측에서 피험자 단위 지표도 함께 보관한다 — 평가단위를 섞지 않기 위해서다.
    results_subject: dict[tuple[str, str], dict] = {}
    per_run: list[dict] = []
    aug_diags: dict[str, dict] = {}

    for aug in augmentations:
        # 증강 전/후가 **동일 split**을 쓰도록 매 조건마다 같은 인덱스에서 시작한다.
        pre = FoldPreprocessor(pre_cfg, auditor=auditor, fold_id=fold_id, seed=seed)
        # all_data 설정에서는 실제 이상치 처리는 위에서 한 번 끝났고, 여기서는 no-op
        # outlier + imputer의 명시된 fit 범위를 그대로 재현한다. 결측은 0건이지만
        # config와 감사 로그가 서로 다른 범위를 말하지 않게 한다.
        pre.fit(data if global_outlier_applied else train0)

        if global_outlier_applied:
            train_clean, valid_clean, test_clean = train0, valid0, test0
        else:
            train_clean = pre.apply_outlier(train0)
            valid_clean = pre.apply_outlier_eval(valid0)
            test_clean = pre.apply_outlier_eval(test0)

        aug_cfg = dict(cfg.get("augmentation") or {})
        aug_cfg["method"] = aug

        # scaler_scope=all_data는 논문 §5.1 흐름의 재현이다 (I-8, 누수).
        # input_space=raw이면 VAE는 원 단위에서 학습되며(§4.2 순서), generators가
        # preprocessor로 inverse scaling을 수행한다.
        pre.fit_scaler(data if pre.scaler_scope == "all_data" else train_clean)
        train_s = pre.transform(train_clean)

        res = augment_train_fold(
            train_s, aug_cfg, auditor=auditor, fold_id=fold_id,
            preprocessor=pre, seed=seed,
        )
        aug_diags[aug] = res.diagnostics
        test_s = pre.transform(test_clean)
        valid_s = pre.transform(valid_clean)
        auditor.record_eval(fold_id, is_synthetic=test_s.is_synthetic, where="test")
        auditor.record_eval(fold_id, is_synthetic=valid_s.is_synthetic, where="valid")

        for model_name in models:
            model = fit_classifier(
                model_name, res.data, cfg, auditor=auditor, fold_id=fold_id,
                class_weight=res.class_weight, validation=valid_s, seed=seed,
            )
            proba = model.predict_proba(test_s.X.to_numpy())
            # 논문은 기록 단위로 평가했다. 그것이 실험 A의 주 지표다.
            m_rec = compute_metrics(test_s.y, proba, unit="record")
            valid_proba = model.predict_proba(valid_s.X.to_numpy())
            m_valid = compute_metrics(valid_s.y, valid_proba, unit="record")
            subj = aggregate_to_subject(
                test_s.subject, test_s.y, proba,
                is_synthetic=test_s.is_synthetic,
                method=(cfg.get("aggregate") or {}).get("method", "mean"),
            )
            m_sub = compute_metrics(subj.y, subj.proba, unit="subject")
            # 기록 단위 지표의 n_Dem은 Dem '행' 수다. 결과표에서 이를 피험자 수로
            # 오인하거나 코호트 전체 12명으로 고정하지 않도록 실제 test 피험자 수를
            # 별도 메타데이터로 전달한다.
            m_rec["n_dem_subjects_eval"] = int(m_sub["n_Dem"])
            m_rec["n_dem_subjects_correct"] = int(m_sub["n_Dem_correct"])
            results[(model_name, aug)] = m_rec
            results_subject[(model_name, aug)] = m_sub
            per_run.append(
                {
                    "experiment": "A",
                    "config_label": label,
                    "model": model_name,
                    "augmentation": aug,
                    "seed": seed,
                    **{f"record_{k}": v for k, v in m_rec.items() if not isinstance(v, (list, dict))},
                    **{f"validation_record_{k}": v for k, v in m_valid.items()
                       if not isinstance(v, (list, dict))},
                    **{f"subject_{k}": v for k, v in m_sub.items() if not isinstance(v, (list, dict))},
                }
            )
            save_json(
                {
                    "record_level": m_rec,
                    "validation_record_level": m_valid,
                    "subject_level": m_sub,
                    "model": model.describe(),
                    "fit_log": model.fit_log,
                },
                paths("per_model", f"{model_name}_{aug}.json"),
            )

        save_provenance(res.data.provenance, paths("synthetic_provenance"))

    metrics_df = pd.DataFrame(per_run)
    save_table(metrics_df, paths("record_level_metrics.csv"))
    save_table(
        compare_with_table5(data, split, aug_diags.get("vae", {}).get("n_synthetic", 0)),
        paths("table5_comparison.csv"),
    )
    save_table(_paper_comparison(results, label), paths("paper_comparison.csv"))
    save_json(aug_diags, paths("augmentation_diagnostics.json"))
    auditor.save(paths("leakage_observation.json"))

    return {
        "results": results,
        "results_subject": results_subject,
        "metrics_frame": metrics_df,
        "audit": auditor.summary(),
        "paths": paths,
    }


def _paper_comparison(results: dict[tuple[str, str], dict], label: str) -> pd.DataFrame:
    """재현값과 논문 보고값을 나란히 놓는다 (수치를 임의로 고치지 않는다 — 지시 10)."""
    rows = []
    for (model, aug), m in sorted(results.items()):
        row: dict[str, object] = {
            "config_label": label,
            "model": model,
            "augmentation": aug,
            "reproduction_macro_f1": round(m["macro_f1"], 4),
            "reproduction_CN_f1": round(m["CN_f1"], 4),
            "reproduction_MCI_f1": round(m["MCI_f1"], 4),
            "reproduction_Dem_f1": round(m["Dem_f1"], 4),
            "reproduction_Dem_recall": round(m["dem_recall"], 4),
            "reproduction_n_eval_rows": m["n"],
            "reproduction_n_dem_subjects_eval": m.get("n_dem_subjects_eval"),
        }
        if model == "wide_deep":
            col = "without_augmentation" if aug == "none" else "with_vae"
            p = TABLE6_WIDE_DEEP[col]
            row.update(
                {
                    "paper_table6_macro_f1": p["macro_avg"]["f1"],
                    "paper_table6_CN_f1": p["CN"]["f1"],
                    "paper_table6_MCI_f1": p["MCI"]["f1"],
                    "paper_table6_Dem_f1": p["Dem"]["f1"],
                    "paper_table6_Dem_recall": p["Dem"]["recall"],
                    "paper_reconstructed_n_eval_rows": RECONSTRUCTED_EVAL_SETS[col]["n_total"],
                }
            )
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["paper_note"] = (
        "논문 표 6은 Wide & Deep만 증강 전/후를 보고한다. 나머지 3모델의 '증강 없음'은 미보고다 (I-12). "
        "또한 표 6의 두 열은 서로 다른 평가셋(N=1097 vs 1095)에서 측정되었다 (I-4). "
        "본 재현은 두 조건을 동일 split·동일 seed에서 비교했다."
    )
    return df
