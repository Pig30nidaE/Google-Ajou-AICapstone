"""실험 B — ``leakage_controlled_non_nested``.

논문 하이퍼파라미터를 **고정**한 채 피험자 분리와 전처리·VAE 범위만 통제한다.
하이퍼파라미터를 다시 선택하지 않는다.

파이프라인 순서(사용자 지시 5절)를 코드가 강제하며, 감사기는 ``enforce`` 모드다.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..audit.leakage import LeakageAuditor
from ..augmentation.generators import augment_train_fold
from ..data.loader import LifelogData
from ..diagnostics.synthetic_quality import full_report as synthetic_full_report
from ..diagnostics.tstr import run_trts, run_tstr
from ..evaluation.aggregate import aggregate_to_subject
from ..evaluation.bootstrap import bootstrap_ci
from ..evaluation.metrics import compute_metrics
from ..evaluation.tables import fold_variability
from ..models.registry import PAPER_MODEL_ORDER, fit_classifier
from ..preprocessing.pipeline import FoldPreprocessor
from ..splits.group_cv import describe_folds, make_group_folds
from ..utils.io import RunPaths, save_json, save_provenance, save_table

log = logging.getLogger(__name__)

__all__ = ["run_experiment_b", "plan_experiment_b", "run_one_fold"]


def _split_cfg(cfg: dict) -> dict:
    s = cfg.get("split") or {}
    return {
        "method": s.get("method", "subject_stratified"),
        "n_splits": int(s.get("n_splits", 3)),
        "n_repeats": int(s.get("n_repeats", 1)),
    }


def plan_experiment_b(data: LifelogData, cfg: dict, *, seed: int) -> dict:
    """--dry-run: fold 구성과 각 단계의 fit 범위를 학습 없이 확인한다."""
    s = _split_cfg(cfg)
    folds = make_group_folds(data, seed=seed, prefix="fold", **s)
    desc = describe_folds(data, folds)
    aug_cfg = (cfg.get("augmentation") or {})
    sub = (aug_cfg.get(aug_cfg.get("method", "none")) or {})

    expected = []
    for f in folds:
        n_dem = int((data.y[f.train_idx] == 2).sum())
        if sub.get("n_synthetic") is not None:
            n_syn = int(sub["n_synthetic"])
        elif sub.get("ratio_to_real") is not None:
            n_syn = int(round(n_dem * float(sub["ratio_to_real"])))
        else:
            n_syn = 0
        expected.append(
            {
                "fold_id": f.fold_id,
                "n_train_rows": int(len(f.train_idx)),
                "n_eval_rows": int(len(f.eval_idx)),
                "n_train_dem_rows": n_dem,
                "n_train_dem_subjects": int(
                    len(set(data.subject[f.train_idx][data.y[f.train_idx] == 2]))
                ),
                "n_eval_dem_subjects": int(
                    len(set(data.subject[f.eval_idx][data.y[f.eval_idx] == 2]))
                ),
                "expected_synthetic_rows": n_syn,
                "vae_fit_rows": n_dem,
                "preprocessing_fit_rows": int(len(f.train_idx)),
            }
        )
    return {
        "experiment": "B",
        "split": s,
        "fold_composition": desc.to_dict(orient="records"),
        "per_fold_plan": expected,
        "audit_mode": cfg.get("audit", {}).get("mode"),
        "scaler_scope": (cfg.get("preprocessing") or {}).get("scaler_scope"),
        "vae_fit_scope": (aug_cfg.get("vae") or {}).get("fit_scope"),
        "note": (
            "모든 fit(이상치·imputer·scaler·VAE)이 train fold 피험자로 제한된다. "
            "합성행은 train에만 추가되고 평가에는 절대 포함되지 않는다."
        ),
    }


def run_one_fold(
    data: LifelogData,
    fold,
    cfg: dict,
    *,
    auditor: LeakageAuditor,
    model_name: str,
    augmentation: str,
    seed: int,
    run_diagnostics: bool = False,
) -> dict:
    """사용자 지시 5절의 순서를 그대로 실행한다."""
    fold_id = fold.fold_id
    train_raw = data.take(fold.train_idx)
    eval_raw = data.take(fold.eval_idx)

    # (1) split은 호출자가 이미 등록했다.
    # (2)(3) 이상치·imputer fit ← train 피험자만
    pre_cfg = {"outlier": cfg.get("outlier"), "preprocessing": cfg.get("preprocessing")}
    pre = FoldPreprocessor(pre_cfg, auditor=auditor, fold_id=fold_id, seed=seed)
    pre.fit(train_raw)
    train_clean = pre.apply_outlier(train_raw)
    eval_clean = pre.apply_outlier_eval(eval_raw)   # 평가셋은 행을 지우지 않는다

    # (4) scaler fit ← train 피험자만
    pre.fit_scaler(train_clean)
    train_s = pre.transform(train_clean)
    eval_s = pre.transform(eval_clean)

    # (5)(6) VAE fit ← train fold의 실제 Dem만, 합성행은 train에만
    aug_cfg = dict(cfg.get("augmentation") or {})
    aug_cfg["method"] = augmentation
    aug = augment_train_fold(
        train_s, aug_cfg, auditor=auditor, fold_id=fold_id, preprocessor=pre, seed=seed
    )

    # (7) 분류기 학습
    model = fit_classifier(
        model_name, aug.data, cfg, auditor=auditor, fold_id=fold_id,
        class_weight=aug.class_weight, seed=seed,
    )

    # (8)(9) 평가 — 합성행 절대 미포함
    auditor.record_eval(
        fold_id, is_synthetic=eval_s.is_synthetic, subjects=eval_s.subject, where="outer_eval"
    )
    proba = model.predict_proba(eval_s.X.to_numpy())
    m_rec = compute_metrics(eval_s.y, proba, unit="record")
    subj = aggregate_to_subject(
        eval_s.subject, eval_s.y, proba,
        is_synthetic=eval_s.is_synthetic,
        method=(cfg.get("aggregate") or {}).get("method", "mean"),
    )
    m_sub = compute_metrics(subj.y, subj.proba, unit="subject")

    out = {
        "fold_id": fold_id,
        "model": model_name,
        "augmentation": augmentation,
        "n_synthetic": aug.n_synthetic,
        "augmentation_diagnostics": aug.diagnostics,
        "preprocess": pre.describe(),
        "record_level": m_rec,
        "subject_level": m_sub,
        "subject_predictions": subj.to_frame(),
        "n_dem_subjects_eval": int(m_sub["n_Dem"]),
        "n_dem_subjects_correct": int(m_sub["n_Dem_correct"]),
        "provenance": aug.data.provenance,
    }

    if run_diagnostics and aug.n_synthetic > 0 and aug.synthetic_X_raw is not None:
        eval_dem_raw = pre.inverse_transform_features(eval_s.X[eval_s.y == 2])
        out["synthetic_quality"] = synthetic_full_report(
            aug.synthetic_X_raw, aug.real_source_X_raw, eval_dem_raw
        )
        out["tstr"] = run_tstr(
            aug.data, eval_s, cfg, model_name=model_name,
            auditor=auditor, fold_id=fold_id, seed=seed,
        )
        out["trts"] = run_trts(
            train_s, aug.data.X[aug.data.is_synthetic], cfg, model_name=model_name,
            auditor=auditor, fold_id=fold_id, seed=seed,
        )
    return out


def run_experiment_b(
    data: LifelogData,
    cfg: dict,
    *,
    out_root: str,
    label: str,
    seed: int = 42,
    models: tuple[str, ...] = PAPER_MODEL_ORDER,
    augmentations: tuple[str, ...] = ("none", "vae"),
    only_fold: int | None = None,
) -> dict:
    """누수 통제 피험자 독립 비중첩 검증."""
    paths = RunPaths(out_root, f"B_{label}")
    auditor = LeakageAuditor(mode="enforce", name=f"B_{label}")

    s = _split_cfg(cfg)
    folds = make_group_folds(data, seed=seed, prefix="fold", **s)
    if only_fold is not None:
        folds = [f for f in folds if f.index == only_fold]
    save_table(describe_folds(data, folds), paths("fold_composition.csv"))

    # 모든 모델·증강조건이 **동일 split**을 쓴다.
    for f in folds:
        auditor.register_split(
            f.fold_id,
            train_subjects=data.subject[f.train_idx],
            eval_subjects=data.subject[f.eval_idx],
            train_row_ids=data.row_id[f.train_idx],
            eval_row_ids=data.row_id[f.eval_idx],
        )

    rows: list[dict] = []
    pooled: dict[tuple[str, str], dict] = {}
    diagnostics: dict[str, object] = {}
    run_diag = bool((cfg.get("diagnostics") or {}).get("enabled", True))

    for model_name in models:
        for aug in augmentations:
            fold_metrics, sub_frames = [], []
            for f in folds:
                r = run_one_fold(
                    data, f, cfg, auditor=auditor, model_name=model_name,
                    augmentation=aug, seed=seed,
                    run_diagnostics=run_diag and aug == "vae" and f.index == 0,
                )
                fold_metrics.append(r["subject_level"])
                sub_frames.append(r["subject_predictions"])
                rows.append(
                    {
                        "experiment": "B",
                        "config_label": label,
                        "fold_id": f.fold_id,
                        "model": model_name,
                        "augmentation": aug,
                        "n_synthetic": r["n_synthetic"],
                        "n_dem_subjects_eval": r["n_dem_subjects_eval"],
                        "n_dem_subjects_correct": r["n_dem_subjects_correct"],
                        **{f"subject_{k}": v for k, v in r["subject_level"].items()
                           if not isinstance(v, (list, dict))},
                        **{f"record_{k}": v for k, v in r["record_level"].items()
                           if not isinstance(v, (list, dict))},
                    }
                )
                if "synthetic_quality" in r:
                    diagnostics[f"{model_name}_{aug}_{f.fold_id}"] = {
                        k: r[k] for k in ("synthetic_quality", "tstr", "trts") if k in r
                    }
                if r["provenance"] is not None:
                    save_provenance(r["provenance"], paths("provenance", f"{f.fold_id}_{aug}"))

            # fold를 가로질러 피험자 예측을 모아 전체 지표를 만든다 (out-of-fold).
            allp = pd.concat(sub_frames, ignore_index=True)
            pcols = [c for c in allp.columns if c.startswith("proba_")]
            pooled[(model_name, aug)] = compute_metrics(
                allp["y_true"].to_numpy(), allp[pcols].to_numpy(), unit="subject"
            )
            pooled[(model_name, aug)]["fold_variability"] = fold_variability(
                fold_metrics
            ).to_dict(orient="records")
            if (cfg.get("bootstrap") or {}).get("enabled", True):
                pooled[(model_name, aug)]["bootstrap_ci"] = bootstrap_ci(
                    allp["y_true"].to_numpy(),
                    allp[pcols].to_numpy(),
                    n_boot=int((cfg.get("bootstrap") or {}).get("n_boot", 2000)),
                    seed=seed,
                )
            save_table(allp, paths("subject_predictions", f"{model_name}_{aug}.csv"),
                       also_markdown=False)

    metrics_df = pd.DataFrame(rows)
    save_table(metrics_df, paths("fold_metrics.csv"))
    save_json({f"{m}|{a}": v for (m, a), v in pooled.items()}, paths("pooled_subject_metrics.json"))
    if diagnostics:
        save_json(diagnostics, paths("synthetic_diagnostics.json"))
    auditor.assert_clean()
    auditor.save(paths("leakage_audit.json"))
    return {"results": pooled, "metrics_frame": metrics_df, "audit": auditor.summary(), "paths": paths}
