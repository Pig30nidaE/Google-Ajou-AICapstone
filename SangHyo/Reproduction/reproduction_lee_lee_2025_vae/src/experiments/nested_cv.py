"""실험 C — ``nested_subject_independent``.

Outer 3-fold × n_repeats / Inner 3-fold, group = 피험자 ID.

**전체 파이프라인 선택이 inner CV 내부에서만 이루어진다.** 이상치 방식·임계값,
VAE 사용 여부·latent·synthetic ratio, 분류기와 그 하이퍼파라미터가 모두 후보이며,
outer test는 선택의 어느 단계에도 관여하지 않는다.

탐색은 config의 ``search.space``로 제한하고 ``search.max_evals``로 상한을 둔다.
상한을 넘으면 ``classifier × augmentation`` arm을 균형 순회하고 arm 내부만
seed 기반으로 무작위화한다(Optuna를 쓰지 않는다 — 사용자 지시 15절).
"""

from __future__ import annotations

import itertools
import json
import logging

import numpy as np
import pandas as pd

from ..audit.leakage import LeakageAuditor
from ..data.loader import LifelogData
from ..evaluation.aggregate import aggregate_to_subject
from ..evaluation.bootstrap import bootstrap_ci
from ..evaluation.metrics import compute_metrics
from ..evaluation.tables import fold_variability
from ..utils.io import RunPaths, save_json, save_table
from .leakage_controlled import run_one_fold
from ..splits.group_cv import describe_folds, make_group_folds

log = logging.getLogger(__name__)

__all__ = ["run_experiment_c", "plan_experiment_c", "enumerate_candidates"]


def _canonicalize_candidate(candidate: dict) -> dict:
    """비활성 분기의 하이퍼파라미터를 제거해 실제 파이프라인 단위로 정규화한다.

    예를 들어 ``augmentation.method=smote``인 후보 두 개가 VAE latent 값만 다르면
    실제 실행은 완전히 같다. 이런 명목상 후보가 탐색 예산을 중복 소비하지 않게 한다.
    """
    out = dict(candidate)
    augmentation = out.get("augmentation.method")
    if augmentation is not None and augmentation != "vae":
        for key in tuple(out):
            if key.startswith("augmentation.vae."):
                out.pop(key)

    outlier = out.get("outlier.method")
    if outlier is not None and outlier != "percentile":
        for key in tuple(out):
            if key.startswith("outlier.percentile."):
                out.pop(key)
    if outlier is not None and outlier != "isolation_forest":
        for key in tuple(out):
            if key.startswith("outlier.isolation_forest."):
                out.pop(key)
    return out


def _candidate_key(candidate: dict) -> str:
    return json.dumps(candidate, sort_keys=True, ensure_ascii=False, default=str)


def _candidate_coverage(candidates: list[dict]) -> list[dict]:
    counts: dict[tuple[str, str], int] = {}
    for candidate in candidates:
        key = (
            str(candidate.get("classifier", "__base__")),
            str(candidate.get("augmentation.method", "__base__")),
        )
        counts[key] = counts.get(key, 0) + 1
    return [
        {"classifier": classifier, "augmentation": augmentation, "n_candidates": count}
        for (classifier, augmentation), count in sorted(counts.items())
    ]


def enumerate_candidates(space: dict, *, max_evals: int, seed: int) -> list[dict]:
    """탐색 공간에서 후보 설정을 만든다.

    비활성 하위 축을 제거한 **유효 파이프라인**을 중복 제거한다. 후보가 상한보다
    많으면 ``classifier × augmentation`` arm별로 round-robin 표집해 특정 모델의
    VAE 후보가 우연히 전부 빠지는 일을 막는다. 각 arm 내부 순서는 seed가 고정된
    무작위 순서다.
    """
    if not space:
        return [{}]
    if max_evals <= 0:
        raise ValueError("search.max_evals는 1 이상이어야 한다")
    keys = sorted(space)
    grids = [list(space[k]) for k in keys]
    if any(not grid for grid in grids):
        raise ValueError("search.space의 각 축은 후보를 하나 이상 가져야 한다")

    unique: dict[str, dict] = {}
    for combo in itertools.product(*grids):
        candidate = _canonicalize_candidate(dict(zip(keys, combo)))
        unique.setdefault(_candidate_key(candidate), candidate)
    candidates = list(unique.values())
    if len(candidates) <= max_evals:
        return candidates

    rng = np.random.default_rng(seed)
    grouped: dict[tuple[str, str], list[dict]] = {}
    for candidate in candidates:
        arm = (
            str(candidate.get("classifier", "__base__")),
            str(candidate.get("augmentation.method", "__base__")),
        )
        grouped.setdefault(arm, []).append(candidate)
    classifier_order = ["xgboost", "dnn", "tabnet", "wide_deep", "__base__"]
    augmentation_order = ["none", "vae", "class_weight", "random_oversampling", "smote", "__base__"]
    active_classifiers = [
        classifier
        for classifier in classifier_order
        if any(key[0] == classifier for key in grouped)
    ]
    active_augmentations = [
        augmentation
        for augmentation in augmentation_order
        if any(key[1] == augmentation for key in grouped)
    ]
    # Latin-square 순서: 어떤 prefix를 잘라도 classifier와 augmentation 양쪽의
    # 후보 수 차이가 가능한 한 1 이내가 되게 한다. max_evals=24에서 앞 모델 두 개만
    # 두 배의 탐색 기회를 얻었던 고정 중첩 순서 편향을 막는다.
    arms: list[tuple[str, str]] = []
    for offset in range(len(active_augmentations)):
        for i, classifier in enumerate(active_classifiers):
            augmentation = active_augmentations[(i + offset) % len(active_augmentations)]
            arm = (classifier, augmentation)
            if arm in grouped:
                arms.append(arm)
    arms += sorted(set(grouped) - set(arms))
    for arm in arms:
        order = rng.permutation(len(grouped[arm]))
        grouped[arm] = [grouped[arm][int(i)] for i in order]

    # VAE arm은 latent_dim × ratio_to_real 때문에 후보가 6배 많다. 단순 round-robin은
    # 다른 arm이 소진된 뒤 남은 예산을 전부 VAE에 몰아줘(24 예산에서 vae 12 대 none 4)
    # "여러 번 시도해서 이긴" 선택 편향을 만든다. 이 연구의 질문이 바로 "VAE 증강이
    # 도움이 되는가"이므로, 예산을 다 쓰는 것보다 arm 간 시행 횟수를 맞추는 것이 우선이다.
    # 따라서 두 축(classifier·augmentation)의 시행 수 편차가 1을 넘지 않는 선에서만
    # 후보를 채우고, 그 이상은 예산이 남아도 뽑지 않는다.
    out: list[dict] = []
    cursor = {arm: 0 for arm in arms}
    classifier_used = {classifier: 0 for classifier in active_classifiers}
    augmentation_used = {augmentation: 0 for augmentation in active_augmentations}
    while len(out) < max_evals:
        progressed = False
        for classifier, augmentation in arms:
            if len(out) >= max_evals:
                break
            arm = (classifier, augmentation)
            i = cursor[arm]
            if i >= len(grouped[arm]):
                continue
            if classifier_used[classifier] + 1 - min(classifier_used.values()) > 1:
                continue
            if augmentation_used[augmentation] + 1 - min(augmentation_used.values()) > 1:
                continue
            out.append(grouped[arm][i])
            cursor[arm] += 1
            classifier_used[classifier] += 1
            augmentation_used[augmentation] += 1
            progressed = True
        if not progressed:
            break
    return out


def _apply_candidate(base_cfg: dict, cand: dict) -> tuple[dict, str, str]:
    """후보의 점 표기 키를 config에 적용한다.

    Returns:
        (적용된 config, 선택된 model 이름, 선택된 augmentation 이름).
    """
    import copy

    cfg = copy.deepcopy(base_cfg)
    model = cand.get("classifier", (base_cfg.get("search") or {}).get("default_classifier", "xgboost"))
    aug = cand.get("augmentation.method", (base_cfg.get("augmentation") or {}).get("method", "none"))
    for key, val in cand.items():
        if key in ("classifier",):
            continue
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    cfg.setdefault("augmentation", {})["method"] = aug
    return cfg, model, aug


def plan_experiment_c(data: LifelogData, cfg: dict, *, seed: int) -> dict:
    """--dry-run: outer/inner fold 구성과 탐색 규모를 확인한다."""
    sp = cfg.get("split") or {}
    outer_cfg = sp.get("outer") or {}
    inner_cfg = sp.get("inner") or {}
    outer = make_group_folds(
        data,
        method=outer_cfg.get("method", "subject_stratified"),
        n_splits=int(outer_cfg.get("n_splits", 3)),
        n_repeats=int(outer_cfg.get("n_repeats", 1)),
        seed=seed,
        prefix="outer",
    )
    search = cfg.get("search") or {}
    space = search.get("space") or {}
    cands = enumerate_candidates(
        space, max_evals=int(search.get("max_evals", 20)), seed=seed
    )
    nominal_count = int(np.prod([len(values) for values in space.values()])) if space else 1
    effective_count = len(
        enumerate_candidates(space, max_evals=max(nominal_count, 1), seed=seed)
    )

    inner_summary = []
    for f in outer:
        tr = data.take(f.train_idx)
        inner = make_group_folds(
            tr,
            method=inner_cfg.get("method", "subject_stratified"),
            n_splits=int(inner_cfg.get("n_splits", 3)),
            n_repeats=1,
            seed=seed,
            prefix=f"{f.fold_id}_inner",
        )
        inner_summary.append(
            {
                "outer_fold": f.fold_id,
                "n_train_subjects": int(len(set(data.subject[f.train_idx]))),
                "n_eval_subjects": int(len(set(data.subject[f.eval_idx]))),
                "n_eval_dem_subjects": int(
                    len(set(data.subject[f.eval_idx][data.y[f.eval_idx] == 2]))
                ),
                "n_inner_folds": len(inner),
                "inner_composition": describe_folds(tr, inner).to_dict(orient="records"),
            }
        )
    return {
        "experiment": "C",
        "n_outer_folds": len(outer),
        "n_candidates": len(cands),
        "n_nominal_candidates_before_conditioning": nominal_count,
        "n_effective_candidates_after_conditioning": effective_count,
        "candidate_coverage": _candidate_coverage(cands),
        "max_evals": int(search.get("max_evals", 20)),
        "total_model_fits": len(outer) * len(cands) * int(inner_cfg.get("n_splits", 3)) + len(outer),
        "candidates_preview": cands[:10],
        "outer_composition": describe_folds(data, outer).to_dict(orient="records"),
        "inner_summary": inner_summary,
        "note": (
            "outer test는 이상치 임계값·scaler·VAE·synthetic ratio·모델·early stopping·"
            "threshold 선택 어디에도 사용되지 않는다. 감사기가 record_selection()으로 강제한다."
        ),
    }


def run_experiment_c(
    data: LifelogData,
    cfg: dict,
    *,
    out_root: str,
    label: str,
    seed: int = 42,
    only_fold: int | None = None,
    restrict_classifier: str | None = None,
    restrict_augmentation: str | None = None,
    max_evals: int | None = None,
) -> dict:
    """피험자 독립 반복 Nested Group CV.

    Args:
        restrict_classifier: 지정하면 분류기를 고정하고 나머지(전처리·증강 강도 등)만
            inner CV에서 탐색한다. 결과표의 모델별 칸을 채울 때 쓴다.
        restrict_augmentation: 지정하면 증강 조건도 고정한다. ``restrict_classifier``와
            함께 쓰면 결과표의 (모델 × 증강) 칸 하나에 대응하는 nested 결과가 된다.
        max_evals: inner 탐색 후보 수 상한 override (per-cell 실행 시 비용 절감용).

    두 restrict 인자를 모두 주면 "이 모델·이 증강으로 고정했을 때, 전처리와
    하이퍼파라미터만 inner CV에서 고른 nested 성능"이 된다. 선택이 여전히 inner
    안에만 갇혀 있으므로 nested의 성질은 유지된다.
    """
    paths = RunPaths(out_root, f"C_{label}")
    auditor = LeakageAuditor(mode="enforce", name=f"C_{label}")

    sp = cfg.get("split") or {}
    outer_cfg, inner_cfg = sp.get("outer") or {}, sp.get("inner") or {}
    search = cfg.get("search") or {}
    selection_metric = search.get("metric", "macro_f1")

    outer = make_group_folds(
        data,
        method=outer_cfg.get("method", "subject_stratified"),
        n_splits=int(outer_cfg.get("n_splits", 3)),
        n_repeats=int(outer_cfg.get("n_repeats", 1)),
        seed=seed,
        prefix="outer",
    )
    if only_fold is not None:
        outer = [f for f in outer if f.index == only_fold]
    save_table(describe_folds(data, outer), paths("outer_fold_composition.csv"))

    space = dict(search.get("space") or {})
    if restrict_classifier is not None:
        space["classifier"] = [restrict_classifier]
    if restrict_augmentation is not None:
        space["augmentation.method"] = [restrict_augmentation]
        if restrict_augmentation != "vae":
            # 증강을 vae가 아닌 것으로 고정하면 VAE 전용 축은 탐색할 이유가 없다.
            space.pop("augmentation.vae.latent_dim", None)
            space.pop("augmentation.vae.ratio_to_real", None)
    candidates = enumerate_candidates(
        space,
        max_evals=int(max_evals if max_evals is not None else search.get("max_evals", 20)),
        seed=seed,
    )
    log.info(
        "nested CV[%s]: outer %d folds × %d candidates%s",
        label, len(outer), len(candidates),
        f" (classifier={restrict_classifier}, augmentation={restrict_augmentation})"
        if restrict_classifier or restrict_augmentation else "",
    )

    inner_log: list[dict] = []
    outer_rows: list[dict] = []
    sub_frames: list[pd.DataFrame] = []
    chosen: list[dict] = []

    for of in outer:
        auditor.register_split(
            of.fold_id,
            train_subjects=data.subject[of.train_idx],
            eval_subjects=data.subject[of.eval_idx],
            train_row_ids=data.row_id[of.train_idx],
            eval_row_ids=data.row_id[of.eval_idx],
        )
        outer_train = data.take(of.train_idx)

        # ---------- inner CV: 여기서만 선택이 일어난다 ----------
        inner = make_group_folds(
            outer_train,
            method=inner_cfg.get("method", "subject_stratified"),
            n_splits=int(inner_cfg.get("n_splits", 3)),
            n_repeats=1,
            seed=seed,
            prefix=f"{of.fold_id}_inner",
        )
        inner_auditor = LeakageAuditor(mode="enforce", name=f"{of.fold_id}_inner")
        for f in inner:
            inner_auditor.register_split(
                f.fold_id,
                train_subjects=outer_train.subject[f.train_idx],
                eval_subjects=outer_train.subject[f.eval_idx],
                train_row_ids=outer_train.row_id[f.train_idx],
                eval_row_ids=outer_train.row_id[f.eval_idx],
            )

        scores: list[tuple[float, dict]] = []
        for ci, cand in enumerate(candidates):
            ccfg, model_name, aug_name = _apply_candidate(cfg, cand)
            vals = []
            for f in inner:
                r = run_one_fold(
                    outer_train, f, ccfg, auditor=inner_auditor,
                    model_name=model_name, augmentation=aug_name,
                    seed=seed, run_diagnostics=False,
                )
                vals.append(r["subject_level"].get(selection_metric, float("nan")))
            score = float(np.nanmean(vals)) if len(vals) else float("nan")
            scores.append((score, cand))
            inner_log.append(
                {
                    "outer_fold": of.fold_id,
                    "candidate_index": ci,
                    "classifier": model_name,
                    "augmentation": aug_name,
                    "candidate": cand,
                    f"inner_mean_{selection_metric}": round(score, 4),
                    "inner_scores": [round(float(v), 4) for v in vals],
                }
            )

        # 선택은 inner 자료만 보았다는 것을 감사기에 신고한다.
        auditor.record_selection(
            "pipeline", of.fold_id, subjects=outer_train.subject
        )
        best_score, best_cand = max(scores, key=lambda t: (-np.inf if np.isnan(t[0]) else t[0]))
        bcfg, best_model, best_aug = _apply_candidate(cfg, best_cand)
        chosen.append(
            {
                "outer_fold": of.fold_id,
                "selected": best_cand,
                "classifier": best_model,
                "augmentation": best_aug,
                f"inner_{selection_metric}": round(best_score, 4),
            }
        )
        log.info("[%s] 선택: %s (inner %s=%.4f)", of.fold_id, best_cand, selection_metric, best_score)

        # ---------- outer 평가: 선택된 파이프라인을 outer train에 재적합 ----------
        r = run_one_fold(
            data, of, bcfg, auditor=auditor,
            model_name=best_model, augmentation=best_aug,
            seed=seed, run_diagnostics=bool((cfg.get("diagnostics") or {}).get("enabled", True)),
        )
        sub_frames.append(r["subject_predictions"])
        outer_rows.append(
            {
                "experiment": "C",
                "config_label": label,
                "fold_id": of.fold_id,
                "selected_classifier": best_model,
                "selected_augmentation": best_aug,
                "selected_candidate": str(best_cand),
                "n_synthetic": r["n_synthetic"],
                "n_dem_subjects_eval": r["n_dem_subjects_eval"],
                "n_dem_subjects_correct": r["n_dem_subjects_correct"],
                **{f"subject_{k}": v for k, v in r["subject_level"].items()
                   if not isinstance(v, (list, dict))},
                **{f"record_{k}": v for k, v in r["record_level"].items()
                   if not isinstance(v, (list, dict))},
            }
        )

    allp = pd.concat(sub_frames, ignore_index=True)
    pcols = [c for c in allp.columns if c.startswith("proba_")]
    pooled = compute_metrics(allp["y_true"].to_numpy(), allp[pcols].to_numpy(), unit="subject")
    pooled["fold_variability"] = fold_variability(
        [{k.replace("subject_", ""): v for k, v in row.items() if k.startswith("subject_")}
         for row in outer_rows]
    ).to_dict(orient="records")
    if (cfg.get("bootstrap") or {}).get("enabled", True):
        pooled["bootstrap_ci"] = bootstrap_ci(
            allp["y_true"].to_numpy(), allp[pcols].to_numpy(),
            n_boot=int((cfg.get("bootstrap") or {}).get("n_boot", 2000)), seed=seed,
        )

    save_table(pd.DataFrame(outer_rows), paths("outer_fold_metrics.csv"))
    save_table(pd.DataFrame(inner_log), paths("inner_selection_log.csv"), also_markdown=False)
    save_table(pd.DataFrame(chosen), paths("selected_pipelines.csv"))
    save_table(allp, paths("subject_predictions.csv"), also_markdown=False)
    save_json(pooled, paths("pooled_subject_metrics.json"))
    auditor.assert_clean()
    auditor.save(paths("leakage_audit.json"))
    return {
        "results": pooled,
        "selected": chosen,
        "metrics_frame": pd.DataFrame(outer_rows),
        "audit": auditor.summary(),
        "paths": paths,
    }
