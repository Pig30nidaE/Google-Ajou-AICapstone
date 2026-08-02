"""Experiment orchestration: EDA -> screening -> nested selection -> reports.

Reporting contract
------------------
Every number this module writes is tagged with how it was produced:

``oof_nested``      model, blend and threshold chosen inside each outer fold.
                    This is the headline.
``oof_fixed``       repeated CV of one fixed configuration.  Honest per model,
                    but reading the table and picking the best row adds
                    arm-selection bias, so it is never the headline.
``non_nested_tuned``the Optuna phase.  Its objective saw every subject; the value
                    is reported *only* next to a nested re-evaluation of the same
                    configuration, to quantify the optimism.
``descriptive``     whole-cohort statistics (univariate AUCs, group means).  Not
                    a performance claim and not used to select anything.

There is no held-out test set: all 12 Dem subjects are needed for fitting, so
train and validation are pooled.  Nothing here may be described as external
validation.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import platform
import sys
import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import (
    BENCHMARK,
    EXPERIMENT_NAME,
    PROFILES,
    RunConfig,
    SUSPECT_FEATURE_PREFIXES,
    TASK_DESCRIPTION,
)
from .data import Cohort, hash_subject_id, load_cohort, quality_flags, resolve_block
from .engine import (
    Budget,
    CVResult,
    NestedResult,
    RuntimeBudgetExceeded,
    fold_fit_predict,
    nested_selection_cv,
    nested_threshold_metrics,
    run_repeated_cv,
)
from .ensemble import rank_normalize, safe_auc
from .evaluation import (
    bootstrap_auc,
    paired_bootstrap_delta,
    save_curves,
    subject_metrics,
    summarize_repeats,
    write_json,
    youden_threshold,
)
from .models import GOOGLE_MODELS, ModelSpec, TREE_BASELINES, environment_report
from .preprocessing import available_resamplers
from .splits import make_folds, split_summary
from .tuning import HAS_OPTUNA, candidates, default_model_pool, default_specs, grid, optuna_refine


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_logger(log_path) -> Callable[[str], None]:
    def log(message: str) -> None:
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        try:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:  # pragma: no cover - logging must never break a run
            pass

    return log


# ------------------------------------------------------------------- EDA -----
def descriptive_report(cohort: Cohort, *, top_n: int = 30) -> dict:
    """Whole-cohort descriptions.  Explicitly *not* used to select features.

    Univariate AUCs over all 174 subjects are reported for interpretation, the
    same way ``Binary_Google_DemScreen`` reported ``top_univariate``.  Feature
    selection inside the pipeline is recomputed in every training fold and never
    consults this table.
    """

    y = cohort.y
    rows = []
    for position, name in enumerate(cohort.feature_names):
        column = cohort.X[:, position]
        finite = np.isfinite(column)
        if finite.sum() < 0.5 * len(column) or np.std(column[finite]) < 1e-12:
            continue
        if len(np.unique(y[finite])) < 2:
            continue
        auc = float(roc_auc_score(y[finite], column[finite]))
        rows.append({"feature": name, "auc_direction_free": max(auc, 1.0 - auc),
                     "auc_signed": auc, "family": name.split("_")[0]})
    table = pd.DataFrame(rows).sort_values("auc_direction_free", ascending=False)

    by_family = {}
    for family, prefix in (("mmse", "mmse_"), ("wearable_daily", "wd_"), ("wearable_intraday", "wi_")):
        subset = table[table.feature.str.startswith(prefix)]
        by_family[family] = {
            "n_features": int(len(subset)),
            "best": subset.head(8)[["feature", "auc_direction_free"]].to_dict("records"),
        }

    return {
        "mode": "descriptive",
        "note": "Whole-cohort univariate AUCs, for interpretation only. Feature "
                "selection inside the pipeline is fold-local and ignores this table.",
        "cohort": {
            "n_subjects": cohort.n_subjects,
            "n_features": cohort.n_features,
            "counts": {d: int((cohort.diagnosis == d).sum()) for d in ("CN", "MCI", "Dem")},
            "n_positive": cohort.n_positive,
            "prevalence": float(cohort.y.mean()),
            "split_origin": {s: int((cohort.split_of == s).sum()) for s in ("train", "val")},
        },
        "top_features": table.head(top_n).to_dict("records"),
        "by_family": by_family,
    }


def reference_baselines(cohort: Cohort) -> dict:
    """Zero-parameter reference points that need no fitting.

    ``mmse_TOTAL`` alone is the number every model in this folder has to beat: it
    requires no training, so its whole-cohort AUC is not optimistic in the way a
    fitted model's would be (the only judgement embedded in it is the clinically
    pre-determined direction, lower score = more impaired).
    """

    out: dict[str, dict] = {"mode": "descriptive"}
    names = set(cohort.feature_names)
    for feature in ("mmse_TOTAL", "mmse_recall", "mmse_orient_time"):
        if feature not in names:
            continue
        column = cohort.X[:, cohort.feature_names.index(feature)]
        finite = np.isfinite(column)
        out[feature] = {
            "roc_auc_negated": float(roc_auc_score(cohort.y[finite], -column[finite])),
            "n_used": int(finite.sum()),
        }
    return out


# --------------------------------------------------------------- screening ---
def screen(cohort: Cohort, folds, models, tracks, *, resampler: str, seed: int,
           budget: Budget, log: Callable[[str], None],
           extra_specs: list | None = None) -> list[CVResult]:
    """Repeated CV for every (model, track) pair on identical folds.

    ``extra_specs`` carries specs that cannot be built from a name alone -- the
    TSMixer arm, which is bound to a cohort-aligned sequence tensor.
    """

    results: list[CVResult] = []
    for track in tracks:
        names = resolve_block(cohort, track)
        if not names:
            log(f"screen: track {track} is empty, skipped")
            continue
        block = cohort.select(names)
        specs = [ModelSpec(name=m, params=grid(m, level="single")[0]) for m in models]
        specs += list(extra_specs or [])
        for spec in specs:
            model = spec.name
            try:
                result = run_repeated_cv(
                    block.X, block.y, block.subject_ids, folds, spec, block=track,
                    resampler=resampler, seed=seed, budget=budget, on_error="record",
                )
            except RuntimeBudgetExceeded:
                raise
            except Exception as error:
                log(f"screen: {model}/{track} failed ({type(error).__name__}: {error})")
                continue
            summary = result.summary
            results.append(result)
            log(f"screen {track:>14s} {model:>16s} "
                f"AUC {summary['mean']:.4f} +- {summary['std']:.4f} "
                f"({result.elapsed_seconds:.1f}s, {len(names)} features)")
    return results


def screening_table(results: list[CVResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        summary = result.summary
        fold_aucs = [f["roc_auc"] for f in result.per_fold_auc if "roc_auc" in f]
        rows.append(
            {
                "model": result.model,
                "track": result.block,
                "resampler": result.resampler,
                "oof_roc_auc_mean": summary["mean"],
                "oof_roc_auc_std": summary["std"],
                "oof_roc_auc_min": summary["min"],
                "oof_roc_auc_max": summary["max"],
                "fold_auc_std": float(np.std(fold_aucs)) if fold_aucs else float("nan"),
                "n_repeats": summary["n"],
                "mean_features": result.mean_features,
                "elapsed_seconds": result.elapsed_seconds,
                "evidence": "oof_fixed",
            }
        )
    return pd.DataFrame(rows).sort_values("oof_roc_auc_mean", ascending=False)


# ------------------------------------------------------------------ output ---
def _oof_frame(cohort: Cohort, columns: dict[str, np.ndarray]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "subject_hash": [hash_subject_id(s) for s in cohort.subject_ids],
            "y_dem": cohort.y,
            "diagnosis": cohort.diagnosis,
            "origin_split": cohort.split_of,
        }
    )
    for name, values in columns.items():
        frame[name] = np.asarray(values, dtype=np.float64)
    return frame


def _fold_frame(results: list[CVResult], nested: dict[str, NestedResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        for record in result.per_fold_auc:
            rows.append({"arm": result.label, "evidence": "oof_fixed", **record})
    for label, result in nested.items():
        for record in result.per_fold_auc:
            rows.append({"arm": label, "evidence": "oof_nested", **record})
    return pd.DataFrame(rows)


def _feature_importance(cohort: Cohort, track: str, model: str, folds, *, seed: int) -> pd.DataFrame:
    """Mean fold-level importance for one model, refit per fold.

    Descriptive: importances describe what the model used, they are not a
    performance claim and are not used to select anything.
    """

    names = list(resolve_block(cohort, track))
    if not names:
        return pd.DataFrame()
    block = cohort.select(names)
    spec = ModelSpec(name=model, params=grid(model, level="single")[0])
    totals = np.zeros(len(names), dtype=np.float64)
    counted = 0
    for fold in folds[: min(len(folds), 10)]:
        try:
            outcome = fold_fit_predict(block.X, block.y, fold.train_index, fold.test_index,
                                       spec, seed=seed + fold.repeat, keep_model=True)
        except Exception:
            continue
        importance = outcome.model.importance() if outcome.model is not None else None
        if importance is None or importance.size != outcome.selected.size:
            continue
        totals[outcome.selected] += np.abs(importance)
        counted += 1
    if counted == 0:
        return pd.DataFrame()
    return (
        pd.DataFrame({"feature": names, "mean_abs_importance": totals / counted})
        .sort_values("mean_abs_importance", ascending=False)
    )


# ------------------------------------------------------------ the experiment --
def run_experiment(config: RunConfig) -> dict:
    started = time.monotonic()
    profile = config.profile_config
    training_dir = config.training_dir
    training_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(training_dir / "run.log")
    budget = Budget(deadline=started + float(config.hard_runtime_seconds))

    log(f"{EXPERIMENT_NAME} | profile={profile.name} | {TASK_DESCRIPTION}")
    environment = environment_report()
    log(f"environment: {environment}")

    cohorts: dict[str, Cohort] = {}
    base = load_cohort(config.data_root, drop_suspect=config.drop_suspect)
    cohorts["full"] = base
    flags = quality_flags(base)
    if config.cohort in ("filtered", "both") and flags:
        keep = np.array([s not in flags for s in base.subject_ids])
        if int(base.y[keep].sum()) >= 4:
            cohorts["filtered"] = base.subset(keep)
    if config.cohort == "filtered" and "filtered" in cohorts:
        cohorts.pop("full")

    log(f"cohort: {base.n_subjects} subjects, {base.n_features} features, "
        f"{base.n_positive} Dem; quality-flagged {len(flags)}")

    models = list(default_model_pool(config.models))
    resampler = config.resamplers[0] if config.resamplers else "class_weight"
    report: dict = {
        "experiment": EXPERIMENT_NAME,
        "task": TASK_DESCRIPTION,
        "positive_class": "Dem",
        "started_utc": _now(),
        "profile": profile.name,
        "profile_description": profile.description,
        "config": {**asdict(config), "models_resolved": models},
        "environment": {
            **environment,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "optuna": HAS_OPTUNA,
            "resamplers_available": list(available_resamplers()),
        },
        "benchmark_prior_best": BENCHMARK,
        "evidence_types": {
            "oof_nested": "model, blend and threshold chosen inside each outer fold (headline)",
            "oof_fixed": "repeated CV of a fixed configuration; arm-selection bias remains",
            "non_nested_tuned": "objective saw every subject; optimism reported alongside",
            "descriptive": "whole-cohort statistics; selects nothing",
        },
        "has_holdout_test_set": False,
        "arms": {},
    }

    # The primary cohort runs first so that exhausting the wall-clock budget
    # costs the sensitivity arm, not the headline.  A budget overrun must still
    # produce a report -- a six-hour run that ends with no artifacts is worse
    # than one that ends with a partial, clearly-labelled one.
    for cohort_name, cohort in cohorts.items():
        log(f"=== cohort arm: {cohort_name} (n={cohort.n_subjects}, Dem={cohort.n_positive}) ===")
        try:
            report["arms"][cohort_name] = run_cohort_arm(
                cohort, cohort_name, config, profile, models, resampler,
                budget=budget, log=log,
            )
        except RuntimeBudgetExceeded as error:
            log(f"cohort arm {cohort_name} stopped: {error}")
            report["arms"][cohort_name] = {"cohort": cohort_name, "status": "incomplete",
                                           "reason": str(error)}
            report["budget_exhausted"] = True
            break

    complete = {name: arm for name, arm in report["arms"].items() if "headline" in arm}
    if not complete:
        raise RuntimeError(
            "No cohort arm produced a nested headline; increase --hours or reduce "
            "--tracks/--models"
        )

    headline_arm = "full" if "full" in complete else next(iter(complete))
    headline = complete[headline_arm].get("headline", {})
    report["headline"] = {
        "cohort": headline_arm,
        "evidence": "oof_nested",
        **{k: v for k, v in headline.items() if k != "per_fold"},
    }
    prior = BENCHMARK["wearable_plus_mmse__full"]
    if headline.get("roc_auc_mean") is not None:
        report["improvement_vs_prior_best"] = {
            "prior_best_arm": "Binary_Google_DemScreen wearable_plus_mmse__full",
            "prior_best_roc_auc": prior,
            "this_run_roc_auc": headline["roc_auc_mean"],
            "delta": float(headline["roc_auc_mean"] - prior),
            "caveat": "Both are cross-validated OOF estimates on the same 174 subjects, "
                      "not independent test sets. With 12 positives the subject "
                      "bootstrap CI is far wider than this delta.",
        }

    report["limitations"] = [
        "양성(Dem) 12명. 어떤 점추정치도 subject bootstrap CI 없이 읽으면 안 된다.",
        "홀드아웃 테스트셋이 없다. 모든 수치는 174명 위의 교차검증(OOF) 결과이며 "
        "독립 외부 테스트가 아니다.",
        "MMSE 포함 결과는 부분적으로 순환적이다. MMSE는 이 진단을 내리는 검사 도구다.",
        "OOF AUC 차이가 0.05 수준이면 12 양성 표본에서는 통계적으로 구분되지 않는다. "
        "paired bootstrap CI가 0을 포함하면 개선으로 선언하지 않는다.",
        "screening 표(oof_fixed)에서 최고 행을 고르는 것 자체가 arm-selection bias이므로 "
        "headline은 nested 결과만 사용한다.",
    ]
    report["finished_utc"] = _now()
    report["wall_seconds"] = float(time.monotonic() - started)
    write_json(training_dir / "FINAL_REPORT.json", report)
    write_json(training_dir / "TRAINING_COMPLETE.json",
               {"status": "complete", "finished_utc": report["finished_utc"],
                "wall_seconds": report["wall_seconds"],
                "headline_roc_auc": report["headline"].get("roc_auc_mean")})
    log(f"done in {report['wall_seconds']:.0f}s -> {training_dir/'FINAL_REPORT.json'}")
    return report


def run_cohort_arm(cohort: Cohort, cohort_name: str, config: RunConfig, profile,
                   models: list[str], resampler: str, *, budget: Budget,
                   log: Callable[[str], None]) -> dict:
    """Everything measured for one cohort definition."""

    output = config.training_dir / cohort_name
    output.mkdir(parents=True, exist_ok=True)

    folds = make_folds(cohort.y, n_splits=profile.cv.outer_k, n_repeats=profile.cv.repeats,
                       seed=config.seed, min_positives_per_fold=profile.cv.min_positives_per_fold)
    screen_folds = [f for f in folds if f.repeat < profile.screen_repeats]
    log(f"folds: {split_summary(folds, cohort.y)}")

    arm: dict = {
        "cohort": cohort_name,
        "n_subjects": cohort.n_subjects,
        "n_positive": cohort.n_positive,
        "n_features": cohort.n_features,
        "feature_fingerprint": cohort.fingerprint,
        "split_summary": split_summary(folds, cohort.y),
        "eda": descriptive_report(cohort),
        "reference_baselines": reference_baselines(cohort),
    }

    tracks = [t for t in config.tracks if resolve_block(cohort, t)]
    arm["tracks"] = {t: len(resolve_block(cohort, t)) for t in tracks}

    # ---- optional TSMixer daily-sequence arm --------------------------------
    sequence_spec = None
    if config.use_sequence_arm or profile.use_sequence_arm:
        try:
            from .sequence import align_bundle, build_sequences, make_sequence_spec

            bundle = align_bundle(build_sequences(config.data_root), cohort.subject_ids)
            sequence_spec = make_sequence_spec(bundle, {})
            arm["sequence_arm"] = {
                "enabled": True,
                "tensor_shape": list(bundle.values.shape),
                "channels": list(bundle.channels),
                "observed_days_min": int(bundle.mask.sum(axis=1).min()),
                "observed_days_max": int(bundle.mask.sum(axis=1).max()),
                "note": "TSMixer over daily sequences + the track's features as static "
                        "covariates. Channel normalisation is fitted per fold.",
            }
            log(f"sequence arm enabled: {bundle.values.shape}")
        except Exception as error:
            arm["sequence_arm"] = {"enabled": False,
                                   "reason": f"{type(error).__name__}: {error}"}
            log(f"sequence arm unavailable ({type(error).__name__}: {error})")
    else:
        arm["sequence_arm"] = {"enabled": False, "reason": "not requested"}

    # ---- screening (oof_fixed) -------------------------------------------
    screen_results = screen(cohort, screen_folds, models, tracks, resampler=resampler,
                            seed=config.seed, budget=budget, log=log,
                            extra_specs=[sequence_spec] if sequence_spec else None)
    table = screening_table(screen_results)
    table.to_csv(output / "model_comparison.csv", index=False)
    arm["screening"] = {
        "evidence": "oof_fixed",
        "n_repeats": profile.screen_repeats,
        "table": table.to_dict("records"),
    }

    if table.empty:
        raise RuntimeError("Screening produced no results; check model availability")

    # Track choice for the headline is made on screening (oof_fixed) data, which
    # is an arm-selection decision; it is recorded as such.  The nested loop then
    # re-derives everything else inside folds.
    best_track = str(table.iloc[0]["track"])
    shortlist = _shortlist(table, best_track, profile.shortlist, models)
    arm["shortlist"] = {"track": best_track, "models": shortlist,
                        "selected_on": "oof_fixed screening table (arm-selection bias applies)"}
    log(f"shortlist on track {best_track}: {shortlist}")

    # ---- nested selection (headline) --------------------------------------
    nested: dict[str, NestedResult] = {}
    nested_families = _nested_families(models)
    nested_specs = candidates(nested_families, level="small")
    if sequence_spec is not None:
        # The sequence arm joins the nested candidates so that, if it wins, it
        # wins on the same terms as everything else -- and if it loses, the loss
        # is measured rather than assumed.
        nested_specs.append([sequence_spec])
        nested_families = nested_families + ["tsmixer"]
    arm["nested_candidate_families"] = {
        "families": nested_families,
        "n_variants": sum(len(v) for v in nested_specs),
        "selected_on": "pre-specified in config.NESTED_CANDIDATE_FAMILIES; not chosen "
                       "from this cohort's scores",
    }
    for track in _headline_tracks(tracks, best_track):
        names = resolve_block(cohort, track)
        block = cohort.select(names)
        log(f"nested selection on track {track} ({len(names)} features, "
            f"{len(nested_specs)} candidate families)")
        try:
            result = nested_selection_cv(
                block.X, block.y, block.subject_ids, folds, nested_specs, block=track,
                inner_k=profile.cv.inner_k, resampler=resampler, seed=config.seed,
                budget=budget, log=log,
            )
        except RuntimeBudgetExceeded:
            log(f"nested selection on {track} stopped: runtime budget exhausted")
            break
        nested[f"nested|{track}"] = result
        summary = result.summary
        log(f"nested {track}: AUC {summary['mean']:.4f} +- {summary['std']:.4f} "
            f"({result.elapsed_seconds:.0f}s) combiners={result.chosen_combiner}")

    # ---- assemble outputs --------------------------------------------------
    oof_columns: dict[str, np.ndarray] = {}
    for result in screen_results:
        if result.oof_by_repeat:
            oof_columns[f"fixed__{result.model}__{result.block}"] = result.mean_oof()
    for label, result in nested.items():
        if result.oof_by_repeat:
            oof_columns[label.replace("|", "__")] = result.mean_oof()
    _oof_frame(cohort, oof_columns).to_csv(output / "oof_predictions_hashed.csv", index=False)
    _fold_frame(screen_results, nested).to_csv(output / "fold_metrics.csv", index=False)

    arm["nested"] = {}
    for label, result in nested.items():
        summary = result.summary
        scores = result.mean_oof()
        arm["nested"][label] = {
            "evidence": "oof_nested",
            "track": result.block,
            "roc_auc_mean": summary["mean"],
            "roc_auc_std": summary["std"],
            "roc_auc_per_repeat": result.per_repeat_auc,
            "per_fold_roc_auc": [f["roc_auc"] for f in result.per_fold_auc],
            "bootstrap_95ci": bootstrap_auc(cohort.y, scores, n_boot=config.n_bootstrap,
                                            seed=config.seed),
            "metrics_at_nested_threshold": nested_threshold_metrics(cohort.y, result),
            "combiner_counts": result.chosen_combiner,
            "member_counts": result.chosen_members,
            "candidates": result.candidates,
            "elapsed_seconds": result.elapsed_seconds,
        }

    if nested:
        best_label = max(nested, key=lambda k: nested[k].summary["mean"])
        best = nested[best_label]
        scores = best.mean_oof()
        arm["headline"] = {
            "arm": best_label,
            "track": best.block,
            "evidence": "oof_nested",
            "roc_auc_mean": best.summary["mean"],
            "roc_auc_std": best.summary["std"],
            "roc_auc_per_repeat": best.per_repeat_auc,
            "per_fold_roc_auc": [f["roc_auc"] for f in best.per_fold_auc],
            "bootstrap_95ci": bootstrap_auc(cohort.y, scores, n_boot=config.n_bootstrap,
                                            seed=config.seed),
            "secondary_metrics": nested_threshold_metrics(cohort.y, best),
            "note": "Selected among nested tracks by nested OOF AUC; the track choice "
                    "itself carries arm-selection bias and is reported for every track.",
        }
        save_curves(cohort.y, scores, output, "headline",
                    title=f"{EXPERIMENT_NAME} / {cohort_name} / {best_label} (OOF, nested)")
        arm["paired_comparisons"] = _paired_comparisons(cohort, scores, screen_results,
                                                        seed=config.seed,
                                                        n_boot=config.n_bootstrap)

    # ---- feature importance (descriptive) ----------------------------------
    for model in _importance_models(shortlist):
        frame = _feature_importance(cohort, best_track, model, folds, seed=config.seed)
        if not frame.empty:
            frame.to_csv(output / f"feature_importance_{model}.csv", index=False)
    arm["feature_importance_note"] = (
        "Mean |importance| over the first 10 folds, refit per fold. Descriptive only."
    )

    # ---- Optuna phase (explicitly non-nested) ------------------------------
    if config.tune and profile.tune_trials > 0:
        arm["tuning"] = _tuning_phase(cohort, best_track, shortlist, screen_folds, folds,
                                      profile=profile, resampler=resampler, seed=config.seed,
                                      budget=budget, log=log)

    arm["suspect_features_present"] = sorted(
        n for n in cohort.feature_names if str(n).startswith(SUSPECT_FEATURE_PREFIXES)
    )
    return arm


def _nested_families(models: list[str]) -> list[str]:
    """Resolve the pre-specified nested candidate list against what is installed."""

    from .config import NESTED_CANDIDATE_FAMILIES, NESTED_FALLBACKS

    available = set(models)
    resolved: list[str] = []
    for family in NESTED_CANDIDATE_FAMILIES:
        if family in available:
            resolved.append(family)
            continue
        fallback = NESTED_FALLBACKS.get(family)
        if fallback and fallback in available:
            resolved.append(fallback)
    resolved = list(dict.fromkeys(resolved))
    if not resolved:  # pragma: no cover - a stock image always has several
        resolved = sorted(available)[:4]
    return resolved


def _headline_tracks(tracks: list[str], best_track: str) -> list[str]:
    """Tracks the nested loop runs on.

    Always includes the strongest screening track plus the two pre-specified
    reference representations, so the intraday contribution is measurable rather
    than assumed: ``wd_full`` is the prior best's representation and
    ``fused_core`` is the compact pre-specified set.
    """

    wanted = [best_track]
    for track in ("wd_full", "fused_core"):
        if track in tracks and track not in wanted:
            wanted.append(track)
    # Capped at three: a nested pass over the 556-feature ``wd_full`` costs about
    # 45 minutes at 20 repeats (measured), so a fourth track routinely pushes a
    # ``both``-cohort run past the six-hour limit.
    return wanted[:3]


def _shortlist(table: pd.DataFrame, track: str, size: int, models: list[str]) -> list[str]:
    subset = table[table.track == track].sort_values("oof_roc_auc_mean", ascending=False)
    picked = list(dict.fromkeys(subset.model.tolist()))[: max(2, size)]
    # Always carry at least one Google model and one strong tree baseline so the
    # required comparisons happen even if screening ranked them lower.
    for required in (GOOGLE_MODELS, TREE_BASELINES):
        if not any(name in picked for name in required):
            available = [name for name in required if name in models]
            if available:
                ranked = subset[subset.model.isin(available)]
                picked.append(str(ranked.iloc[0]["model"]) if len(ranked) else available[0])
    return list(dict.fromkeys(picked))


def _importance_models(shortlist: list[str]) -> list[str]:
    preferred = [m for m in shortlist if m in
                 ("random_forest", "extra_trees", "hist_gb", "lightgbm", "xgboost",
                  "catboost", "logreg_en", "logreg_l2", "ydf_gbt", "ydf_rf")]
    return preferred[:3] or shortlist[:1]


def _paired_comparisons(cohort: Cohort, headline_scores: np.ndarray,
                        screen_results: list[CVResult], *, seed: int, n_boot: int) -> dict:
    """Paired bootstrap of the headline against the strongest fixed baselines."""

    out: dict = {
        "note": "Paired subject bootstrap on identical subjects. A CI containing 0 "
                "means the difference is not resolved at this sample size.",
        "caveat": "The headline vector pools every repeat; the fixed-configuration "
                  "baselines pool only the screening repeats, so the two differ in "
                  "split-averaging as well as in method.",
    }
    by_model: dict[str, CVResult] = {}
    for result in screen_results:
        if not result.oof_by_repeat:
            continue
        key = result.model
        if key not in by_model or result.summary["mean"] > by_model[key].summary["mean"]:
            by_model[key] = result
    interesting = [name for name in ("univariate", "rank_mean", "logreg_en", "random_forest",
                                     "lightgbm", "catboost", "ydf_gbt", "ydf_rf", "tabnet")
                   if name in by_model]
    for name in interesting:
        baseline = by_model[name]
        out[f"vs_{name}__{baseline.block}"] = paired_bootstrap_delta(
            cohort.y, headline_scores, baseline.mean_oof(), n_boot=n_boot, seed=seed
        )
    if "mmse_TOTAL" in cohort.feature_names:
        column = cohort.X[:, cohort.feature_names.index("mmse_TOTAL")]
        filled = np.where(np.isfinite(column), column, np.nanmedian(column))
        out["vs_mmse_total_single_feature"] = paired_bootstrap_delta(
            cohort.y, headline_scores, -filled, n_boot=n_boot, seed=seed
        )
    return out


def _tuning_phase(cohort: Cohort, track: str, shortlist: list[str], screen_folds, folds,
                  *, profile, resampler: str, seed: int, budget: Budget,
                  log: Callable[[str], None]) -> dict:
    """Optuna refinement, reported with its optimism gap.

    The objective is repeated-CV OOF AUC over the whole cohort, maximised over
    many trials, so its best value is optimistic in two distinct ways:

    ``fold_reuse``  every trial was scored on the *same* folds, so the winner is
        partly a fit to those particular splits.  Measured exactly by re-scoring
        the winning configuration on folds drawn from a different seed.
    ``selection``   the configuration itself was chosen with all 174 subjects in
        view.  Measured by running the same model's grid *inside* each outer
        fold, which is what the headline already does.

    ``Binary_Google_MaxAUC_Tuned`` measured a combined +0.084 on the neighbouring
    task; reporting both components is the point of this phase, not the tuned
    score itself.
    """

    from .tuning import has_search_space

    names = resolve_block(cohort, track)
    block = cohort.select(names)
    out: dict = {"evidence": "non_nested_tuned", "track": track, "models": {}}

    for model in shortlist[:2]:
        if not has_search_space(model):
            continue

        def objective(params: dict, _model=model) -> float:
            try:
                result = run_repeated_cv(
                    block.X, block.y, block.subject_ids, screen_folds,
                    ModelSpec(name=_model, params=params), block=track,
                    resampler=resampler, seed=seed, budget=budget, on_error="record",
                )
            except RuntimeBudgetExceeded:
                raise
            except Exception:
                return 0.5
            return float(result.summary["mean"]) if result.per_repeat_auc else 0.5

        try:
            study = optuna_refine(model, objective, n_trials=profile.tune_trials,
                                  seed=seed, log=log)
        except RuntimeBudgetExceeded:
            log("tuning stopped: runtime budget exhausted")
            break
        if "best_params" not in study:
            continue

        # (a) same tuned configuration, folds redrawn from a different seed.
        fresh_folds = make_folds(cohort.y, n_splits=profile.cv.outer_k,
                                 n_repeats=len({f.repeat for f in screen_folds}),
                                 seed=seed + 991,
                                 min_positives_per_fold=profile.cv.min_positives_per_fold)
        fold_reuse_value = None
        try:
            rechecked = run_repeated_cv(
                block.X, block.y, block.subject_ids, fresh_folds,
                ModelSpec(name=model, params=study["best_params"]), block=track,
                resampler=resampler, seed=seed, budget=budget, on_error="record",
            )
            fold_reuse_value = rechecked.summary["mean"] if rechecked.per_repeat_auc else None
        except RuntimeBudgetExceeded:
            raise
        except Exception as error:
            log(f"tuning: fresh-fold re-check for {model} failed "
                f"({type(error).__name__}: {error})")

        # (b) the same model's grid searched *inside* each outer fold.
        selection_value = None
        try:
            nested_grid = nested_selection_cv(
                block.X, block.y, block.subject_ids, screen_folds,
                [candidates([model], level="small")[0]], block=track,
                inner_k=profile.cv.inner_k, resampler=resampler, seed=seed, budget=budget,
            )
            selection_value = nested_grid.summary["mean"]
        except RuntimeBudgetExceeded:
            raise
        except Exception as error:
            log(f"tuning: in-fold grid re-check for {model} failed "
                f"({type(error).__name__}: {error})")

        study.pop("history", None)
        best_value = float(study["best_value_non_nested"])
        out["models"][model] = {
            **study,
            "recheck_fresh_folds_roc_auc": fold_reuse_value,
            "optimism_from_fold_reuse": (None if fold_reuse_value is None
                                         else float(best_value - fold_reuse_value)),
            "recheck_in_fold_grid_roc_auc": selection_value,
            "optimism_from_selection": (None if selection_value is None
                                        else float(best_value - selection_value)),
            "interpretation": "best_value_non_nested is NOT a performance claim. The two "
                              "optimism figures are the reportable quantities: how much of "
                              "it came from reusing the same folds, and how much from "
                              "choosing the configuration with every subject in view.",
        }
    return out


__all__ = ["descriptive_report", "reference_baselines", "run_cohort_arm", "run_experiment",
           "screen", "screening_table"]
