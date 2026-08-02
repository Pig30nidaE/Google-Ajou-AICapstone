"""Single entry point for the Hong et al. (2024) reproduction.

    USER_FOLDER = "SangHyo"
    RUN_FILE    = "Reproduction/reproduction_hong_2024/run.py"

Usage::

    python run.py --config configs/paper_temporal_5day.yaml
    python run.py --config configs/strict_same_subject_temporal.yaml
    python run.py --config configs/fixed_subject_independent.yaml
    python run.py --config configs/nested_subject_independent.yaml

Options::

    --inspect-data       paper-vs-data discrepancy report only, no split, no fit
    --audit-only         dataset-level leakage audit only
    --dry-run            splits, sequence counts, shapes and every audit; no fit
    --sequence-length N  restrict to one of 3 / 4 / 5
    --fold N             restrict to one outer fold
    --seed N             override config.seed
    --resume             reuse completed checkpoints
    --model NAME         restrict to one model (repeatable)
    --estimand A|B       refuse to run unless the config estimates this estimand
    --compare            rebuild the cross-experiment comparison tables

Recommended Colab runtime: **GPU** for the LSTM arms, **CPU / High-RAM** when only
the tree and linear baselines are enabled.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

EXPERIMENT_NAME = "reproduction_hong_2024"
EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[2]
REQUIREMENTS_FILE = EXPERIMENT_ROOT / "requirements_colab.txt"

for path in (str(EXPERIMENT_ROOT), str(REPOSITORY_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _resolve_data_root(namespace: dict[str, Any], explicit: str | None) -> Path:
    """Match the repository convention: PROJECT_ROOT/DATA_ROOT first, then Drive."""
    from src.data.loader import resolve_data_root

    candidates: list[str | Path] = []
    for value in (explicit, namespace.get("DATA_ROOT"), os.environ.get("SANGHYO_DATA_ROOT")):
        if value:
            candidates.append(value)
    project = namespace.get("PROJECT_ROOT")
    if project:
        candidates.append(Path(project) / "Data")
    candidates += [
        REPOSITORY_ROOT / "Data",
        Path("/content/drive/Shareddrives/GoogleAI_contest/Data"),
        Path("/content/drive/MyDrive/GoogleAI_contest/Data"),
    ]
    return resolve_data_root(candidates)


def _ensure_dependencies(skip_install: bool, *, need_torch: bool, need_xgboost: bool) -> dict[str, bool]:
    required = {"numpy": "numpy", "pandas": "pandas", "scikit-learn": "sklearn",
                "scipy": "scipy", "pyyaml": "yaml"}
    missing = [pkg for pkg, mod in required.items() if importlib.util.find_spec(mod) is None]
    if need_torch and importlib.util.find_spec("torch") is None:
        missing.append("torch")
    if need_xgboost and importlib.util.find_spec("xgboost") is None:
        missing.append("xgboost")

    if missing and not skip_install:
        print(f"[run] installing: {', '.join(missing)}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
             "-r", str(REQUIREMENTS_FILE)],
            check=True,
        )
        importlib.invalidate_caches()
    elif missing:
        raise ModuleNotFoundError(f"missing dependencies: {', '.join(missing)}")

    return {
        "torch": importlib.util.find_spec("torch") is not None,
        "xgboost": importlib.util.find_spec("xgboost") is not None,
    }


# --- modes --------------------------------------------------------------------

def mode_inspect_data(data_root: Path, output_dir: Path) -> dict[str, Any]:
    from src.data.inspect import inspect_data, render_discrepancy_markdown
    from src.utils.io import write_json, write_text

    report = inspect_data(data_root)
    write_json(output_dir / "inspection" / "data_inspection.json", report)
    write_text(
        output_dir / "inspection" / "discrepancy_report.md",
        render_discrepancy_markdown(report),
    )

    print(f"\n=== 데이터 점검: {data_root} ===")
    for check in report["checks"]:
        mark = "OK  " if check["agrees"] else "DIFF"
        print(f"  [{mark}] {check['description']}: 논문={check['paper']} 실측={check['measured']}")
    print(f"\n불일치 {report['n_disagreements']}건 "
          f"(high: {report['high_severity_disagreements'] or '없음'})")
    print("\n  파생변수 공식 검증:")
    for item in report["formula_verification"]:
        print(f"    - {item['feature']:24s} 일치율 {item['agreement']:.3f} → {item['verdict']}")
    collisions = report["day_key_collisions"]
    print(f"\n  같은 날짜 충돌: bedtime_end {collisions['bedtime_end']}행 / "
          f"bedtime_start {collisions['bedtime_start']}행 → "
          f"{collisions['chosen']} 사용, {collisions['n_rows_dropped_by_dedup']}행 제거")
    gaps = report["calendar_gaps"]
    print(f"  달력 공백이 있는 피험자: {gaps['n_subjects_with_any_gap']}명 / 공백 {gaps['total_gaps']}건")
    for pair in report["identical_feature_pairs"]:
        print(f"  주의: {pair['left']} == {pair['right']} (완전히 동일한 열)")
    print(f"\n  보고서: {output_dir / 'inspection' / 'discrepancy_report.md'}")
    return report


def mode_audit_only(data, output_dir: Path) -> dict[str, Any]:
    from src.audit.leakage import audit_dataset
    from src.utils.io import write_json

    log = audit_dataset(data)
    summary = log.summary()
    write_json(output_dir / "audit" / "dataset_audit.json", summary)
    print("\n=== 데이터셋 누수 감사 ===")
    for check in summary["checks"]:
        print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {check['check']}")
    print(f"\n전체 통과: {summary['all_passed']}")
    return summary


def mode_dry_run(data, config, output_dir: Path, device: str, deps: dict[str, bool],
                 *, only_length: int | None = None) -> dict[str, Any]:
    """Everything except fitting: splits, sequence counts, shapes, audits, budget."""
    import numpy as np

    from src.audit import leakage
    from src.evaluation.compare import verify_paper_arithmetic
    from src.models import registry
    from src.preprocessing.scaler import SequenceScaler
    from src.sampling.undersample import undersample
    from src.sequences.builder import build_sequences, build_sequences_literal
    from src.splits import group as group_splits
    from src.splits import temporal as temporal_splits
    from src.utils.io import write_json

    lengths = [only_length] if only_length else list(config.sequence_lengths)
    report: dict[str, Any] = {
        "experiment": config.experiment,
        "estimand": config.estimand,
        "config_path": str(config.path),
        "seed": config.seed,
        "device": device,
        "dependencies": {**deps, **registry.dependency_report()},
        "data": {**data.describe(), "notes": data.notes},
        "paper_arithmetic_check": verify_paper_arithmetic(),
        "sequence_lengths": lengths,
    }

    dataset_audit = leakage.audit_dataset(data)
    dataset_audit.raise_if_failed()
    report["dataset_audit"] = dataset_audit.summary()

    per_length: list[dict[str, Any]] = []
    audit_ok = True
    stride = int(config.get("sequence.stride", 1))
    probe_model = config.models[0]

    for length in lengths:
        entry: dict[str, Any] = {"sequence_length": length}

        if config.split_mode in ("final_week_temporal", "final_week_temporal_literal"):
            strict = config.experiment == "strict_same_subject_temporal"
            embargo = (length - 1) if strict else int(config.get("split.embargo_days", 0))
            split = temporal_splits.final_week_split(
                data.daily,
                final_week_mode=str(config.get("split.final_week_mode", "calendar_days")),
                final_week_length=int(config.get("split.final_week_length", 7)),
                embargo_days=embargo,
                validation_days=int(config.get("split.validation_days", 0)),
                name=config.experiment,
            )
            temporal_splits.assert_no_shared_days(split)
            split_audit = leakage.audit_temporal_split(split, sequence_length=length)
            entry["split"] = split.describe()
            entry["thin_subjects"] = temporal_splits.summarise_thin_subjects(split, length)
            entry["split_audit"] = split_audit.summary()
            audit_ok &= split_audit.passed

            if config.experiment == "paper_literal_variant":
                cuts = temporal_splits.first_test_dates(
                    data.daily,
                    final_week_mode=str(config.get("split.final_week_mode", "calendar_days")),
                    final_week_length=int(config.get("split.final_week_length", 7)),
                )
                train, test, literal = build_sequences_literal(
                    data.daily, data.feature_columns, sequence_length=length,
                    stride=stride, test_start_by_subject=cuts,
                    require_consecutive=bool(config.get("sequence.require_consecutive", False)),
                    leakage_diagnostic_only=True,
                )
                entry["literal_split_report"] = literal
            else:
                train = build_sequences(split.train_days, data.feature_columns,
                                        sequence_length=length, stride=stride,
                                        split_name="train")
                test = build_sequences(split.test_days, data.feature_columns,
                                       sequence_length=length, stride=stride,
                                       split_name="test")
            expect_overlap = True
            estimand = "A"
        else:
            outer = group_splits.stratified_group_splits(
                data.subjects,
                n_splits=int(config.get("split.outer_k", 5)),
                n_repeats=int(config.get("split.n_repeats", 1)),
                seed=config.seed,
                name=config.experiment,
                stratify_on=str(config.get("split.stratify_on", "label")),
            )
            labels = data.labels_by_subject()
            entry["n_outer_splits"] = len(outer)
            entry["split_viability"] = group_splits.check_split_viability(outer, labels)
            entry["splits"] = [s.describe(labels) for s in outer[:5]]
            probe = outer[0]
            train = build_sequences(
                group_splits.iter_days(data.daily, probe.train_subjects),
                data.feature_columns, sequence_length=length, stride=stride,
                split_name="outer_train", outer_fold=probe.fold,
            )
            test = build_sequences(
                group_splits.iter_days(data.daily, probe.test_subjects),
                data.feature_columns, sequence_length=length, stride=stride,
                split_name="outer_test", outer_fold=probe.fold,
            )
            if config.experiment == "nested_subject_independent":
                inner = group_splits.inner_splits(
                    data.subjects, probe,
                    n_splits=int(config.get("split.inner_k", 3)), seed=config.seed,
                )
                isolation = leakage.audit_outer_test_isolation(
                    probe.test_subjects, inner_splits=inner,
                    selection_scores_source="inner_cv",
                )
                entry["inner_isolation_audit"] = isolation.summary()
                audit_ok &= isolation.passed
            expect_overlap = False
            estimand = "B"

        entry["train_sequences"] = train.describe()
        entry["test_sequences"] = test.describe()

        sampled, sampling = undersample(
            train,
            strategy=str(config.get("sampling.strategy", "random_sequence")),
            target_ratio=float(config.get("sampling.target_ratio", 1.0)),
            seed=config.seed,
        )
        sampling["split_applied_to"] = "train"
        entry["sampling"] = sampling

        scaler = SequenceScaler(
            method="standard" if registry.needs_scaling(probe_model) else "none"
        )
        scaled_train, scaled_test = scaler.fit_transform_pair(sampled, test)
        entry["scaler"] = scaler.describe()

        audit = leakage.audit_sequence_split(
            scaled_train, scaled_test,
            context=f"dry_run/{config.experiment}/L{length}",
            estimand=estimand,
            scaler=scaler, scaler_fit_source=scaled_train,
            sampling_report=sampling,
            sequence_length_source=(
                "inner_cv" if config.experiment == "nested_subject_independent"
                else "config_fixed"
            ),
            hyperparameter_source=(
                "inner_cv" if config.tuning_enabled else "paper_reported"
            ),
            early_stopping_source="none",
            expect_subject_overlap=expect_overlap,
            allow_boundary_crossing=config.experiment == "paper_literal_variant",
        )
        entry["leakage_audit"] = audit.summary()
        audit_ok &= audit.passed

        entry["model_input_shapes"] = {}
        for model_name in config.models:
            X = registry.model_input(
                scaled_train, model_name,
                representation=str(config.get("models.representation", "flatten")),
            )
            limit = config.get("tuning.max_candidates")
            entry["model_input_shapes"][model_name] = {
                "train_X_shape": list(np.asarray(X).shape),
                "finite": bool(np.isfinite(np.asarray(X)).all()),
                "runnable_here": model_name in registry.dependency_report()["runnable_models"],
                "n_candidates": len(
                    registry.search_space(
                        model_name,
                        enabled=config.tuning_enabled,
                        limit=int(limit) if limit else None,
                        seed=config.seed,
                    )
                ),
            }
        per_length.append(entry)

    report["per_sequence_length"] = per_length
    report["all_audits_passed"] = bool(dataset_audit.passed and audit_ok)
    report["fit_budget"] = _fit_budget(config, lengths, per_length)
    report["planned_steps"] = _planned_steps(config, lengths, report["fit_budget"])
    write_json(output_dir / "dry_run_report.json", report)

    _print_dry_run(report, config, output_dir)
    return report


def _fit_budget(config, lengths: list[int], per_length: list[dict[str, Any]]) -> dict[str, Any]:
    n_models = len(config.models)
    nested = config.experiment == "nested_subject_independent"
    if config.split_mode in ("final_week_temporal", "final_week_temporal_literal"):
        outer = 1
        total = n_models * len(lengths)
        inner_fits = 0
        candidates = {}
        per_model = {m: len(lengths) for m in config.models}
    else:
        outer = int(config.get("split.outer_k", 5)) * int(config.get("split.n_repeats", 1))
        inner_k = int(config.get("split.inner_k", 0)) if nested else 0
        # Candidate counts differ per model, so the budget is summed, not maxed.
        candidates = {
            m: max(entry["model_input_shapes"][m]["n_candidates"] for entry in per_length)
            for m in config.models
        }
        if nested:
            # Inner CV pays candidates x lengths x inner folds, per model per outer fold.
            per_model = {
                m: outer * (inner_k * candidates[m] * len(lengths) + 1)
                for m in config.models
            }
            inner_fits = sum(
                outer * inner_k * candidates[m] * len(lengths) for m in config.models
            )
            total = sum(per_model.values())
        else:
            per_model = {m: outer * len(lengths) for m in config.models}
            inner_fits = 0
            total = sum(per_model.values())
    return {
        "n_outer_splits": outer,
        "n_models": n_models,
        "n_sequence_lengths": len(lengths),
        "candidates_per_model": candidates,
        "fits_per_model": per_model,
        "inner_fits": inner_fits,
        "total_model_fits": total,
    }


def _planned_steps(config, lengths: list[int], budget: dict[str, Any]) -> list[str]:
    nested = config.experiment == "nested_subject_independent"
    return [
        f"1. 일별 표 로드 (피험자 174명 / 32개 변수)",
        (
            "2. 원시 날짜 분할 (피험자별 마지막 1주일)"
            if config.split_mode.startswith("final_week")
            else f"2. 피험자 분할 (StratifiedGroupKFold, outer {budget['n_outer_splits']}개)"
        ),
        f"3. 분할 이후 각 split 내부에서만 시퀀스 생성 (길이 {lengths})",
        "4. train에만 undersampling, train에만 scaler fit",
        (
            f"5. inner CV에서 길이·하이퍼파라미터·threshold 선택 "
            f"(inner fit {budget['inner_fits']}회)"
            if nested else "5. 논문 보고 설정 고정 (선택 없음)"
        ),
        f"6. 학습: 총 {budget['total_model_fits']}회 fit",
        "7. 시퀀스 단위 + 피험자 단위 지표 산출, bootstrap CI",
        "8. 비교표 생성 (--compare)",
    ]


def _print_dry_run(report: dict[str, Any], config, output_dir: Path) -> None:
    print(f"\n=== DRY RUN: {config.experiment} (estimand {report['estimand']}) ===")
    print(f"  config      : {config.path}")
    data = report["data"]
    print(f"  피험자/행/변수: {data['n_subjects']} / {data['n_daily_rows']} / {data['n_features']}")
    print(f"  진단 분포    : {data['diagnosis_counts']}")
    print(f"  모델         : {list(config.models)}")

    for entry in report["per_sequence_length"]:
        length = entry["sequence_length"]
        train, test = entry["train_sequences"], entry["test_sequences"]
        print(f"\n  --- 시퀀스 길이 {length}일 ---")
        print(f"    train 시퀀스 : {train['n_sequences']} "
              f"(양성 {train['n_positive_sequences']} / 음성 {train['n_negative_sequences']}, "
              f"피험자 {train['n_subjects']}명)")
        print(f"    test  시퀀스 : {test['n_sequences']} "
              f"(양성 {test['n_positive_sequences']} / 음성 {test['n_negative_sequences']}, "
              f"피험자 {test['n_subjects']}명)")
        print(f"    입력 shape   : {entry['model_input_shapes'][config.models[0]]['train_X_shape']}")
        sampling = entry["sampling"]
        print(f"    undersampling: {sampling['strategy']} → "
              f"{sampling['before']['n_sequences']}개에서 {sampling['after']['n_sequences']}개 "
              f"({sampling['n_removed']}개 제거)")
        for warning in sampling["warnings"]:
            print(f"      [경고] {warning}")
        if "thin_subjects" in entry:
            thin = entry["thin_subjects"]
            print(f"    test에서 연속 {length}일을 만들 수 있는 피험자: "
                  f"{thin['n_subjects_with_evaluable_test_sequence']}/"
                  f"{thin['n_subjects_total']}명 "
                  f"(평가 불가 {thin['n_subjects_lost_to_gaps_or_length']}명)")
        if "literal_split_report" in entry:
            literal = entry["literal_split_report"]
            print(f"    [누수 진단] 경계 교차 윈도우 "
                  f"{literal['n_boundary_crossing_sequences']}개 "
                  f"({literal['boundary_crossing_fraction']:.1%}), "
                  f"train과 공유되는 test 날짜 "
                  f"{literal['n_subject_dates_in_both_splits']}/"
                  f"{literal['n_subject_dates_in_test']} "
                  f"({literal['subject_date_leak_fraction']:.1%})")
        audit = entry["leakage_audit"]
        print(f"    누수 검사    : {audit['n_checks']}개 중 실패 {audit['n_failures']}개")
        for failure in audit["checks"]:
            if not failure["passed"]:
                print(f"      [{failure['severity'].upper()}] {failure['check']}: {failure['detail']}")

    budget = report["fit_budget"]
    print(f"\n  예상 학습 횟수: {budget['total_model_fits']}회 "
          f"(inner {budget['inner_fits']}회 포함)")
    if budget["fits_per_model"]:
        print(f"    모델별: {budget['fits_per_model']}")
    if budget["candidates_per_model"]:
        print(f"    후보수: {budget['candidates_per_model']}")
    print(f"  전체 누수 검사: {'전부 통과' if report['all_audits_passed'] else '실패 있음'}")
    print(f"  논문 산술 검증: {report['paper_arithmetic_check']['all_models_consistent']}")
    print("\n  예정 실행 단계:")
    for step in report["planned_steps"]:
        print(f"    {step}")
    print(f"\n  보고서: {output_dir / 'dry_run_report.json'}")
    print("\n  학습은 수행하지 않았다.")


def mode_compare(output_dir: Path, report_paths: dict[str, str]) -> dict[str, Any]:
    from src.evaluation.compare import (
        build_comparison, load_reports, render_comparison_markdown,
    )
    from src.utils.io import write_json, write_text

    reports = load_reports(report_paths)
    comparison = build_comparison(reports)
    write_json(output_dir / "comparison.json", comparison)
    write_text(output_dir / "comparison.md", render_comparison_markdown(comparison))
    print(render_comparison_markdown(comparison))
    if not reports:
        print("주의: FINAL_REPORT.json을 찾지 못해 논문 열만 채워졌다.")
    return comparison


# --- main ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hong et al. (2024) reproduction runner")
    parser.add_argument("--config", type=str, help="path to a YAML config")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="validate splits, shapes and audits without training")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--inspect-data", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=None, choices=(3, 4, 5))
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", action="append", default=None,
                        help="restrict to one model; repeat the flag for several")
    parser.add_argument("--estimand", type=str, default=None, choices=("A", "B"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--skip-install", action="store_true")
    return parser


def run_pipeline(*, namespace: dict[str, Any] | None = None,
                 argv: list[str] | None = None) -> dict[str, Any]:
    namespace = globals() if namespace is None else namespace
    args = build_parser().parse_args(argv)

    no_training = args.dry_run or args.audit_only or args.inspect_data or args.compare
    _ensure_dependencies(args.skip_install, need_torch=False, need_xgboost=False)

    from src.utils.io import resolve_output_dir, write_json, write_status

    output_dir = Path(resolve_output_dir(args.output_dir, allow_local=no_training))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        return mode_compare(
            output_dir,
            {
                "paper_temporal_reconstruction":
                    os.environ.get("HONG2024_REPORT_A", "outputs/A/FINAL_REPORT.json"),
                "paper_literal_variant":
                    os.environ.get("HONG2024_REPORT_A_LITERAL", "outputs/A_literal/FINAL_REPORT.json"),
                "strict_same_subject_temporal":
                    os.environ.get("HONG2024_REPORT_B1", "outputs/B1/FINAL_REPORT.json"),
                "fixed_subject_independent":
                    os.environ.get("HONG2024_REPORT_B2", "outputs/B2/FINAL_REPORT.json"),
                "nested_subject_independent":
                    os.environ.get("HONG2024_REPORT_C", "outputs/C/FINAL_REPORT.json"),
            },
        )

    data_root = _resolve_data_root(namespace, args.data_root)

    if args.inspect_data:
        return mode_inspect_data(data_root, output_dir)

    if not args.config:
        raise SystemExit("--config is required (or use --inspect-data / --compare)")

    from src.utils.config import load_config

    config = load_config(args.config)
    if args.seed is not None:
        config.raw["seed"] = int(args.seed)
    if args.model:
        unknown = set(args.model) - set(config.models)
        if unknown:
            raise SystemExit(f"--model {sorted(unknown)} not in config.models {list(config.models)}")
        config.raw.setdefault("models", {})["enabled"] = list(args.model)
    if args.estimand and args.estimand != config.estimand:
        raise SystemExit(
            f"--estimand {args.estimand} was requested but {config.experiment} "
            f"estimates estimand {config.estimand}. 두 추정량을 같은 표에 섞지 않기 위해 중단한다."
        )

    needs_torch = "lstm" in config.models
    needs_xgboost = "xgboost" in config.models
    deps = _ensure_dependencies(
        args.skip_install or no_training,
        need_torch=needs_torch and not no_training,
        need_xgboost=needs_xgboost and not no_training,
    )

    from src.data.loader import load_lifelog
    from src.utils.seeding import resolve_device, seed_everything

    seed_everything(config.seed)
    device = resolve_device(args.device) if not no_training else "cpu"

    data = load_lifelog(
        data_root,
        sleep_date_source=str(config.get("data.sleep_date_source", "bedtime_end")),
        duplicate_policy=str(config.get("data.duplicate_policy", "longest_duration")),
        rmssd_source=str(config.get("data.rmssd_source", "intraday_mean")),
    )

    if args.audit_only:
        return mode_audit_only(data, output_dir)
    if args.dry_run:
        return mode_dry_run(data, config, output_dir, device, deps,
                            only_length=args.sequence_length)

    # --- full run -------------------------------------------------------------
    from src.engine import run_experiment
    from src.evaluation.compare import build_comparison, render_comparison_markdown
    from src.utils.io import write_text

    started = time.monotonic()
    write_status(output_dir, {
        "status": "starting", "experiment": config.experiment,
        "estimand": config.estimand, "config": str(config.path),
        "data_root": str(data_root), "output_dir": str(output_dir),
        "device": device, "seed": config.seed,
    })

    try:
        report = run_experiment(
            data, config, output_dir=output_dir, device=device,
            only_fold=args.fold, only_length=args.sequence_length, resume=args.resume,
        )
    except Exception as error:
        write_status(output_dir, {
            "status": "failed", "experiment": config.experiment,
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.monotonic() - started,
        })
        raise

    report["elapsed_seconds"] = time.monotonic() - started
    write_json(output_dir / "FINAL_REPORT.json", report)
    write_json(output_dir / "TRAINING_COMPLETE.json", {
        "experiment": config.experiment,
        "estimand": report["estimand"],
        "result_keys": list(report["results"]),
        "all_audits_passed": report["all_audits_passed"],
        "elapsed_seconds": report["elapsed_seconds"],
    })

    comparison = build_comparison({config.experiment: report})
    write_text(output_dir / "comparison_partial.md", render_comparison_markdown(comparison))

    unit = "subject_level" if config.estimand == "B" else "sequence_level"
    headline = {
        key: block.get(unit, {}).get("roc_auc")
        for key, block in report["results"].items()
    }
    write_status(output_dir, {
        "status": "complete", "experiment": config.experiment,
        "estimand": report["estimand"],
        "elapsed_seconds": report["elapsed_seconds"],
        "all_audits_passed": report["all_audits_passed"],
        "headline_roc_auc": headline, "headline_unit": unit,
        "final_report": str(output_dir / "FINAL_REPORT.json"),
    })

    print(f"\n완료 ({config.experiment}, estimand {report['estimand']}) — "
          f"{report['elapsed_seconds'] / 60:.1f}분")
    print(f"  {unit} ROC-AUC: {headline}")
    print(f"  보고서: {output_dir / 'FINAL_REPORT.json'}")
    return report


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
