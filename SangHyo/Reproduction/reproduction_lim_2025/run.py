"""Single entry point for the Lim (2025) reproduction.

    USER_FOLDER = "SangHyo"
    RUN_FILE    = "Reproduction/reproduction_lim_2025/run.py"

Usage::

    python run.py --config configs/paper_reproduction.yaml
    python run.py --config configs/leakage_controlled_non_nested.yaml
    python run.py --config configs/nested_subject_independent.yaml

Options::

    --dry-run        resolve paths, build shapes, run every audit; train nothing
    --audit-only     dataset-level leakage audit only
    --inspect-data   paper-vs-data discrepancy report only
    --fold N         restrict to one outer fold
    --seed N         override config.seed
    --resume         reuse completed (model, repeat, fold) checkpoints
    --compare        rebuild the cross-experiment comparison table from reports

Recommended Colab runtime: **CPU / High-RAM** for the tree models,
**GPU** only when the LSTM / Bi-LSTM / 1D-CNN arms are enabled.
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

EXPERIMENT_NAME = "reproduction_lim_2025"
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
                "scipy": "scipy", "joblib": "joblib", "pyyaml": "yaml"}
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
    print(f"보고서: {output_dir / 'inspection' / 'discrepancy_report.md'}")
    return report


def mode_audit_only(data, output_dir: Path) -> dict[str, Any]:
    from src.audit.leakage import audit_dataset
    from src.utils.io import write_json

    log = audit_dataset(data)
    summary = log.summary()
    write_json(output_dir / "audit" / "dataset_audit.json", summary)
    print(f"\n=== 데이터셋 누수 감사 ===")
    for check in summary["checks"]:
        print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {check['check']}")
    print(f"\n전체 통과: {summary['all_passed']}")
    return summary


def mode_dry_run(data, config, output_dir: Path, device: str, deps: dict[str, bool]) -> dict[str, Any]:
    """Everything except fitting: paths, shapes, splits, audits, planned steps."""
    import numpy as np

    from src.audit import leakage
    from src.evaluation.compare import verify_paper_arithmetic
    from src.features.representations import FoldPreprocessor, fit_transform_pair
    from src.engine import _needs_scaling, materialise_pair
    from src.models import registry
    from src.splits import splitters
    from src.utils.io import write_json

    labels = data.labels_by_subject()
    report: dict[str, Any] = {
        "experiment": config.experiment,
        "config_path": str(config.path),
        "seed": config.seed,
        "device": device,
        "dependencies": {**deps, **registry.dependency_report()},
        "data": {
            "n_subjects": data.n_subjects,
            "n_daily_rows": data.n_rows,
            "n_features": len(data.feature_columns),
            "feature_columns": list(data.feature_columns),
            "class_counts": data.subjects[  # subject-level
                "label"
            ].value_counts().sort_index().to_dict(),
            "notes": data.notes,
        },
        "paper_arithmetic_check": verify_paper_arithmetic(),
    }

    dataset_audit = leakage.audit_dataset(data)
    report["dataset_audit"] = dataset_audit.summary()

    splits = splitters.build_outer_splits(data, config)
    report["splits"] = [s.describe(labels) for s in splits]
    report["n_splits"] = len(splits)

    # Every split must be sane before a single epoch is spent.
    min_positive = min(
        int(labels.reindex(list(s.test_subjects)).sum()) for s in splits
    )
    report["min_test_positives"] = min_positive
    if min_positive < 1:
        report.setdefault("warnings", []).append(
            "적어도 한 fold의 test에 양성이 없다. ROC-AUC가 정의되지 않는다."
        )

    # --- shapes, without fitting any model -----------------------------------
    shapes: dict[str, Any] = {}
    probe = splits[0]
    for model_name in config.models:
        train_rep, test_rep = materialise_pair(
            data, config, model_name, probe.train_subjects, probe.test_subjects
        )
        pre = FoldPreprocessor(standardize=_needs_scaling(model_name, config))
        fit_transform_pair(pre, train_rep, test_rep)

        audit_log = leakage.audit_split(
            probe,
            train_rep=train_rep, test_rep=test_rep, preprocessor=pre,
            experiment=config.experiment,
            threshold_source=str(config.get("threshold.policy", "fixed")),
            include_cognitive=bool(config.get("features.include_cognitive_tests", False)),
        )
        shapes[model_name] = {
            "representation": config.get(f"representations.{model_name}"),
            "train": train_rep.describe(),
            "test": test_rep.describe(),
            "train_X_shape": list(train_rep.X.shape),
            "test_X_shape": list(test_rep.X.shape),
            "finite_values": bool(np.isfinite(train_rep.X).all()),
            "audit_passed": audit_log.passed,
            "audit_failures": audit_log.failures,
            "runnable_here": model_name in registry.dependency_report()["runnable_models"],
        }
    report["model_inputs"] = shapes

    n_models = len(config.models)
    tuning_on = bool(config.get("tuning.enabled", False))
    candidates = {m: len(registry.search_space(m, enabled=tuning_on)) for m in config.models}
    is_nested = config.experiment == "nested_subject_independent"
    inner_k = int(config.get("split.inner_k", 0)) if is_nested else 0
    # Nested pays for the inner CV too: (1 outer + inner_k x candidates) per model per split.
    total_fits = len(splits) * sum(1 + inner_k * candidates[m] for m in config.models)
    report["fit_budget"] = {
        "n_outer_splits": len(splits),
        "n_models": n_models,
        "candidates_per_model": candidates,
        "inner_k": inner_k,
        "total_model_fits": total_fits,
    }
    report["planned_steps"] = [
        f"1. 데이터 로드: {data.n_rows}행 / {data.n_subjects}명 / 특징 {len(data.feature_columns)}개",
        f"2. 분할 생성: {config.split_mode}, outer split {len(splits)}개",
        (f"3. inner CV: {inner_k}-fold, 모델별 후보 {candidates}"
         if is_nested else "3. inner CV 없음 (하이퍼파라미터 고정)"),
        f"4. 학습: 총 {total_fits}회 fit "
        f"({'outer ' + str(n_models * len(splits)) + '회 + inner ' + str(total_fits - n_models * len(splits)) + '회' if is_nested else str(n_models) + '개 모델 × ' + str(len(splits)) + '개 split'})",
        "5. 피험자 단위 예측 통합 및 지표 산출",
        "6. bootstrap CI / fold 변동 / 비교표 생성",
    ]
    report["all_audits_passed"] = bool(
        dataset_audit.passed and all(s["audit_passed"] for s in shapes.values())
    )

    write_json(output_dir / "dry_run_report.json", report)

    print(f"\n=== DRY RUN: {config.experiment} ===")
    print(f"  config       : {config.path}")
    print(f"  피험자/행/특징 : {data.n_subjects} / {data.n_rows} / {len(data.feature_columns)}")
    print(f"  클래스(피험자): {report['data']['class_counts']}")
    print(f"  분할         : {config.split_mode} → {len(splits)} split(s)")
    for split in report["splits"][:3]:
        print(f"    - {split['name']} r{split['repeat']}f{split['fold']}: "
              f"train {split['n_train_subjects']} (양성 {split.get('n_train_positive')}) / "
              f"test {split['n_test_subjects']} (양성 {split.get('n_test_positive')})")
    if len(report["splits"]) > 3:
        print(f"    ... 총 {len(report['splits'])}개")
    print("  모델 입력 shape:")
    for name, info in shapes.items():
        flag = "" if info["runnable_here"] else "  [이 환경에서는 라이브러리 없음]"
        print(f"    - {name:14s} {info['representation']:26s} "
              f"train={info['train_X_shape']} test={info['test_X_shape']}{flag}")
    print(f"  누수 검사     : {'전부 통과' if report['all_audits_passed'] else '실패 있음'}")
    print(f"  논문 산술 검증 : {report['paper_arithmetic_check']['all_models_consistent']}")
    print("\n  예정 실행 단계:")
    for step in report["planned_steps"]:
        print(f"    {step}")
    print(f"\n  보고서: {output_dir / 'dry_run_report.json'}")
    print("\n  학습은 수행하지 않았다.")
    return report


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
    parser = argparse.ArgumentParser(description="Lim (2025) reproduction runner")
    parser.add_argument("--config", type=str, help="path to a YAML config")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="validate paths, shapes, splits and audits without training")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--inspect-data", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--skip-install", action="store_true")
    return parser


def run_pipeline(*, namespace: dict[str, Any] | None = None, argv: list[str] | None = None) -> dict[str, Any]:
    namespace = globals() if namespace is None else namespace
    args = build_parser().parse_args(argv)

    no_training = args.dry_run or args.audit_only or args.inspect_data or args.compare

    _ensure_dependencies(args.skip_install, need_torch=False, need_xgboost=False)

    from src.utils.io import resolve_output_dir, write_json, write_status

    output_dir = Path(
        resolve_output_dir(args.output_dir, allow_local=no_training)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        return mode_compare(
            output_dir,
            {
                "paper_reported_reconstruction":
                    os.environ.get("LIM2025_REPORT_A", "outputs/A/FINAL_REPORT.json"),
                "leakage_controlled_non_nested":
                    os.environ.get("LIM2025_REPORT_B", "outputs/B/FINAL_REPORT.json"),
                "nested_subject_independent":
                    os.environ.get("LIM2025_REPORT_C", "outputs/C/FINAL_REPORT.json"),
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

    needs_torch = any(m in ("lstm", "bilstm", "cnn1d") for m in config.models)
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
        join_mode=str(config.get("data.join_mode", "positional")),
        sleep_date_source=str(config.get("data.sleep_date_source", "bedtime_end")),
        feature_set=str(config.get("features.feature_set", "paper_code_verbatim")),
        drop_zero_variance=bool(config.get("features.drop_zero_variance", False)),
        drop_administrative=bool(config.get("features.drop_administrative", False)),
    )

    if args.audit_only:
        return mode_audit_only(data, output_dir)
    if args.dry_run:
        return mode_dry_run(data, config, output_dir, device, deps)

    # --- full run -------------------------------------------------------------
    from src.engine import run_experiment
    from src.evaluation.compare import render_comparison_markdown, build_comparison
    from src.utils.io import write_text

    started = time.monotonic()
    write_status(output_dir, {
        "status": "starting", "experiment": config.experiment,
        "config": str(config.path), "data_root": str(data_root),
        "output_dir": str(output_dir), "device": device, "seed": config.seed,
    })

    try:
        report = run_experiment(
            data, config, output_dir=output_dir, device=device,
            only_fold=args.fold, resume=args.resume,
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
        "models": list(report["models"]),
        "all_audits_passed": report["all_audits_passed"],
        "elapsed_seconds": report["elapsed_seconds"],
    })

    comparison = build_comparison({config.experiment: report})
    write_text(output_dir / "comparison_partial.md", render_comparison_markdown(comparison))

    headline = {
        model: block["subject_level"]["roc_auc"]
        for model, block in report["models"].items()
        if block.get("subject_level")
    }
    write_status(output_dir, {
        "status": "complete", "experiment": config.experiment,
        "elapsed_seconds": report["elapsed_seconds"],
        "all_audits_passed": report["all_audits_passed"],
        "headline_subject_roc_auc": headline,
        "final_report": str(output_dir / "FINAL_REPORT.json"),
    })

    print(f"\n완료 ({config.experiment}) — {report['elapsed_seconds'] / 60:.1f}분")
    print(f"  피험자 단위 ROC-AUC: {headline}")
    print(f"  보고서: {output_dir / 'FINAL_REPORT.json'}")
    return report


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
