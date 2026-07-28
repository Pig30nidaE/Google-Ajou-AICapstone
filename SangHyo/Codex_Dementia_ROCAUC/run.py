"""Strict command-line entry point.

Training requires a literal acknowledgement token so code validation and data
audit can never start an expensive model fit accidentally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    package_root = Path(__file__).resolve().parent
    repository_root = package_root.parents[1]
    for import_root in (repository_root, package_root.parent):
        import_path = str(import_root)
        if import_path not in sys.path:
            sys.path.insert(0, import_path)
    __package__ = "Codex_Dementia_ROCAUC"

from .artifacts import write_json
from .config import make_config


TRAINING_ACKNOWLEDGEMENT = "I_UNDERSTAND_THIS_RUNS_TRAINING"


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default="Data")
    parser.add_argument("--output-dir", default="Codex_Dementia_ROCAUC_results")
    parser.add_argument("--profile", choices=("standard", "max"), default="standard")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CN+MCI versus Dem subject-level ROC-AUC pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser(
        "validate-code",
        help="syntax/synthetic split/leakage checks; executes zero model fits",
    )
    validate.add_argument(
        "--output",
        default=None,
        help="optional JSON path for the zero-fit validation report",
    )
    audit = subparsers.add_parser(
        "audit-data",
        help="read-only schema/ID audit; executes zero model fits",
    )
    _common(audit)
    make_splits = subparsers.add_parser(
        "make-splits",
        help="save the primary subject split registry; executes zero model fits",
    )
    _common(make_splits)
    train = subparsers.add_parser("train", help="execute the full training protocol")
    _common(train)
    train.add_argument(
        "--execute-training",
        required=True,
        metavar="ACK",
        help=f"must equal {TRAINING_ACKNOWLEDGEMENT}",
    )
    return parser


def _config(args) -> object:
    return make_config(
        profile=args.profile,
        data_root=args.data_root,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
        device=args.device,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate-code":
        from .validation import validate_code_without_fit

        report = validate_code_without_fit(Path(__file__).resolve().parent)
        if args.output:
            write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "audit-data":
        from .validation import audit_data_without_training

        config = _config(args)
        report = audit_data_without_training(config)
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "DATA_AUDIT_ZERO_FIT.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "make-splits":
        from .data import load_development_cohort
        from .splits import (
            build_repeated_group_plan,
            save_split_plan,
        )

        config = _config(args)
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        reports = {}
        for track in config.data.tracks:
            cohort = load_development_cohort(config.data, track)
            plan = build_repeated_group_plan(
                cohort.y,
                cohort.groups,
                n_splits=config.cv.outer_folds,
                n_repeats=config.cv.outer_repeats,
                seed=config.cv.seed,
                minimum_positive_validation=(
                    config.cv.minimum_positive_per_validation_fold
                ),
                layer=f"{track}_outer",
            )
            path = output / track / "split_registry_outer.json"
            save_split_plan(
                path,
                plan,
                subject_ids=cohort.subject_ids,
                y=cohort.y,
            )
            reports[track] = str(path.resolve())
        write_json(
            output / "SPLIT_PLAN_ZERO_FIT.json",
            {"training_calls_executed": 0, "tracks": reports},
        )
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return 0
    if args.command == "train":
        if args.execute_training != TRAINING_ACKNOWLEDGEMENT:
            parser.error(
                "--execute-training must exactly equal "
                f"{TRAINING_ACKNOWLEDGEMENT}"
            )
        config = _config(args)
        output = config.runtime.resolved_output()
        if output.exists() and any(output.iterdir()):
            parser.error(f"refusing to train into non-empty output directory: {output}")
        from .train import run_experiment

        run_experiment(config)
        return 0
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
