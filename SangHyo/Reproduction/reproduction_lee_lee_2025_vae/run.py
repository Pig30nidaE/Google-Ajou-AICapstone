#!/usr/bin/env python
"""이민지·이석훈(2025) VAE 재현 — 단일 실행 진입점.

    python run.py --config configs/paper_percentile_latent500.yaml
    python run.py --config configs/leakage_controlled_non_nested.yaml
    python run.py --config configs/nested_subject_independent.yaml

먼저 ``--dry-run``으로 절차와 규모를 확인한 뒤 실제 학습을 시작하는 것을 권장한다.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.inspect import format_inspection, inspect_data, percentile_retention_scan  # noqa: E402
from src.data.loader import load_lifelog  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io import save_json, save_table  # noqa: E402
from src.utils.seeding import set_global_seed  # noqa: E402

log = logging.getLogger("run")

EXPERIMENT_DISPATCH = {
    "paper_reported_reconstruction": "A",
    "leakage_controlled_non_nested": "B",
    "nested_subject_independent": "C",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="이민지·이석훈(2025) VAE 기반 치매 조기 탐지 재현",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=str, help="config yaml 경로")
    p.add_argument("--data-root", type=str, default=None, help="Data/ 경로 (config 기본값 override)")
    p.add_argument("--out-root", type=str, default=None, help="출력 루트")

    p.add_argument("--inspect-data", action="store_true", help="데이터 구조·계약만 점검하고 종료")
    p.add_argument("--audit-only", action="store_true", help="누수 검사와 이상치 재현 검증만 수행")
    p.add_argument("--dry-run", action="store_true", help="학습 없이 절차·규모·누수 검사만 수행")

    p.add_argument("--fold", type=int, default=None, help="특정 fold만 실행")
    p.add_argument("--seed", type=int, default=None, help="난수 seed (config override)")
    p.add_argument("--resume", action="store_true", help="이미 완료된 산출물을 건너뛴다")
    p.add_argument("--skip-vae", action="store_true", help="VAE 조건을 제외하고 실행")
    p.add_argument(
        "--augmentation",
        choices=["none", "vae", "class_weight", "random_oversampling", "smote"],
        action="append",
        help="실행할 증강 조건 (여러 번 지정 가능)",
    )
    p.add_argument("--models", type=str, default=None, help="쉼표로 구분한 모델 목록")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---------------- config
    if args.config:
        cfg = load_config(args.config)
    elif args.inspect_data:
        cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    else:
        log.error("--config가 필요하다 (또는 --inspect-data)")
        return 2

    data_root = args.data_root or cfg.get_path("data.root", "../../../Data")
    if not Path(data_root).is_absolute():
        data_root = (REPO_ROOT / data_root).resolve()
    out_root = args.out_root or cfg.get_path("output.root", "outputs")
    if not Path(out_root).is_absolute():
        out_root = (REPO_ROOT / out_root).resolve()
    seed = args.seed if args.seed is not None else int(cfg.get_path("seed", 42))
    set_global_seed(seed)

    label = cfg.get_path("label", Path(args.config).stem if args.config else "base")
    log.info("config=%s label=%s seed=%d", cfg.get_path("_meta.config_path"), label, seed)

    # ---------------- data
    data = load_lifelog(
        data_root,
        drop_duplicate_columns=bool(cfg.get_path("features.drop_duplicate_columns", False)),
        extra_features=tuple(cfg.get_path("features.extra_sleep_features", []) or []),
    )

    # ---------------- --inspect-data
    if args.inspect_data:
        report = inspect_data(data)
        print(format_inspection(report))
        save_json(report, Path(out_root) / "inspection" / "data_inspection.json")
        scan = percentile_retention_scan(data)
        print("\n-- 분위수 절단이 논문 §5.1 행 수를 재현하는가 (I-1 증거 B·C) --")
        print(scan.to_string(index=False))
        print(f"\n논문 목표: {scan.attrs['paper_target']}")
        print(scan.attrs["note"])
        save_table(scan, Path(out_root) / "inspection" / "percentile_retention_scan.csv")
        return 0

    experiment = cfg.get_path("experiment.name")
    kind = EXPERIMENT_DISPATCH.get(experiment)
    if kind is None:
        log.error("알 수 없는 experiment.name: %r (가능: %s)", experiment, list(EXPERIMENT_DISPATCH))
        return 2

    # ---------------- 실행 조건
    augs = tuple(args.augmentation) if args.augmentation else tuple(
        cfg.get_path("run.augmentations", ["none", "vae"])
    )
    if args.skip_vae:
        augs = tuple(a for a in augs if a != "vae")
        log.info("--skip-vae: 증강 조건 = %s", augs)
    models = tuple(
        args.models.split(",") if args.models else cfg.get_path("run.models", ["xgboost", "dnn", "tabnet", "wide_deep"])
    )

    # ---------------- --dry-run / --audit-only
    if args.dry_run or args.audit_only:
        return _dry_run(data, cfg, kind, out_root, label, seed, args)

    # ---------------- 실제 실행
    if kind == "A":
        from src.experiments.paper_reconstruction import run_experiment_a

        res = run_experiment_a(
            data, cfg, out_root=out_root, label=label, seed=seed,
            models=models, augmentations=augs,
        )
    elif kind == "B":
        from src.experiments.leakage_controlled import run_experiment_b

        res = run_experiment_b(
            data, cfg, out_root=out_root, label=label, seed=seed,
            models=models, augmentations=augs, only_fold=args.fold,
        )
    else:
        from src.experiments.nested_cv import run_experiment_c

        res = run_experiment_c(
            data, cfg, out_root=out_root, label=label, seed=seed, only_fold=args.fold
        )

    audit = res["audit"]
    log.info(
        "완료: %s. 누수 위반 %d건 (mode=%s). 출력 -> %s",
        experiment, audit["n_violations"], audit["mode"], res["paths"].root,
    )
    if audit["n_violations"] and audit["mode"] == "observe":
        log.warning(
            "실험 A는 논문 절차의 누수를 의도적으로 재현한다. "
            "관측된 위반은 leakage_observation.json에 정리되어 있다."
        )
    return 0


def _dry_run(data, cfg, kind, out_root, label, seed, args) -> int:
    """학습 없이 검증만 수행한다 (사용자 지시 14절)."""
    from src.data.schema import PAPER_FEATURES
    from src.audit.checks import check_forbidden_features

    print("=" * 78)
    print(f"DRY-RUN — experiment {kind} / config {label} / seed {seed}")
    print("=" * 78)

    missing = [c for c in PAPER_FEATURES if c not in data.features]
    print(f"[변수] 논문 46개 중 존재: {46 - len(missing)}/46, 누락: {missing or '없음'}")
    v = check_forbidden_features(data.features)
    print(f"[금지변수] {'위반 없음 ✅' if not v else v}")
    print(f"[피험자] 총 {len(data.subjects())}명, 클래스별 {data.class_counts(by='subject')}")
    print(f"[기록]   총 {data.n:,}행, 클래스별 {data.class_counts(by='record')}")

    if kind == "A":
        from src.experiments.paper_reconstruction import plan_experiment_a

        plan = plan_experiment_a(data, cfg, seed=seed)
    elif kind == "B":
        from src.experiments.leakage_controlled import plan_experiment_b

        plan = plan_experiment_b(data, cfg, seed=seed)
    else:
        from src.experiments.nested_cv import plan_experiment_c

        plan = plan_experiment_c(data, cfg, seed=seed)

    _print_plan(kind, plan)
    save_json(plan, Path(out_root) / f"{kind}_{label}" / "dry_run_plan.json")

    if args.audit_only and kind == "A":
        print("\n-- 이상치 재현 검증 (--audit-only) --")
        _audit_outlier(data, cfg, out_root, label)
    print("\n학습은 실행하지 않았다. 실제 실행은 --dry-run 없이 같은 명령을 쓰면 된다.")
    return 0


def _print_plan(kind: str, plan: dict) -> None:
    if kind == "A":
        print("\n-- 실험 A 계획 --")
        print(f"  {'이상치 방식':38s}: {plan['outlier_method']}")
        print(f"  {'이상치 제거 전 행 수':38s}: {plan['rows_before_outlier']}")
        print(f"  {'이상치 제거 후 행 수':38s}: {plan['rows_after_outlier']}")
        print(f"  {'논문 §5.1 보고값':38s}: {plan['paper_after_outlier']}")
        print(f"  {'→ 논문 행 수 재현':38s}: "
              f"{'예 ✅' if plan['outlier_matches_paper'] else '아니오 ❌ (I-1 참조)'}")
        for k in (
            "split_unit", "n_train", "n_valid", "n_test",
            "n_train_subjects", "n_test_subjects",
            "subject_overlap_train_test", "subject_overlap_train_valid",
            "preprocessing_fit_scope", "scaler_scope", "vae_fit_scope",
            "dem_train_rows_real", "expected_synthetic_rows",
            "dem_train_rows_after_augmentation", "paper_table5_dem_train",
        ):
            print(f"  {k:38s}: {plan.get(k)}")
        print(f"  {'note':38s}: {plan['note']}")
        print("\n  표 5 대조:")
        import pandas as pd

        print(pd.DataFrame(plan["table5_comparison"]).to_string(index=False))
    elif kind == "B":
        import pandas as pd

        print("\n-- 실험 B 계획 --")
        print(f"  split: {plan['split']}, audit_mode={plan['audit_mode']}")
        print(f"  scaler_scope={plan['scaler_scope']}, vae_fit_scope={plan['vae_fit_scope']}")
        print("\n  fold 구성:")
        print(pd.DataFrame(plan["fold_composition"]).to_string(index=False))
        print("\n  fold별 전처리·VAE 범위와 예상 합성행:")
        print(pd.DataFrame(plan["per_fold_plan"]).to_string(index=False))
        print(f"\n  {plan['note']}")
    else:
        import pandas as pd

        print("\n-- 실험 C 계획 --")
        print(f"  outer folds     : {plan['n_outer_folds']}")
        print(f"  후보 설정 수    : {plan['n_candidates']} (max_evals={plan['max_evals']})")
        print(f"  총 모델 적합 수 : {plan['total_model_fits']}")
        print("\n  outer fold 구성:")
        print(pd.DataFrame(plan["outer_composition"]).to_string(index=False))
        print("\n  후보 미리보기:")
        for c in plan["candidates_preview"]:
            print(f"    {c}")
        print(f"\n  {plan['note']}")


def _audit_outlier(data, cfg, out_root, label) -> None:
    """논문 §5.1의 행 수를 어느 이상치 방식이 재현하는지 검증한다.

    IsolationForest 적합은 분류기·VAE 학습이 아니라 데이터 특성 확인이며,
    ``--audit-only``에서만 수행된다.
    """
    import pandas as pd

    from src.data.paper_reference import SECTION51_AFTER_OUTLIER
    from src.preprocessing.outliers import make_outlier_handler

    target = SECTION51_AFTER_OUTLIER
    rows = []
    for name, ocfg in (
        ("percentile q=0.10 (논문 §5.1 본문)", {"method": "percentile", "percentile": {"q": 0.10}}),
        ("percentile q=0.003 (최근접)", {"method": "percentile", "percentile": {"q": 0.003}}),
        ("isolation_forest c=0.1 (논문 §4.2·그림 1)",
         {"method": "isolation_forest", "isolation_forest": {"contamination": 0.1}}),
    ):
        h = make_outlier_handler(ocfg, seed=int(cfg.get_path("seed", 42)))
        h.fit(data.X, data.y)
        res = h.transform(data.X, data.y)
        got = {
            cls: int(((data.y == code) & res.keep_mask).sum())
            for code, cls in enumerate(("CN", "MCI", "Dem"))
        }
        rows.append(
            {
                "method": name,
                "kept_total": int(res.keep_mask.sum()),
                "retention": round(float(res.keep_mask.mean()), 4),
                **got,
                "paper_CN": target["CN"], "paper_MCI": target["MCI"], "paper_Dem": target["Dem"],
                "L1_distance": sum(abs(got[k] - target[k]) for k in target),
            }
        )
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\n논문 §5.1 보고: CN 7,075 / MCI 3,374 / Dem 515 (합 10,964 = 전체의 89.994%)")
    save_table(df, Path(out_root) / f"A_{label}" / "outlier_method_audit.csv")


if __name__ == "__main__":
    raise SystemExit(main())
