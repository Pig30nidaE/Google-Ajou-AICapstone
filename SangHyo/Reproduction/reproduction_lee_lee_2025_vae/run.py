#!/usr/bin/env python
"""이민지·이석훈(2025) VAE 재현 — 단일 실행 진입점.

로컬 / 순수 CLI::

    python run.py --config configs/paper_isoforest_latent500.yaml
    python run.py --config configs/leakage_controlled_non_nested.yaml
    python run.py --config configs/nested_subject_independent.yaml

먼저 ``--dry-run``으로 절차와 규모를 확인한 뒤 실제 학습을 시작하는 것을 권장한다.

Colab / ``base.ipynb`` — 한 번에 실행
--------------------------------------
저장소 루트 ``base.ipynb``의 **Cell 2만 수정하고 그대로 위에서 아래로 실행**하면 된다::

    USER_FOLDER = "SangHyo"
    RUN_FILE    = "Reproduction/reproduction_lee_lee_2025_vae/run.py"

``Reproduction/`` 한 단계를 빠뜨리면 Cell 3에서 ``FileNotFoundError``가 난다.

이 스크립트는 base.ipynb Cell 5의 실행 방식(``runpy.run_path(..., run_name="__main__")``)을
스스로 감지한다. 그 경로에서는 ``sys.argv``가 노트북 인자가 아니라 **Jupyter 커널 자신의
인자**(``-f kernel-xxxx.json``)이므로, argparse에 넘기면 ``unrecognized arguments``로 죽는다.
그래서 커널 argv를 감지하면 **무시하고 전체 파이프라인(:func:`run_all`)을 실행**한다.
Cell 1이 주입한 ``PROJECT_ROOT``/``DATA_ROOT``도 자동으로 사용한다.

노트북에서 특정 단계만 돌리고 싶으면 Cell 5 **앞에** 환경변수를 지정한다::

    import os
    os.environ["VAE2025_ARGS"] = "--config configs/leakage_controlled_non_nested.yaml --dry-run"

또는 셀에서 직접 함수를 부른다 (``argv``를 명시하면 커널 argv를 완전히 무시한다)::

    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "SangHyo/Reproduction/reproduction_lee_lee_2025_vae"))
    from run import run_pipeline
    run_pipeline(namespace=globals(), argv=["--config", "configs/base.yaml", "--dry-run"])

산출물 저장 위치
----------------
``/content``는 런타임이 끊기면 사라지므로, Drive가 마운트되어 있으면 산출물을
``/content/drive/MyDrive/GoogleAI_Capstone_Results/reproduction_lee_lee_2025_vae/``
아래 **타임스탬프 폴더**에 저장한다. 전체 실행은 모든 단계가 한 폴더를 공유하며,
이전 실행 결과를 덮어쓰지 않는다. ``--out-root`` 또는 ``VAE2025_OUT_ROOT``로 변경 가능하다.
자세한 우선순위는 :func:`_resolve_output_root` 참조.

GPU / 런타임
------------
데이터·모델 규모가 작다(최대 학습 행 9,746개, 46 feature, 은닉층 최대 512).
``XGBoost``는 ``tree_method="hist"``로 CPU에서 돌며 GPU 이득이 없다.
반면 기본 실행 목록(``run.models``)의 ``DNN``·``TabNet``·``Wide & Deep``과
``augmentation.method: vae``는 torch/pytorch-tabnet 기반이라 GPU가 있으면 유의미하게 빨라진다.
**Colab Pro+에서 T4(표준) GPU면 충분하다. A100/L4는 이 규모에 낭비다.**
실험 C는 outer 3 × 후보 24 × inner 3 + outer 3 = 219회 모델 적합이므로
``search.max_evals``를 줄이거나 ``--fold``로 나눠 실행하는 것을 권장한다.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("run")

EXPERIMENT_DISPATCH = {
    "paper_reported_reconstruction": "A",
    "leakage_controlled_non_nested": "B",
    "nested_subject_independent": "C",
}

#: base.ipynb Cell 1이 이 이름 그대로 만드는 Colab 공유 드라이브 경로.
COLAB_SHARED_DATA_ROOT = Path("/content/drive/Shareddrives/GoogleAI_contest/Data")
#: 저장소 어디서든 데이터 경로를 override할 때 쓰는 환경변수 (다른 SangHyo 실험과 동일 이름).
DATA_ROOT_ENV_VAR = "SANGHYO_DATA_ROOT"
#: 노트북에서 CLI 인자를 넘기고 싶을 때 쓰는 환경변수.
ARGS_ENV_VAR = "VAE2025_ARGS"
#: 산출물 저장 위치를 바꾸고 싶을 때 쓰는 환경변수.
OUTPUT_ROOT_ENV_VAR = "VAE2025_OUT_ROOT"

#: Colab에서 Google Drive가 마운트되는 위치 (base.ipynb Cell 1이 drive.mount("/content/drive")).
COLAB_MYDRIVE = Path("/content/drive/MyDrive")
#: MyDrive 안에 만들 산출물 폴더. 런타임이 죽어도 여기 있는 결과는 남는다.
DRIVE_OUTPUT_SUBDIR = "GoogleAI_Capstone_Results/reproduction_lee_lee_2025_vae"


def _is_writable_dir(path: Path) -> bool:
    """디렉터리를 만들고 실제로 쓸 수 있는지 확인한다 (Drive FUSE는 마운트만 되고
    쓰기가 안 되는 경우가 있어 touch까지 해본다)."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:  # noqa: BLE001 - 어떤 이유로든 못 쓰면 폴백한다
        return False


def _resolve_output_root(namespace: dict[str, Any], explicit: str | None, cfg=None) -> Path:
    """산출물 **기준** 디렉터리를 정한다.

    우선순위:

    1. ``--out-root``
    2. ``VAE2025_OUT_ROOT`` 환경변수
    3. config ``output.root``가 절대경로인 경우
    4. **Colab에서 Drive가 마운트되어 있으면 MyDrive 안의 전용 폴더** ← 기본 동작
    5. 저장소 로컬 ``outputs/``

    4번이 핵심이다. base.ipynb는 저장소를 ``/content/Google-Ajou-AICapstone``에 clone하는데
    ``/content``는 **런타임이 끊기면 사라진다.** 기본값을 저장소 안(``REPO_ROOT/outputs``)으로
    두면 몇 시간짜리 실험 결과가 통째로 날아간다. 그래서 Drive를 우선한다.
    """
    for candidate in (explicit, os.environ.get(OUTPUT_ROOT_ENV_VAR)):
        if candidate:
            p = Path(candidate).expanduser()
            if not p.is_absolute():
                p = (REPO_ROOT / p).resolve()
            return p

    if cfg is not None:
        configured = cfg.get_path("output.root")
        if configured and Path(configured).is_absolute():
            return Path(configured)

    # Colab: Drive가 마운트되어 쓰기 가능하면 MyDrive에 저장한다.
    mydrive = COLAB_MYDRIVE
    project = namespace.get("PROJECT_ROOT")
    ephemeral = project is not None and str(project).startswith("/content/")
    if mydrive.exists() and _is_writable_dir(mydrive / DRIVE_OUTPUT_SUBDIR):
        return mydrive / DRIVE_OUTPUT_SUBDIR
    if ephemeral:
        log.warning(
            "PROJECT_ROOT가 /content 아래(런타임 임시 저장소)인데 Google Drive를 쓸 수 없다. "
            "결과가 런타임 종료 시 사라진다. base.ipynb Cell 1의 drive.mount가 "
            "성공했는지 확인하거나 --out-root로 영구 경로를 지정하라."
        )

    fallback = cfg.get_path("output.root", "outputs") if cfg is not None else "outputs"
    p = Path(fallback)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def _make_session_dir(base: Path, tag: str = "run") -> Path:
    """전체 실행(run_all)용 타임스탬프 디렉터리. 이전 스윕 결과를 덮어쓰지 않는다."""
    session = base / f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}"
    session.mkdir(parents=True, exist_ok=True)
    try:
        (base / "LATEST.txt").write_text(session.name + "\n", encoding="utf-8")
    except Exception:  # noqa: BLE001 - 편의 기능일 뿐이다
        pass
    return session

#: 한 번에 실행(run_all)할 때 쓰는 실험 A config.
#: A1(percentile)은 Dem이 6행만 남아 8:1:1 분할이 불가능하므로(I-1 증거 F)
#: 기본 파이프라인은 A3(Isolation Forest)를 쓴다. A1은 실패를 기록만 하고 넘어간다.
RUN_ALL_CONFIGS = {
    "A": "configs/paper_isoforest_latent500.yaml",
    "B": "configs/leakage_controlled_non_nested.yaml",
    "C": "configs/nested_subject_independent.yaml",
}
RUN_ALL_A_PRIMARY = "configs/paper_percentile_latent500.yaml"


def _running_under_kernel() -> bool:
    """base.ipynb Cell 5처럼 Jupyter/Colab 커널 안에서 실행 중인지 판정한다.

    이 경우 ``sys.argv``는 노트북 인자가 아니라 커널 런처의 인자이므로
    argparse에 넘기면 안 된다.
    """
    try:
        get_ipython  # type: ignore[name-defined]  # noqa: B018
        return True
    except NameError:
        pass
    if "ipykernel" in sys.modules or "google.colab" in sys.modules:
        return True
    prog = Path(sys.argv[0]).name.lower() if sys.argv else ""
    if any(k in prog for k in ("ipykernel_launcher", "colab_kernel_launcher", "colab-fileshim")):
        return True
    rest = sys.argv[1:]
    return "-f" in rest and any("kernel-" in a and a.endswith(".json") for a in rest)


def _resolve_argv(argv: list[str] | None) -> list[str] | None:
    """실제로 사용할 CLI 인자를 정한다.

    Returns:
        인자 리스트, 또는 ``None``(= 전체 파이프라인을 한 번에 실행하라는 뜻).
    """
    if argv is not None:
        return list(argv)
    env = os.environ.get(ARGS_ENV_VAR)
    if env is not None and env.strip():
        log.info("%s에서 인자를 읽는다: %s", ARGS_ENV_VAR, env)
        return shlex.split(env)
    if _running_under_kernel():
        return None
    return sys.argv[1:]

#: --inspect-data / --dry-run 은 이 목록만 있으면 동작한다.
_CORE_DEPS = {"numpy": "numpy", "pandas": "pandas", "scikit-learn": "sklearn",
              "scipy": "scipy", "pyyaml": "yaml"}
_TRAIN_DEPS = {"torch": "torch", "xgboost": "xgboost", "pytorch-tabnet": "pytorch_tabnet",
               "imbalanced-learn": "imblearn"}


def _resolve_data_root(namespace: dict[str, Any], explicit: str | None) -> Path:
    """Data/ 경로를 찾는다. 우선순위: 명시 인자 > namespace(DATA_ROOT/PROJECT_ROOT) >
    환경변수 > 저장소 로컬 Data/ > Colab 공유 드라이브.

    ``namespace``는 base.ipynb Cell 1이 만든 ``globals()``를 그대로 받는 용도다
    (``run_pipeline(namespace=globals(), ...)``로 호출될 때).
    """
    import os

    candidates: list[str | Path] = []
    if explicit:
        candidates.append(explicit)
    if namespace.get("DATA_ROOT"):
        candidates.append(namespace["DATA_ROOT"])
    if os.environ.get(DATA_ROOT_ENV_VAR):
        candidates.append(os.environ[DATA_ROOT_ENV_VAR])
    if namespace.get("PROJECT_ROOT"):
        candidates.append(Path(namespace["PROJECT_ROOT"]) / "Data")
    candidates += [
        (REPO_ROOT / "../../../Data").resolve(),
        COLAB_SHARED_DATA_ROOT,
    ]
    for c in candidates:
        p = Path(c)
        if (p / "1.Training").exists():
            return p
    raise FileNotFoundError(
        "Data/ 를 찾지 못했다. 시도한 경로: " + ", ".join(str(Path(c)) for c in candidates) +
        f". --data-root로 명시하거나 {DATA_ROOT_ENV_VAR} 환경변수를 설정하라."
    )


def _ensure_dependencies(*, need_training: bool, skip: bool = False) -> dict[str, bool]:
    """필수 의존성을 확인하고, 없으면 설치한다.

    base.ipynb Cell 4는 ``SangHyo/requirements.txt``만 보고 이 폴더처럼 중첩된
    ``requirements.txt``는 찾지 못하므로, 이 스크립트가 스스로 설치를 책임진다.

    학습이 필요할 때는 버전을 고정한 ``requirements_colab.txt``로 설치한다
    (torch는 포함하지 않는다 — Colab이 CUDA와 맞물려 미리 설치해 둔 것을 건드리면
    버전 불일치가 나므로, torch는 없을 때만 별도로 최신 버전을 설치한다).
    """
    missing_core = [pkg for pkg, mod in _CORE_DEPS.items() if importlib.util.find_spec(mod) is None]
    if missing_core and not skip:
        log.info("설치(core): %s", ", ".join(missing_core))
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *missing_core],
            check=True,
        )
    elif missing_core:
        raise ModuleNotFoundError(f"누락된 의존성: {', '.join(missing_core)} (--skip-install 해제 필요)")

    if need_training:
        missing_train = [pkg for pkg, mod in _TRAIN_DEPS.items() if importlib.util.find_spec(mod) is None]
        if missing_train and not skip:
            req_file = REPO_ROOT / "requirements_colab.txt"
            log.info("설치(training, 버전 고정): %s", req_file)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                 "-r", str(req_file)],
                check=True,
            )
            if importlib.util.find_spec("torch") is None:
                log.info("torch 미설치 — 최신 버전 설치 (Colab이면 보통 이미 있어 이 분기를 타지 않는다)")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "torch"],
                    check=True,
                )
        elif missing_train:
            raise ModuleNotFoundError(f"누락된 의존성: {', '.join(missing_train)} (--skip-install 해제 필요)")
        importlib.invalidate_caches()

    return {mod: importlib.util.find_spec(mod) is not None for mod in (*_TRAIN_DEPS.values(), "torch")}


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
    p.add_argument("--skip-install", action="store_true", help="의존성 자동 설치를 건너뛴다")
    p.add_argument("--all", action="store_true",
                   help="점검·감사·실험 A/B/C·비교표를 한 번에 실행 (노트북 기본 동작)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def run_pipeline(*, namespace: dict[str, Any] | None = None, argv: list[str] | None = None) -> int:
    """실제 진입점.

    ``argv``가 ``None``이면 실행 환경을 보고 스스로 정한다:

    * 일반 셸 → ``sys.argv[1:]``
    * ``VAE2025_ARGS`` 환경변수가 있으면 그것을 파싱
    * Jupyter/Colab 커널(= base.ipynb Cell 5) → 커널 argv를 무시하고
      :func:`run_all`로 전체 파이프라인을 한 번에 실행
    """
    namespace = globals() if namespace is None else namespace
    resolved = _resolve_argv(argv)
    if resolved is None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        return run_all(namespace=namespace)

    args = build_parser().parse_args(resolved)
    if args.all:
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        return run_all(namespace=namespace, seed=args.seed, skip_install=args.skip_install)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    no_training = args.inspect_data or args.dry_run or args.audit_only
    _ensure_dependencies(need_training=not no_training, skip=args.skip_install)

    from src.data.inspect import format_inspection, inspect_data, percentile_retention_scan
    from src.data.loader import load_lifelog
    from src.utils.config import load_config
    from src.utils.io import save_json, save_table
    from src.utils.seeding import set_global_seed

    # ---------------- config
    if args.config:
        cfg = load_config(args.config)
    elif args.inspect_data:
        cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    else:
        log.error("--config가 필요하다 (또는 --inspect-data)")
        return 2

    data_root = _resolve_data_root(namespace, args.data_root)
    out_root = _resolve_output_root(namespace, args.out_root, cfg)
    Path(out_root).mkdir(parents=True, exist_ok=True)
    seed = args.seed if args.seed is not None else int(cfg.get_path("seed", 42))
    set_global_seed(seed)

    label = cfg.get_path("label", Path(args.config).stem if args.config else "base")
    log.info("config=%s label=%s seed=%d", cfg.get_path("_meta.config_path"), label, seed)
    log.info("data_root=%s", data_root)
    log.info("out_root =%s", out_root)

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
    # run_all이 교차 비교표를 만들 수 있도록 결과 dict를 그대로 돌려준다.
    # (셸 종료코드로 쓰이는 경로에서는 main()이 int가 아닌 값을 0으로 처리한다.)
    return res


def _dry_run(data, cfg, kind, out_root, label, seed, args) -> int:
    """학습 없이 검증만 수행한다 (사용자 지시 14절)."""
    from src.data.schema import PAPER_FEATURES
    from src.audit.checks import check_forbidden_features
    from src.utils.io import save_json

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
    from src.utils.io import save_table

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


# ======================================================================================
# 한 번에 실행 (base.ipynb Cell 5 기본 동작)
# ======================================================================================

def _stage(name: str, index: int, total: int) -> None:
    print("\n" + "=" * 78, flush=True)
    print(f"[{index}/{total}] {name}", flush=True)
    print("=" * 78, flush=True)


def run_all(
    *,
    namespace: dict[str, Any] | None = None,
    seed: int | None = None,
    skip_install: bool = False,
) -> int:
    """점검 → 감사 → dry-run → 실험 A·B·C → 비교표를 한 번에 실행한다.

    각 단계는 독립적으로 실패를 흡수한다. 예컨대 primary config(A1)는
    논문 §5.1 본문 방식이 Dem 6행만 남겨 8:1:1 분할이 불가능하므로 반드시 실패하는데
    (report_inconsistencies.md I-1 증거 F), 그 실패는 **결과의 일부**이므로
    기록하고 다음 단계로 넘어간다.
    """
    namespace = globals() if namespace is None else namespace
    started = time.monotonic()

    # 산출물 위치를 **한 번만** 정하고 모든 단계에 --out-root로 넘긴다.
    # 단계마다 따로 정하면 타임스탬프가 어긋나 결과가 여러 폴더로 흩어진다.
    base_root = _resolve_output_root(namespace, None, None)
    session = _make_session_dir(base_root, tag="full")
    out_root = str(session)

    print("=" * 78, flush=True)
    print(f"산출물 저장 위치: {out_root}", flush=True)
    if str(session).startswith("/content/") and "/drive/" not in str(session):
        print("⚠️  경고: 런타임 임시 저장소다. 세션이 끊기면 결과가 사라진다.", flush=True)
    else:
        print("   (런타임이 끊겨도 유지된다)", flush=True)
    print("=" * 78, flush=True)

    common: list[str] = ["--out-root", out_root]
    if seed is not None:
        common += ["--seed", str(seed)]
    if skip_install:
        common.append("--skip-install")

    # (표시 이름, argv, 결과를 어느 실험으로 수집할지)
    stages: list[tuple[str, list[str], str | None]] = [
        ("데이터 점검 (--inspect-data)", ["--inspect-data"], None),
        ("이상치 방식 검증 (--audit-only)",
         ["--config", RUN_ALL_CONFIGS["A"], "--audit-only"], None),
        ("실험 A primary(A1) 실행 가능성 확인 — 실패가 예상되며 그 자체가 결과다",
         ["--config", RUN_ALL_A_PRIMARY, "--dry-run"], None),
        ("실험 A dry-run", ["--config", RUN_ALL_CONFIGS["A"], "--dry-run"], None),
        ("실험 B dry-run", ["--config", RUN_ALL_CONFIGS["B"], "--dry-run"], None),
        ("실험 C dry-run", ["--config", RUN_ALL_CONFIGS["C"], "--dry-run"], None),
        ("실험 A 실행 (논문 방법 재구성)", ["--config", RUN_ALL_CONFIGS["A"]], "A"),
        ("실험 B 실행 (누수 통제 non-nested)", ["--config", RUN_ALL_CONFIGS["B"]], "B"),
        ("실험 C 실행 (Nested Group CV)", ["--config", RUN_ALL_CONFIGS["C"]], "C"),
    ]

    from src.experiments.compare import CrossExperimentResults

    collected = CrossExperimentResults()
    collect = {"A": collected.add_experiment_a,
               "B": collected.add_experiment_b,
               "C": collected.add_experiment_c}
    outcomes: list[dict] = []
    total = len(stages) + 1

    for i, (name, argv, kind) in enumerate(stages, start=1):
        _stage(name, i, total)
        t0 = time.monotonic()
        try:
            result = run_pipeline(namespace=namespace, argv=argv + common)
            ok = isinstance(result, dict) or result in (0, None)
            if kind and isinstance(result, dict):
                collect[kind](result)
            outcomes.append({"stage": name, "status": "ok" if ok else "nonzero_exit",
                             "seconds": round(time.monotonic() - t0, 1)})
        except Exception as error:  # noqa: BLE001 - 단계 실패를 결과로 남긴다
            outcomes.append({
                "stage": name, "status": "failed",
                "error_type": type(error).__name__, "error": str(error),
                "seconds": round(time.monotonic() - t0, 1),
            })
            print(f"\n[단계 실패] {type(error).__name__}: {error}", flush=True)
            log.debug("%s", traceback.format_exc())

    # ---- 비교표
    _stage("교차 실험 비교표 생성", total, total)
    summary: dict = {}
    try:
        from src.experiments.compare import assemble_comparison

        summary = assemble_comparison(collected, out_root=out_root)
        print(f"비교표 저장 -> {Path(out_root) / 'COMPARISON'}", flush=True)
    except Exception as error:  # noqa: BLE001
        print(f"[비교표 실패] {type(error).__name__}: {error}", flush=True)
        log.debug("%s", traceback.format_exc())

    from src.utils.io import save_json

    elapsed = time.monotonic() - started
    save_json(
        {"stages": outcomes, "comparison": summary, "elapsed_seconds": round(elapsed, 1)},
        Path(out_root) / "RUN_ALL_SUMMARY.json",
    )

    print("\n" + "=" * 78, flush=True)
    print(f"전체 완료 — {int(elapsed // 60):02d}:{int(elapsed % 60):02d} 소요", flush=True)
    for o in outcomes:
        mark = {"ok": "✅", "nonzero_exit": "⚠️", "failed": "❌"}.get(o["status"], "?")
        print(f"  {mark} {o['stage']}  ({o['seconds']}s)", flush=True)
    print(f"\n산출물: {out_root}", flush=True)
    if COLAB_MYDRIVE.exists() and str(out_root).startswith(str(COLAB_MYDRIVE)):
        rel = Path(out_root).relative_to(COLAB_MYDRIVE)
        print(f"  → Google Drive > 내 드라이브 > {rel}", flush=True)
        print("  런타임이 끊겨도 유지된다.", flush=True)
    print("=" * 78, flush=True)
    return 0


def main() -> None:
    rc = run_pipeline()
    # 노트북 안에서 SystemExit를 던지면 셀이 예외로 붉게 표시되므로 셸에서만 던진다.
    if not _running_under_kernel():
        raise SystemExit(rc if isinstance(rc, int) else 0)


if __name__ == "__main__":
    main()
