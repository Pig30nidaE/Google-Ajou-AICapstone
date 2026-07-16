#!/usr/bin/env python3
"""Generate the clean-output Training-only Colab discovery notebook."""

from __future__ import annotations

import json
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = LAB_ROOT / "01_train_only_discovery_colab.ipynb"


def source_lines(text: str) -> list[str]:
    text = text.strip("\n") + "\n"
    return text.splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines(text),
    }


cells = [
    markdown(
        r"""
# PerformanceLab - Training-only CN/MCI/DEM discovery

이 notebook은 wearable activity/sleep만 사용해 `CN=0`, `MCI=1`, `DEM=2`를
subject 단위로 분류한다. 미래 전환 예측이 아니라 마지막 activity 관측 시점의
동시점 상태 분류다.

중요 계약:

- 이 notebook은 Training 경로만 알고 있으며 역사적 benchmark 경로·라벨을
  탐색하지 않는다.
- MMSE, 진단 필드, ID, 절대 날짜는 feature가 아니다.
- 모든 학습된 전처리·모델 선택은 subject-grouped nested CV 안에서 수행한다.
- `FAST_MODE=True`는 기능 점검일 뿐 성능 결과나 frozen run이 아니다.
- Full run 결과가 사전등록 GO gate를 통과하기 전에는 benchmark 단계로 가지 않는다.
"""
    ),
    code(
        r"""
import json
import os
import subprocess
import sys
from importlib import metadata as bootstrap_metadata

# PyTorch strict CUDA determinism requires this before the first CUDA context.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Install before importing google.colab or any ABI-sensitive scientific package.
PINNED_PACKAGES = [
    "numpy==2.2.6",
    "pandas==2.3.3",
    "scipy==1.16.3",
    "scikit-learn==1.7.2",
    "joblib==1.5.2",
    "pyarrow==21.0.0",
    "matplotlib==3.10.7",
    "seaborn==0.13.2",
]
PACKAGE_TO_MODULE = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "joblib": "joblib",
    "pyarrow": "pyarrow",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
}
preloaded_modules = {
    package: module in sys.modules for package, module in PACKAGE_TO_MODULE.items()
}
versions_before_install = {}
for package in PACKAGE_TO_MODULE:
    try:
        versions_before_install[package] = bootstrap_metadata.version(package)
    except bootstrap_metadata.PackageNotFoundError:
        versions_before_install[package] = None
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", *PINNED_PACKAGES],
    check=True,
)
versions_after_install = {
    package: bootstrap_metadata.version(package) for package in PACKAGE_TO_MODULE
}
changed_while_preloaded = {
    package: {"before": versions_before_install[package], "after": versions_after_install[package]}
    for package, was_loaded in preloaded_modules.items()
    if was_loaded and versions_before_install[package] != versions_after_install[package]
}
if changed_while_preloaded:
    raise RuntimeError(
        "An already-imported scientific package changed version. Restart a clean runtime and run again: "
        + json.dumps(changed_while_preloaded, sort_keys=True)
    )
pip_check = subprocess.run(
    [sys.executable, "-m", "pip", "check"],
    check=False,
    capture_output=True,
    text=True,
)
PIP_CHECK_REPORT = {
    "return_code": pip_check.returncode,
    "output": (pip_check.stdout + pip_check.stderr).strip()[:8000],
}
if pip_check.returncode:
    print("Global Colab pip check reported a preloaded-environment conflict; locked imports are verified next.")
"""
    ),
    code(
        r"""
from google.colab import drive
drive.mount("/content/drive")
"""
    ),
    markdown(
        r"""
## 1. 단일 설정 cell

처음에는 `FAST_MODE=True`로 smoke test하고, clean A100 High-RAM runtime에서
`False`로 바꿔 Full run을 실행한다. 세 override는 Drive 배치가 기본 후보와 다를
때만 지정한다.
"""
    ),
    code(
        r"""
from pathlib import Path

PROJECT_ROOT_OVERRIDE = None
TRAINING_ROOT_OVERRIDE = None
OUTPUT_BASE_OVERRIDE = None

FAST_MODE = False
REQUIRE_A100 = True
MIN_RAM_GIB = 40
RANDOM_SEED = 137

project_candidates = [
    Path("/content/drive/MyDrive/GoogleAI_contest/AI_Capstone_Project"),
    Path("/content/drive/MyDrive/GoogleAI_contest"),
    Path("/content/drive/Shareddrives/GoogleAI_contest/AI_Capstone_Project"),
    Path("/content/drive/Shareddrives/GoogleAI_contest"),
]
PROJECT_ROOT = Path(PROJECT_ROOT_OVERRIDE) if PROJECT_ROOT_OVERRIDE else next(
    (
        path for path in project_candidates
        if (path / "ThreeClass_PerformanceLab" / "performance_lab_core.py").is_file()
        and (path / "Data" / "1.Training").is_dir()
    ),
    None,
)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Project root was not found. Set PROJECT_ROOT_OVERRIDE.")

LAB_ROOT = PROJECT_ROOT / "ThreeClass_PerformanceLab"
TRAINING_ROOT = (
    Path(TRAINING_ROOT_OVERRIDE)
    if TRAINING_ROOT_OVERRIDE
    else PROJECT_ROOT / "Data" / "1.Training"
)
if TRAINING_ROOT.name != "1.Training" or not TRAINING_ROOT.is_dir():
    raise ValueError("TRAINING_ROOT must point exactly to the 1.Training directory.")
OUTPUT_BASE = (
    Path(OUTPUT_BASE_OVERRIDE)
    if OUTPUT_BASE_OVERRIDE
    else Path("/content/drive/MyDrive/GoogleAI_contest/outputs/ThreeClass_PerformanceLab")
)

print({
    "project_root": str(PROJECT_ROOT),
    "training_root": str(TRAINING_ROOT),
    "output_base": str(OUTPUT_BASE),
    "fast_mode": FAST_MODE,
})
"""
    ),
    markdown(
        r"""
## 2. 재현 가능한 실행환경 확인

CUDA와 맞물린 PyTorch는 Colab A100 runtime 제공 build를 그대로 사용하고 정확한
버전을 기록한다. CPU-side 고정 버전은 첫 cell에서 import 전에 설치했으며,
repository requirements와 일치하는지 다시 검사한다.
"""
    ),
    code(
        r"""
import getpass
import hashlib
import hmac
import importlib
import json
import os
import platform
import shutil
import time
from datetime import datetime, timezone
from importlib import metadata

import joblib
import matplotlib
import numpy as np
import pandas as pd
import pyarrow
import scipy
import seaborn
import sklearn
import torch
from google.colab import userdata

expected_versions = {
    item.split("==", 1)[0].replace("scikit-learn", "scikit_learn"): item.split("==", 1)[1]
    for item in PINNED_PACKAGES
}
actual_versions = {
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scipy": scipy.__version__,
    "scikit_learn": sklearn.__version__,
    "joblib": joblib.__version__,
    "pyarrow": pyarrow.__version__,
    "matplotlib": matplotlib.__version__,
    "seaborn": seaborn.__version__,
}
if actual_versions != expected_versions:
    raise RuntimeError({"expected_versions": expected_versions, "actual_versions": actual_versions})

requirements_pins = {
    line.strip().replace("scikit-learn", "scikit_learn")
    for line in (LAB_ROOT / "requirements_colab.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
}
runtime_pins = {item.replace("scikit-learn", "scikit_learn") for item in PINNED_PACKAGES}
if requirements_pins != runtime_pins:
    raise RuntimeError("Notebook pins and requirements_colab.txt differ.")

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"
ram_gib = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
if REQUIRE_A100 and "A100" not in gpu_name.upper():
    raise RuntimeError(f"A100 is required for the Full contract; allocated GPU={gpu_name!r}")
if not FAST_MODE and ram_gib < MIN_RAM_GIB:
    raise RuntimeError(f"High-RAM runtime required: detected {ram_gib:.1f} GiB")

try:
    SUBJECT_HASH_KEY = userdata.get("SUBJECT_HASH_KEY")
except Exception:
    SUBJECT_HASH_KEY = None
if not SUBJECT_HASH_KEY:
    SUBJECT_HASH_KEY = getpass.getpass("SUBJECT_HASH_KEY (32+ chars, input hidden): ")
if len(SUBJECT_HASH_KEY) < 32:
    raise ValueError("SUBJECT_HASH_KEY must contain at least 32 characters.")

environment = {
    "runtime_started_utc": datetime.now(timezone.utc).isoformat(),
    "python": platform.python_version(),
    "platform": platform.platform(),
    "processor": platform.processor(),
    "cpu_count": os.cpu_count(),
    "timezone": list(time.tzname),
    **actual_versions,
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "gpu": gpu_name,
    "ram_gib": round(ram_gib, 2),
    "mixed_precision": "disabled_locked_float32",
    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    "fast_mode": FAST_MODE,
    "global_seed": RANDOM_SEED,
    "model_seeds": [17011, 27011],
    "pip_check": PIP_CHECK_REPORT,
}

sys.path.insert(0, str(LAB_ROOT))
import performance_lab_core as plc
importlib.reload(plc)
plc.set_all_seeds(RANDOM_SEED)
environment["torch_deterministic_algorithms_enabled"] = (
    torch.are_deterministic_algorithms_enabled()
)
if environment["cublas_workspace_config"] != ":4096:8":
    raise RuntimeError("Strict CUDA determinism requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
if not environment["torch_deterministic_algorithms_enabled"]:
    raise RuntimeError("Strict PyTorch deterministic algorithms were not enabled.")
attempt_started = time.perf_counter()
print(environment)
"""
    ),
    markdown(
        r"""
## 3. Input fingerprint와 새/resume run

Run ID는 locked config, notebook/core/requirements/design, 실제 runtime 계약,
Training input fingerprint, hash-key verifier와 smoke/full mode에서 결정된다.
완료된 run은 덮어쓰지 않고, 미완료 run은 identity가 완전히 같을 때만 outer-fold
checkpoint에서 재개한다.
"""
    ),
    code(
        r"""
locked_config_path = LAB_ROOT / "config" / "locked_discovery_v1.json"
locked_config = json.loads(locked_config_path.read_text(encoding="utf-8"))
if RANDOM_SEED != locked_config["cv"]["global_seed"]:
    raise RuntimeError("Notebook RANDOM_SEED differs from the locked config.")
if environment["model_seeds"] != locked_config["cv"]["model_seeds"]:
    raise RuntimeError("Notebook model seeds differ from the locked config.")

input_paths = plc.training_input_paths(TRAINING_ROOT)
input_manifest = plc.build_input_manifest(input_paths)
source_paths = {
    "notebook": LAB_ROOT / "01_train_only_discovery_colab.ipynb",
    "core": LAB_ROOT / "performance_lab_core.py",
    "requirements": LAB_ROOT / "requirements_colab.txt",
    "design": LAB_ROOT / "EXPERIMENT_DESIGN_KO.md",
    "config": locked_config_path,
}
def canonical_notebook_source_sha256(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    canonical = {
        "nbformat": payload.get("nbformat"),
        "nbformat_minor": payload.get("nbformat_minor"),
        "cells": [
            {"cell_type": cell.get("cell_type"), "source": cell.get("source", [])}
            for cell in payload.get("cells", [])
        ],
    }
    return hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

source_hashes = {
    name: (
        canonical_notebook_source_sha256(path)
        if name == "notebook"
        else plc.sha256_file(path)
    )
    for name, path in source_paths.items()
}
key_verifier = hmac.new(
    SUBJECT_HASH_KEY.encode("utf-8"),
    b"PerformanceLab-v1-key-verifier",
    hashlib.sha256,
).hexdigest()
environment_contract = {
    key: environment[key]
    for key in [
        "python", "numpy", "pandas", "scipy", "scikit_learn", "joblib",
        "pyarrow", "matplotlib", "seaborn", "torch", "cuda_runtime", "cudnn",
        "gpu", "mixed_precision", "cublas_workspace_config",
        "torch_deterministic_algorithms_enabled", "global_seed", "model_seeds",
    ]
}
identity_payload = {
    "config": locked_config,
    "source_hashes": source_hashes,
    "inputs": input_manifest,
    "environment_contract": environment_contract,
    "subject_hash_key_verifier": key_verifier,
    "global_seed": RANDOM_SEED,
    "fast_mode": FAST_MODE,
}
run_digest = hashlib.sha256(
    json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()[:12]
run_id = f"{'fast' if FAST_MODE else 'full'}_event28_{run_digest}"
RUN_DIR = OUTPUT_BASE / run_id
if RUN_DIR.exists():
    run_config_path = RUN_DIR / "run_config.json"
    if not run_config_path.is_file():
        raise RuntimeError("Existing partial directory lacks immutable run_config.json.")
    existing_identity = json.loads(run_config_path.read_text(encoding="utf-8"))
    if existing_identity != identity_payload:
        raise RuntimeError("Resume identity mismatch; start a new run directory.")
    if (RUN_DIR / "TRAINING_COMPLETE.json").exists():
        raise RuntimeError(f"Completed run is immutable: {RUN_DIR}")
    if (RUN_DIR / "RUN_INVALID_PRIVACY.json").exists():
        raise RuntimeError(
            "A prior privacy failure permanently invalidated this run directory; "
            "do not resume or delete the evidence. Start a clean run with a new identity."
        )
    resume_mode = True
else:
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    plc.write_json(RUN_DIR / "run_config.json", identity_payload)
    plc.write_json(RUN_DIR / "training_input_manifest.json", input_manifest)
    resume_mode = False

snapshot = RUN_DIR / "code_snapshot"
if resume_mode:
    for name, original in source_paths.items():
        snap = snapshot / original.name
        snapshot_hash = (
            canonical_notebook_source_sha256(snap)
            if name == "notebook" and snap.is_file()
            else plc.sha256_file(snap) if snap.is_file() else None
        )
        if not snap.is_file() or snapshot_hash != source_hashes[name]:
            raise RuntimeError(f"Code snapshot mismatch on resume: {name}")
else:
    snapshot.mkdir(exist_ok=False)
    for original in source_paths.values():
        shutil.copy2(original, snapshot / original.name)

environment_path = RUN_DIR / "environment.json"
if environment_path.exists():
    environment_history = json.loads(environment_path.read_text(encoding="utf-8"))
    if environment_history["environment_contract"] != environment_contract:
        raise RuntimeError("Runtime contract changed during resume.")
else:
    environment_history = {"environment_contract": environment_contract, "resume_events": []}
environment_history["resume_events"].append({
    "started_utc": environment["runtime_started_utc"],
    "ram_gib": environment["ram_gib"],
    "resume_mode": resume_mode,
})
plc.write_json(environment_path, environment_history)

print({"run_id": run_id, "run_dir": str(RUN_DIR), "resume_mode": resume_mode})
"""
    ),
    markdown(
        r"""
## 4. Training-only load, audit, deterministic feature construction

출력에는 원본 ID가 나오지 않는다. ID는 메모리에서 modality 정렬과 split에만 쓰고,
저장 시 keyed hash로 치환한다.
"""
    ),
    code(
        r"""
training = plc.load_training_dataset(TRAINING_ROOT)
labels = training.labels
class_counts = {plc.CLASS_NAMES[key]: int((labels == key).sum()) for key in range(3)}
if len(labels) != 141 or class_counts != {"CN": 85, "MCI": 47, "DEM": 9}:
    raise AssertionError(f"Training contract mismatch: n={len(labels)}, counts={class_counts}")

print({
    "training_subjects": len(labels),
    "class_counts": class_counts,
    "activity_rows": len(training.activity),
    "sleep_rows": len(training.sleep),
    "label_copies_consistent": training.audit["label_copies_consistent"],
})

bundle = plc.build_feature_bundle(training.activity, training.sleep, training.subject_ids)
plc.validate_feature_bundle(bundle, expected_subjects=141)
plc.write_json(RUN_DIR / "data_audit.json", {**training.audit, **bundle.diagnostics})
plc.write_json(RUN_DIR / "feature_manifest_event_summary.json", bundle.summary_manifest())
plc.write_json(RUN_DIR / "feature_manifest_event_sequence.json", bundle.sequence_manifest())
plc.write_json(RUN_DIR / "coverage_audit.json", bundle.coverage_audit())

print(bundle.public_summary())
"""
    ),
    markdown(
        r"""
## 5. Fresh repeated nested subject CV

Full mode는 outer 5 seeds x 3 folds, 각 outer-train 안에서 inner 2 seeds x 3 folds를
실행한다. 모델 선택, class weight, imputation/scaling은 outer-valid를 보지 않는다.
Checkpoint가 있으면 동일 hash의 완료 outer fold만 건너뛴다.
"""
    ),
    code(
        r"""
started = time.perf_counter()
nested_result = plc.run_nested_cv(
    bundle=bundle,
    labels=labels,
    subject_ids=training.subject_ids,
    output_dir=RUN_DIR,
    subject_hash_key=SUBJECT_HASH_KEY,
    locked_config=locked_config,
    fast_mode=FAST_MODE,
    resume=True,
)
elapsed_seconds = time.perf_counter() - started
environment["nested_cv_elapsed_seconds"] = elapsed_seconds
environment_history["resume_events"][-1].update({
    "nested_cv_elapsed_seconds": elapsed_seconds,
    "completed_utc": datetime.now(timezone.utc).isoformat(),
})
plc.write_json(RUN_DIR / "environment.json", environment_history)

print({
    "nested_macro_f1_mean": nested_result["summary"]["macro_f1_mean"],
    "nested_macro_f1_sd": nested_result["summary"]["macro_f1_sd"],
    "elapsed_minutes": round(elapsed_seconds / 60, 2),
})

# A privacy failure is an immediate STOP.  No selection decision or frozen
# model may be created until all nested-CV artifacts pass the byte scan.
pre_freeze_privacy = plc.audit_output_privacy(
    output_dir=RUN_DIR,
    raw_subject_ids=training.subject_ids,
    forbidden_secret=SUBJECT_HASH_KEY,
)
plc.write_json(RUN_DIR / "privacy_audit_pre_freeze.json", pre_freeze_privacy)
if not pre_freeze_privacy["passed"]:
    plc.write_json(
        RUN_DIR / "RUN_INVALID_PRIVACY.json",
        {
            "stage": "before_selection_and_freeze",
            "decision": "INVALID-NO-GO",
            "offending_file_count": pre_freeze_privacy["offending_file_count"],
        },
    )
    raise RuntimeError("Pre-freeze privacy audit failed; no model was frozen.")
"""
    ),
    markdown(
        r"""
## 6. 사전등록 selection 및 STOP/GO

Legacy calendar TCN과 coverage-only control은 winner가 될 수 없다. Full run이 모든
GO gate를 통과할 때만 Training 141명 refit/freeze를 수행한다. Smoke run은 항상
NO-GO다.
"""
    ),
    code(
        r"""
decision = plc.select_and_assess(nested_result, locked_config, fast_mode=FAST_MODE)
plc.write_json(RUN_DIR / "selection_report.json", decision["selection"])
plc.write_json(RUN_DIR / "stop_go_decision.json", decision["stop_go"])
print(json.dumps(decision["stop_go"], ensure_ascii=False, indent=2))

frozen = None
if decision["stop_go"]["decision"] == "GO":
    frozen = plc.fit_frozen_training_bundle(
        bundle=bundle,
        labels=labels,
        selected_candidate=decision["selection"]["selected_candidate"],
        output_dir=RUN_DIR,
        locked_config=locked_config,
        nested_result=nested_result,
        decision=decision,
    )
    plc.write_json(RUN_DIR / "frozen_config_before_validation.json", frozen["frozen_config"])
else:
    print("NO-GO: no frozen deployable model was created.")
"""
    ),
    markdown(
        r"""
## 7. Privacy/contract audit와 완료 marker

원본 ID·secret이 output text/CSV/JSON에 남지 않았는지, 금지 feature와 source overlap이
없는지 마지막으로 검사한다. Full GO run만 frozen model artifact를 갖는다.
"""
    ),
    code(
        r"""
attempt_elapsed_seconds = time.perf_counter() - attempt_started
completion_utc = datetime.now(timezone.utc).isoformat()
environment["end_to_end_attempt_elapsed_seconds"] = attempt_elapsed_seconds
environment["completed_utc"] = completion_utc
environment_history["resume_events"][-1].update({
    "end_to_end_attempt_elapsed_seconds": attempt_elapsed_seconds,
    "completed_utc": completion_utc,
})
plc.write_json(RUN_DIR / "environment.json", environment_history)
cumulative_nested_seconds = sum(
    float(event.get("nested_cv_elapsed_seconds", 0.0))
    for event in environment_history["resume_events"]
)

final_report = plc.build_final_training_report(
    nested_result=nested_result,
    decision=decision,
    environment=environment,
    run_id=run_id,
    frozen=frozen,
)
plc.write_json(RUN_DIR / "FINAL_TRAINING_REPORT.json", final_report)

# Audit every human-readable deliverable created above.  The count-only audit
# file and the minimal completion marker are written only after this passes.
privacy_audit = plc.audit_output_privacy(
    output_dir=RUN_DIR,
    raw_subject_ids=training.subject_ids,
    forbidden_secret=SUBJECT_HASH_KEY,
)
plc.write_json(RUN_DIR / "privacy_audit.json", privacy_audit)
if not privacy_audit["passed"]:
    plc.write_json(
        RUN_DIR / "RUN_INVALID_PRIVACY.json",
        {
            "stage": "after_optional_frozen_artifacts",
            "decision": "INVALID-NO-GO",
            "offending_file_count": privacy_audit["offending_file_count"],
        },
    )
    raise RuntimeError("Final privacy audit failed; completion marker was not created.")
if (RUN_DIR / "RUN_INVALID_PRIVACY.json").exists():
    raise RuntimeError("An invalidated run can never receive a completion marker.")

plc.write_json(
    RUN_DIR / "TRAINING_COMPLETE.json",
    {
        "run_id": run_id,
        "fast_mode": FAST_MODE,
        "decision": decision["stop_go"]["decision"],
        "privacy_audit_passed": True,
        "nested_cv_attempt_elapsed_seconds": elapsed_seconds,
        "nested_cv_recorded_cumulative_seconds": cumulative_nested_seconds,
        "end_to_end_attempt_elapsed_seconds": attempt_elapsed_seconds,
        "completed_utc": completion_utc,
        "artifact_hashes": plc.hash_public_artifacts(RUN_DIR),
    },
)

SUBJECT_HASH_KEY = None
print({
    "run_dir": str(RUN_DIR),
    "decision": decision["stop_go"]["decision"],
    "primary_macro_f1": nested_result["summary"]["macro_f1_mean"],
    "primary_macro_f1_sd": nested_result["summary"]["macro_f1_sd"],
    "privacy_audit_passed": True,
})
"""
    ),
    markdown(
        r"""
## 전달할 결과

`nested_cv_report.json`, inner/outer metrics CSV, coverage/legacy comparator 결과,
`selection_report.json`, `stop_go_decision.json`, `environment.json`을 전달한다.
Benchmark는 이 결과를 검토하고 GO frozen hash가 확인된 뒤 별도 notebook에서만
진행한다.
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
