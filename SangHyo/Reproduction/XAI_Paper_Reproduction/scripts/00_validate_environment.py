from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT_PATH.parents[1] / "src"))

from xai_paper_reproduction import ProjectPaths, find_one, make_daily_binary_dataset


REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "sklearn",
    "scipy",
    "matplotlib",
    "lightgbm",
    "shap",
    "joblib",
]

RAW_FILES = {
    "train_activity.csv": (),
    "train_sleep.csv": (),
    "training_label.csv": ("라벨링데이터", "1.걸음걸이"),
    "val_activity.csv": (),
    "val_sleep.csv": (),
    "val_label.csv": ("라벨링데이터", "1.걸음걸이"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Colab/local environment for XAI paper reproduction.")
    parser.add_argument("--raw-dir", type=str, default=None, help="Path to 128.치매 고위험군 라이프로그")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any dependency or raw file is missing.")
    parser.add_argument("--run-preprocess-smoke", action="store_true", help="Build the daily dataset in memory and check paper counts.")
    return parser.parse_args()


def check_packages() -> list[str]:
    missing = []
    print("[packages]")
    for name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "ok")
            print(f"  OK      {name} {version}")
        except Exception as exc:
            missing.append(name)
            print(f"  MISSING {name}: {exc}")
    return missing


def check_raw_files(raw_dir: Path) -> list[str]:
    missing = []
    print(f"\n[raw data] {raw_dir}")
    if not raw_dir.exists():
        print("  MISSING raw directory")
        return list(RAW_FILES)
    for filename, filters in RAW_FILES.items():
        try:
            path = find_one(raw_dir, filename, filters)
            print(f"  OK      {filename}: {path}")
        except Exception as exc:
            missing.append(filename)
            print(f"  MISSING {filename}: {exc}")
    return missing


def run_preprocess_smoke(raw_dir: Path) -> bool:
    print("\n[preprocess smoke]")
    daily, features, summary = make_daily_binary_dataset(raw_dir)
    print("  summary:", summary)
    ok = (
        summary["rows"] == 12183
        and summary["subjects"] == 174
        and summary["class_counts"] == {"0": 7737, "1": 4446}
        and len(features) > 0
        and not daily.empty
    )
    print("  result:", "OK" if ok else "CHECK")
    return ok


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_script(SCRIPT_PATH, raw_dir=args.raw_dir, output_dir=args.output_dir)
    missing_packages = check_packages()
    missing_raw = check_raw_files(paths.raw_dir)
    smoke_ok = True
    if args.run_preprocess_smoke and not missing_raw:
        smoke_ok = run_preprocess_smoke(paths.raw_dir)

    print("\n[summary]")
    print("  missing_packages:", missing_packages)
    print("  missing_raw_files:", missing_raw)
    print("  preprocess_smoke_ok:", smoke_ok)
    if missing_packages:
        print("\nInstall command:")
        print("  pip install -r training/XAI_Paper_Reproduction/requirements.txt")

    if args.strict and (missing_packages or missing_raw or not smoke_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
