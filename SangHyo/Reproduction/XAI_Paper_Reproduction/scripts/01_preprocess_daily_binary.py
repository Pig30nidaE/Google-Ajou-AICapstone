from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT_PATH.parents[1] / "src"))

from xai_paper_reproduction import ProjectPaths, ensure_dir, make_daily_binary_dataset, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-style daily binary lifelog dataset from raw AI-Hub files.")
    parser.add_argument("--raw-dir", type=str, default=None, help="Path to 128.치매 고위험군 라이프로그")
    parser.add_argument("--output-dir", type=str, default=None, help="Output root. Defaults to this reproduction folder outputs.")
    parser.add_argument(
        "--merge-policy",
        choices=["left_activity", "inner"],
        default="left_activity",
        help="left_activity matches the paper's 12,183 daily row count; inner requires both activity and sleep.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_script(SCRIPT_PATH, raw_dir=args.raw_dir, output_dir=args.output_dir)
    data_dir = ensure_dir(paths.output_dir / "data")

    daily, feature_cols, summary = make_daily_binary_dataset(paths.raw_dir, merge_policy=args.merge_policy)

    daily_path = data_dir / "daily_binary_lifelog.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    save_json(feature_cols, data_dir / "feature_columns.json")
    save_json(summary, data_dir / "preprocess_summary.json")

    print("[saved]", daily_path)
    print("[saved]", data_dir / "feature_columns.json")
    print("[saved]", data_dir / "preprocess_summary.json")
    print("[summary]", summary)


if __name__ == "__main__":
    main()
