#!/usr/bin/env python
"""Entry point: python run.py --config configs/<experiment>.yaml [--smoke] [--data-root PATH]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR / "src"))

from cheon.pipeline import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="1 seed, 1 outer fold, 1 inner fold, max_k=6; writes to *_smoke")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--seeds", default=None, help="override seeds, comma-separated")
    ap.add_argument("--output-dir", default=None, help="override output_dir (relative to the experiment directory)")
    ap.add_argument("--sleep-time-definition", default=None, choices=["clock_difference", "timestamp_difference"])
    a = ap.parse_args()
    cfg_path = (EXP_DIR / a.config) if not Path(a.config).is_absolute() else Path(a.config)
    cfg_text = cfg_path.read_text()
    cfg = yaml.safe_load(cfg_text)
    if a.seeds:
        cfg["seeds"] = [int(s) for s in a.seeds.split(",")]
    if a.output_dir:
        cfg["output_dir"] = a.output_dir
    if a.sleep_time_definition:
        cfg["sleep_time_definition"] = a.sleep_time_definition
    repo_root = EXP_DIR.parents[2]
    data_root = Path(a.data_root) if a.data_root else repo_root / "Data"
    out = run(cfg, cfg_text, EXP_DIR, repo_root, data_root, smoke=a.smoke)
    print(f"DONE -> {out}")


if __name__ == "__main__":
    main()
