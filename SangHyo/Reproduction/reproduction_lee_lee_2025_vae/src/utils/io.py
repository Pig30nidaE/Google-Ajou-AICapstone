"""출력 저장 유틸."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

__all__ = ["ensure_dir", "save_json", "save_table", "save_provenance", "RunPaths"]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class _NpEncoder(json.JSONEncoder):
    def default(self, o: Any):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, cls=_NpEncoder)
    log.info("saved %s", path)
    return path


def save_table(df: pd.DataFrame, path: str | Path, *, also_markdown: bool = True) -> Path:
    """CSV로 저장하고, 필요하면 같은 이름의 .md도 남긴다."""
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    if also_markdown:
        md = path.with_suffix(".md")
        try:
            body = df.to_markdown(index=False)
        except ImportError:  # tabulate 미설치
            body = df.to_string(index=False)
        caveat = df.attrs.get("caveat")
        note = df.attrs.get("paper_note")
        with md.open("w", encoding="utf-8") as fh:
            fh.write(body + "\n")
            if note:
                fh.write(f"\n> {note}\n")
            if caveat:
                fh.write(f"\n> ⚠️ {caveat}\n")
    log.info("saved %s", path)
    return path


def save_provenance(prov: pd.DataFrame | None, path: str | Path) -> Path | None:
    """합성행 provenance 저장. parquet을 시도하고 실패하면 CSV."""
    if prov is None or len(prov) == 0:
        return None
    path = Path(path)
    ensure_dir(path.parent)
    try:
        prov.to_parquet(path.with_suffix(".parquet"), index=False)
        return path.with_suffix(".parquet")
    except Exception:
        prov.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
        return path.with_suffix(".csv")


class RunPaths:
    """한 실행의 출력 경로 묶음."""

    def __init__(self, root: str | Path, label: str) -> None:
        self.root = ensure_dir(Path(root) / label)
        self.label = label

    def __call__(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        ensure_dir(p.parent)
        return p

    def __repr__(self) -> str:  # pragma: no cover
        return f"RunPaths({self.root})"
