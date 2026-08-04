"""Seed sweep for the paper arm.

The paper reports no random seed (``unresolved_questions.md`` Q15) and evaluates
on 33 subjects of whom **7** are positive.  At that size one subject flipping
moves Recall by 14.3 percentage points and ROC-AUC by 1/182 per rank pair, so a
single seed is a lottery, not a measurement.

Asking "does our number equal the paper's to two decimals?" is therefore the
wrong question.  The answerable one is: **does the paper's reported value fall
inside the range our reconstruction produces across seeds?**  If it does, the
reconstruction is consistent with the paper and the gap is seed noise.  If it
does not, something about the method genuinely differs.

This module runs the same configuration over several seeds and reports that
distribution.  It never selects a seed by its test score -- every seed is
reported, and the summary is the whole distribution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .compare import PAPER_REPORTED

#: Metrics compared against the paper, in the paper's own reporting order.
COMPARED_METRICS = ("accuracy", "roc_auc", "f1", "sensitivity")


def summarise_seed_sweep(
    per_seed: dict[int, dict[str, dict[str, Any]]],
    *,
    tolerance: float = 0.01,
) -> dict[str, Any]:
    """Summarise ``{seed: {model: subject_level_metrics}}`` against the paper.

    ``tolerance`` is the agreement band the task asks about (two decimals).
    """
    seeds = sorted(per_seed)
    models = sorted({m for block in per_seed.values() for m in block})

    rows: list[dict[str, Any]] = []
    for model in models:
        paper = PAPER_REPORTED.get(model, {})
        entry: dict[str, Any] = {"model": model, "n_seeds": 0, "metrics": {}}
        for metric in COMPARED_METRICS:
            values = [
                float(per_seed[s][model][metric])
                for s in seeds
                if model in per_seed[s]
                and per_seed[s][model].get(metric) is not None
                and np.isfinite(float(per_seed[s][model][metric]))
            ]
            if not values:
                continue
            entry["n_seeds"] = max(entry["n_seeds"], len(values))
            array = np.asarray(values, dtype=float)
            target = paper.get(metric)
            record: dict[str, Any] = {
                "paper": target,
                "min": float(array.min()),
                "max": float(array.max()),
                "mean": float(array.mean()),
                "median": float(np.median(array)),
                "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                "values": [round(v, 6) for v in values],
            }
            if target is not None:
                record["paper_within_observed_range"] = bool(
                    array.min() - 1e-9 <= target <= array.max() + 1e-9
                )
                gaps = np.abs(array - target)
                record["closest_gap"] = float(gaps.min())
                record["closest_seed"] = int(seeds[int(np.argmin(gaps))])
                record["any_seed_within_tolerance"] = bool(gaps.min() <= tolerance)
                record["mean_gap"] = float(np.mean(array - target))
            entry["metrics"][metric] = record
        rows.append(entry)

    auc_rows = {
        row["model"]: row["metrics"].get("roc_auc", {}) for row in rows
    }
    consistent = [
        model for model, block in auc_rows.items()
        if block.get("paper_within_observed_range")
    ]
    inconsistent = [
        model for model, block in auc_rows.items()
        if block and not block.get("paper_within_observed_range")
    ]

    return {
        "seeds": seeds,
        "tolerance": tolerance,
        "rows": rows,
        "roc_auc_paper_inside_range": consistent,
        "roc_auc_paper_outside_range": inconsistent,
        "interpretation": (
            "논문값이 seed 분포 안에 들어오면 그 모델은 seed 노이즈로 설명되는 "
            "범위에서 재현된 것이다. 분포 밖이면 방법 자체가 다르다는 신호다. "
            "양성 7명에서는 단일 seed 일치/불일치를 재현 성공/실패로 읽으면 안 된다."
        ),
    }


def render_seed_sweep_markdown(summary: dict[str, Any]) -> str:
    """Render the sweep as ``seed_sweep.md``."""
    lines = [
        "# Seed sweep — 논문값이 재현 분포 안에 있는가",
        "",
        f"- seed: {summary['seeds']}",
        f"- 허용오차: ±{summary['tolerance']}",
        "",
        "평가집단은 33명·양성 7명이다. 한 명이 바뀌면 Recall이 14.3%p 움직이므로,",
        "단일 seed의 일치 여부가 아니라 **분포 포함 여부**로 판단한다.",
        "",
        "| 모델 | 지표 | 논문 | 관측 min~max | 중앙값 | 논문값 포함 | 최소격차 |",
        "| --- | --- | ---: | --- | ---: | :---: | ---: |",
    ]
    for row in summary["rows"]:
        for metric, block in row["metrics"].items():
            paper = block.get("paper")
            inside = block.get("paper_within_observed_range")
            mark = "—" if inside is None else ("✅" if inside else "❌")
            gap = block.get("closest_gap")
            lines.append(
                f"| {row['model']} | {metric} | "
                f"{'—' if paper is None else f'{paper:.4f}'} | "
                f"{block['min']:.4f}~{block['max']:.4f} | {block['median']:.4f} | "
                f"{mark} | {'—' if gap is None else f'{gap:.4f}'} |"
            )
    lines += [
        "",
        f"**ROC-AUC 기준 논문값이 분포 안**: {summary['roc_auc_paper_inside_range'] or '없음'}",
        "",
        f"**ROC-AUC 기준 논문값이 분포 밖**: {summary['roc_auc_paper_outside_range'] or '없음'}",
        "",
        f"> {summary['interpretation']}",
        "",
    ]
    return "\n".join(lines)


def parse_seed_list(raw: str | Sequence[int] | None, *, default_seed: int) -> list[int]:
    """Parse ``--seeds 1,2,3`` or ``--seeds 5`` (meaning 5 seeds from the base)."""
    if raw is None:
        return [int(default_seed)]
    if not isinstance(raw, str):
        return [int(s) for s in raw]
    text = raw.strip()
    if not text:
        return [int(default_seed)]
    if "," in text:
        return [int(part) for part in text.split(",") if part.strip()]
    count = int(text)
    if count < 1:
        raise ValueError("--seeds must be a positive count or a comma-separated list")
    return [int(default_seed) + offset for offset in range(count)]


def write_seed_sweep(output_dir: str | Path, summary: dict[str, Any]) -> Path:
    from ..utils.io import write_json, write_text

    output_dir = Path(output_dir)
    write_json(output_dir / "seed_sweep.json", summary)
    return write_text(output_dir / "seed_sweep.md", render_seed_sweep_markdown(summary))
