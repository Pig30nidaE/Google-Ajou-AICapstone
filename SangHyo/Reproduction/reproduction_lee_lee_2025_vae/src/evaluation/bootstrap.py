"""피험자 단위 bootstrap 신뢰구간.

Dem 피험자가 12명뿐이므로 CI가 매우 넓게 나오는 것이 **정상이며 보고 대상**이다.
"""

from __future__ import annotations

import numpy as np

from .metrics import compute_metrics

__all__ = ["bootstrap_ci", "DEFAULT_CI_METRICS"]

DEFAULT_CI_METRICS = (
    "macro_f1",
    "balanced_accuracy",
    "macro_roc_auc_ovr",
    "dem_recall",
    "dem_f1",
)


def bootstrap_ci(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    metrics: tuple[str, ...] = DEFAULT_CI_METRICS,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    unit: str = "subject",
) -> dict[str, dict[str, float]]:
    """피험자를 재표집해 percentile 95% CI를 구한다.

    Returns:
        ``{metric: {"point", "lo", "hi", "n_valid"}}``.
    """
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    n = len(y_true)
    point = compute_metrics(y_true, proba, unit=unit)

    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(set(y_true[idx].tolist())) < 2:
            continue
        m = compute_metrics(y_true[idx], proba[idx], unit=unit)
        for name in metrics:
            v = m.get(name, float("nan"))
            if np.isfinite(v):
                draws[name].append(float(v))

    out: dict[str, dict[str, float]] = {}
    for name in metrics:
        vals = np.asarray(draws[name], dtype=float)
        if len(vals) < 20:
            out[name] = {
                "point": float(point.get(name, float("nan"))),
                "lo": float("nan"),
                "hi": float("nan"),
                "n_valid": int(len(vals)),
            }
            continue
        out[name] = {
            "point": float(point.get(name, float("nan"))),
            "lo": float(np.percentile(vals, 100 * alpha / 2)),
            "hi": float(np.percentile(vals, 100 * (1 - alpha / 2))),
            "n_valid": int(len(vals)),
        }
    return out
