"""합성자료 품질·암기 진단 (synthetic_data_risk.md §3).

분류 성능만으로 증강의 효과를 판단하지 않기 위한 모듈이다.
가장 중요한 산출물은 ``memorization_ratio`` — VAE가 학습 표본을 복제하고 있는지를
정량화한다.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

__all__ = ["distribution_report", "memorization_report", "correlation_report", "full_report"]


def distribution_report(real: pd.DataFrame, synth: pd.DataFrame) -> pd.DataFrame:
    """변수별 평균·표준편차·분위수 차이와 1D Wasserstein 거리."""
    from scipy.stats import wasserstein_distance

    rows = []
    for col in real.columns:
        r, s = real[col].to_numpy(float), synth[col].to_numpy(float)
        row = {
            "feature": col,
            "real_mean": float(r.mean()),
            "synth_mean": float(s.mean()),
            "mean_diff": float(s.mean() - r.mean()),
            "real_std": float(r.std(ddof=1)) if len(r) > 1 else 0.0,
            "synth_std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        }
        row["std_ratio"] = (
            row["synth_std"] / row["real_std"] if row["real_std"] > 0 else float("nan")
        )
        for q in (5, 25, 50, 75, 95):
            row[f"real_p{q}"] = float(np.percentile(r, q))
            row[f"synth_p{q}"] = float(np.percentile(s, q))
            row[f"p{q}_diff"] = row[f"synth_p{q}"] - row[f"real_p{q}"]
        try:
            row["wasserstein"] = float(wasserstein_distance(r, s))
        except Exception:  # pragma: no cover - scipy 부재 등
            row["wasserstein"] = float("nan")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["warning_low_variance"] = [
        c for c, v in zip(df["feature"], df["std_ratio"]) if np.isfinite(v) and v < 0.5
    ]
    return df


def correlation_report(real: pd.DataFrame, synth: pd.DataFrame) -> dict:
    """상관행렬 차이."""
    cr, cs = real.corr().to_numpy(), synth.corr().to_numpy()
    diff = np.nan_to_num(cs) - np.nan_to_num(cr)
    return {
        "frobenius_norm": float(np.linalg.norm(diff)),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "n_features": int(real.shape[1]),
    }


def _nn_distance(query: np.ndarray, reference: np.ndarray, *, block: int = 512) -> np.ndarray:
    """query 각 행에서 reference까지의 최근접 유클리드 거리."""
    out = np.empty(len(query), dtype=float)
    for i in range(0, len(query), block):
        chunk = query[i : i + block]
        d = np.linalg.norm(chunk[:, None, :] - reference[None, :, :], axis=2)
        out[i : i + block] = d.min(axis=1)
    return out


def memorization_report(
    synth: pd.DataFrame,
    train_real: pd.DataFrame,
    holdout_real: pd.DataFrame | None = None,
    *,
    near_duplicate_eps: float = 0.01,
) -> dict:
    """암기(memorization) 진단.

    ``memorization_ratio = median(d_train) / median(d_holdout)``.
    0.5 미만이면 합성행이 학습 표본에 유의하게 더 가깝다 = 암기 의심.
    """
    S = synth.to_numpy(float)
    T = train_real.to_numpy(float)
    d_train = _nn_distance(S, T)
    eps = near_duplicate_eps * np.sqrt(S.shape[1])

    out: dict = {
        "n_synthetic": int(len(S)),
        "n_train_real": int(len(T)),
        "nn_distance_to_train_median": float(np.median(d_train)),
        "nn_distance_to_train_p05": float(np.percentile(d_train, 5)),
        "exact_duplicate_rate": float(np.mean(d_train == 0.0)),
        "near_duplicate_rate": float(np.mean(d_train < eps)),
        "near_duplicate_eps": float(eps),
    }
    if len(T) > 1:
        # 실제 행끼리의 최근접 거리 기준선. 자기 자신(거리 0)은 대각선을 inf로 두어 제외한다.
        dd = np.linalg.norm(T[:, None, :] - T[None, :, :], axis=2)
        np.fill_diagonal(dd, np.inf)
        out["real_real_nn_baseline_median"] = float(np.median(dd.min(axis=1)))

    if holdout_real is not None and len(holdout_real):
        d_hold = _nn_distance(S, holdout_real.to_numpy(float))
        out["nn_distance_to_holdout_median"] = float(np.median(d_hold))
        denom = out["nn_distance_to_holdout_median"]
        out["memorization_ratio"] = (
            float(out["nn_distance_to_train_median"] / denom) if denom > 0 else float("inf")
        )
        if out["memorization_ratio"] < 0.5:
            out["warning"] = "memorization_suspected"
            log.warning(
                "합성자료 암기 의심: memorization_ratio=%.3f (< 0.5). "
                "VAE가 학습 표본을 복제하고 있을 수 있다 (synthetic_data_risk.md §3.2).",
                out["memorization_ratio"],
            )
    if out["exact_duplicate_rate"] > 0:
        log.warning("합성행 중 %.2f%%가 실제 학습행과 완전히 일치한다", 100 * out["exact_duplicate_rate"])
    return out


def full_report(
    synth: pd.DataFrame,
    train_real: pd.DataFrame,
    holdout_real: pd.DataFrame | None = None,
) -> dict:
    """분포·상관·암기 진단을 한 번에."""
    return {
        "distribution": distribution_report(train_real, synth).to_dict(orient="records"),
        "correlation": correlation_report(train_real, synth),
        "memorization": memorization_report(synth, train_real, holdout_real),
        "caveat": (
            "합성행은 소수의 실제 Dem 피험자에서 생성되었다. 행 수 증가를 표본 크기 "
            "증가로 해석하면 안 된다 (synthetic_data_risk.md §1)."
        ),
    }
