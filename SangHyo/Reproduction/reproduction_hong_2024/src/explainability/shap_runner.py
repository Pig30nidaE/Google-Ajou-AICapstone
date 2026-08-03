"""Deep SHAP for the LSTM, computed on out-of-fold evaluation data only.

Hong et al. §3.3 and §4.3 use Deep SHAP on the five-day LSTM and plot it "across
the entire dataset" -- i.e. one model, trained once, explained on data it was
trained on.  That answers "what did this fitted model use", which is a legitimate
question, but it is not evidence that the ranking generalises.

This module therefore does both, and labels them differently:

* ``mode='paper_style'`` reproduces the paper's single-model, all-data attribution.
* ``mode='out_of_fold'`` (the default) computes SHAP inside each outer fold, on
  that fold's held-out data only, and reports rank stability across folds.

**Nothing here is executed in the current scope of work.**  The engine only calls
it when ``explainability.enabled`` is true, which every shipped config sets to
false.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..sequences.builder import SequenceSet
from ..utils.io import hash_subject

TIME_AGGREGATIONS = ("mean_abs", "sum", "last_step", "max_abs")
SHAP_MODES = ("out_of_fold", "paper_style")


@dataclass
class ShapConfig:
    enabled: bool = False
    mode: str = "out_of_fold"
    algorithm: str = "deep"          # DeepExplainer, matching the paper's Deep SHAP
    n_background: int = 100
    background_source: str = "train_fold_random"
    n_explain: int = 500
    time_aggregation: str = "mean_abs"
    subject_aggregation: str = "mean_abs"
    seed: int = 42

    def describe(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ShapResult:
    values: np.ndarray                       # (n_sequences, sequence_length, n_features)
    feature_columns: tuple[str, ...]
    fold: int
    mode: str
    meta: dict[str, Any] = field(default_factory=dict)

    def feature_importance(self, *, time_aggregation: str = "mean_abs") -> pd.DataFrame:
        """Collapse the time axis, then average |SHAP| over sequences."""
        if time_aggregation not in TIME_AGGREGATIONS:
            raise ValueError(f"time_aggregation must be one of {TIME_AGGREGATIONS}")
        values = self.values
        if time_aggregation == "mean_abs":
            per_sequence = np.abs(values).mean(axis=1)
        elif time_aggregation == "sum":
            per_sequence = values.sum(axis=1)
        elif time_aggregation == "last_step":
            per_sequence = values[:, -1, :]
        else:
            per_sequence = np.abs(values).max(axis=1)

        importance = np.abs(per_sequence).mean(axis=0)
        frame = pd.DataFrame(
            {"feature": list(self.feature_columns), "mean_abs_shap": importance}
        ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        frame["rank"] = np.arange(1, len(frame) + 1)
        frame["fold"] = self.fold
        return frame

    def subject_importance(
        self, subjects: Sequence[str], *, time_aggregation: str = "mean_abs"
    ) -> pd.DataFrame:
        """Per-subject importance, so a 118-window subject cannot dominate."""
        per_sequence = np.abs(self.values).mean(axis=1)
        frame = pd.DataFrame(per_sequence, columns=list(self.feature_columns))
        frame["subject_hash"] = [hash_subject(subject) for subject in subjects]
        return frame.groupby("subject_hash").mean().reset_index()


def compute_shap(
    model: Any,
    background: SequenceSet,
    explain: SequenceSet,
    config: ShapConfig,
    *,
    fold: int = 0,
    device: str = "cpu",
) -> ShapResult:
    """Deep SHAP values for *explain*, with *background* drawn from training data.

    ``background`` must come from the training side.  A background sampled from
    the data being explained would define the baseline in terms of the evaluation
    set, which is the SHAP equivalent of fitting a scaler on test.
    """
    import shap
    import torch

    if config.mode not in SHAP_MODES:
        raise ValueError(f"mode must be one of {SHAP_MODES}")
    if not len(background) or not len(explain):
        raise ValueError("both background and explain sets must be non-empty")

    rng = np.random.default_rng(config.seed)
    bg_idx = rng.choice(
        len(background), size=min(config.n_background, len(background)), replace=False
    )
    ex_idx = rng.choice(
        len(explain), size=min(config.n_explain, len(explain)), replace=False
    )
    bg_tensor = torch.as_tensor(background.X[bg_idx].astype(np.float32)).to(device)
    ex_tensor = torch.as_tensor(explain.X[ex_idx].astype(np.float32)).to(device)

    explainer = shap.DeepExplainer(model.model, bg_tensor)
    values = explainer.shap_values(ex_tensor, check_additivity=False)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 4:                     # (n, L, F, 1) for a single output
        values = values[..., 0]

    return ShapResult(
        values=values,
        feature_columns=explain.feature_columns,
        fold=fold,
        mode=config.mode,
        meta={
            "algorithm": config.algorithm,
            "n_background": int(len(bg_idx)),
            "background_source": config.background_source,
            "background_split": background.split_name,
            "n_explained": int(len(ex_idx)),
            "explained_split": explain.split_name,
            "sequence_length": explain.sequence_length,
            "explained_subject_hashes": sorted(
                {hash_subject(subject) for subject in explain.subjects[ex_idx].tolist()}
            ),
            "caveat": (
                "paper_style는 학습에 사용한 자료 위에서 계산한 단일 모델 설명이다. "
                "일반화 가능한 변수 중요도로 해석하지 않는다."
                if config.mode == "paper_style"
                else "outer fold의 held-out 자료에서만 계산했다."
            ),
        },
    )


def rank_stability(results: Sequence[ShapResult], *, top_k: int = 10) -> dict[str, Any]:
    """How consistently a feature is ranked highly across outer folds."""
    if not results:
        return {"n_folds": 0}

    frames = [r.feature_importance() for r in results]
    combined = pd.concat(frames, ignore_index=True)

    selection = (
        combined[combined["rank"] <= top_k]
        .groupby("feature")
        .size()
        .rename("n_folds_in_top_k")
        .reset_index()
    )
    ranks = (
        combined.groupby("feature")["rank"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"mean": "mean_rank", "std": "std_rank"})
    )
    merged = ranks.merge(selection, on="feature", how="left").fillna({"n_folds_in_top_k": 0})
    merged = merged.sort_values(["mean_rank"]).reset_index(drop=True)

    return {
        "n_folds": len(results),
        "top_k": top_k,
        "per_feature": merged.to_dict(orient="records"),
        "stable_top_features": merged[
            merged["n_folds_in_top_k"] == len(results)
        ]["feature"].tolist(),
        "paper_reported_top_features": [
            "sleep_breath_average", "rmssd_average", "sleep_rem",
            "sleep_deep", "sleep_restless", "sleep_light",
        ],
    }


def compare_with_paper(stability: dict[str, Any], *, top_k: int = 6) -> dict[str, Any]:
    """Did the paper's six named drivers survive an out-of-fold recomputation?"""
    reported = stability.get("paper_reported_top_features", [])
    per_feature = {row["feature"]: row for row in stability.get("per_feature", [])}
    ours = [row["feature"] for row in stability.get("per_feature", [])][:top_k]
    return {
        "paper_top_features": reported,
        "reproduction_top_features": ours,
        "overlap": sorted(set(reported) & set(ours)),
        "paper_features_ranks_here": {
            name: per_feature.get(name, {}).get("mean_rank") for name in reported
        },
        "note": (
            "논문은 단일 5일 모델의 전체 자료 SHAP을 보고했다. 여기서의 fold별 순위와 "
            "직접적인 동일 비교가 아니다."
        ),
    }


def summarise_by_length(results_by_length: dict[int, Sequence[ShapResult]]) -> dict[str, Any]:
    """The spec's "3/4/5일 모델별 중요변수 차이" table."""
    return {
        str(length): rank_stability(results)
        for length, results in results_by_length.items()
    }
