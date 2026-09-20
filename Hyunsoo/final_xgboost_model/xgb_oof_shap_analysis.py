"""Leakage-resistant SHAP analysis for the frozen final XGBoost model.

For each repeated outer CV split, the model and preprocessing are fitted only
on the training subjects. Exact TreeSHAP contributions are then calculated for
the held-out subjects. SHAP values are on the XGBoost raw-margin (log-odds)
scale; positive values push the prediction toward MCI/Dementia (label 1).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from xgboost import DMatrix, XGBClassifier

from run_experiment import ALL, load_data


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SEEDS = (42, 13, 73, 101, 2026)
BOOTSTRAP_SEED = 20260919
BOOTSTRAP_REPETITIONS = 10_000


def make_model() -> XGBClassifier:
    return XGBClassifier(
        max_depth=3,
        learning_rate=0.04,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        scale_pos_weight=1.0,
        random_state=42,
        n_jobs=1,
    )


def preprocess(
    training: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    transformed_training = scaler.fit_transform(imputer.fit_transform(training))
    transformed_test = scaler.transform(imputer.transform(test))
    return transformed_training, transformed_test


def calculate_oof_shap(
    features: pd.DataFrame, labels: np.ndarray, seeds: tuple[int, ...]
) -> pd.DataFrame:
    rows: list[dict] = []
    for seed in seeds:
        outer_cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        for fold, (training_indices, test_indices) in enumerate(
            outer_cv.split(features, labels), start=1
        ):
            x_train, x_test = preprocess(
                features.iloc[training_indices], features.iloc[test_indices]
            )
            model = make_model()
            model.fit(x_train, labels[training_indices])
            probabilities = model.predict_proba(x_test)[:, 1]

            matrix = DMatrix(x_test, feature_names=ALL)
            contributions = model.get_booster().predict(matrix, pred_contribs=True)
            raw_margin = model.get_booster().predict(matrix, output_margin=True)
            if contributions.shape != (len(test_indices), len(ALL) + 1):
                raise AssertionError("Unexpected SHAP contribution shape")
            if not np.allclose(contributions.sum(axis=1), raw_margin, atol=1e-5):
                raise AssertionError("SHAP contributions do not sum to the raw margin")
            if not np.allclose(expit(raw_margin), probabilities, atol=1e-6):
                raise AssertionError("Raw margins do not reproduce predicted probabilities")

            for position, subject in enumerate(test_indices):
                row = {
                    "seed": seed,
                    "fold": fold,
                    "subject_row": int(subject),
                    "y": int(labels[subject]),
                    "score": float(probabilities[position]),
                    "base_value": float(contributions[position, -1]),
                }
                row.update(
                    {
                        f"value__{feature}": float(features.iloc[subject][feature])
                        for feature in ALL
                    }
                )
                row.update(
                    {
                        f"shap__{feature}": float(contributions[position, column])
                        for column, feature in enumerate(ALL)
                    }
                )
                rows.append(row)
            print(f"seed={seed} fold={fold}/5 complete", flush=True)
    return pd.DataFrame(rows)


def aggregate_subjects(oof_values: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for subject, group in oof_values.groupby("subject_row", sort=True):
        if group.seed.nunique() != len(DEFAULT_SEEDS):
            raise ValueError("Each subject must have one held-out SHAP value per seed")
        row = {
            "subject_row": int(subject),
            "y": int(group.y.iloc[0]),
            "score_mean": float(group.score.mean()),
        }
        for feature in ALL:
            row[f"value__{feature}"] = float(group[f"value__{feature}"].iloc[0])
            row[f"shap_mean__{feature}"] = float(group[f"shap__{feature}"].mean())
            row[f"shap_abs_mean__{feature}"] = float(
                group[f"shap__{feature}"].abs().mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_importance(
    subject_values: pd.DataFrame,
    oof_values: pd.DataFrame,
    repetitions: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sample_indices = rng.integers(
        0, len(subject_values), size=(repetitions, len(subject_values))
    )
    seed_importance = {}
    for seed, group in oof_values.groupby("seed"):
        seed_importance[int(seed)] = {
            feature: float(group[f"shap__{feature}"].abs().mean()) for feature in ALL
        }

    rows = []
    for feature in ALL:
        subject_importance = subject_values[f"shap_abs_mean__{feature}"].to_numpy()
        draws = subject_importance[sample_indices].mean(axis=1)
        estimate = float(subject_importance.mean())
        low, high = np.quantile(draws, [0.025, 0.975])
        per_seed = np.array(
            [seed_importance[seed][feature] for seed in sorted(seed_importance)]
        )
        rows.append(
            {
                "feature": feature,
                "mean_abs_shap": estimate,
                "ci_95_low": float(low),
                "ci_95_high": float(high),
                "seed_min": float(per_seed.min()),
                "seed_max": float(per_seed.max()),
            }
        )
    result = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    result["importance_share"] = result.mean_abs_shap / result.mean_abs_shap.sum()
    result["rank"] = np.arange(1, len(result) + 1)

    rank_rows = []
    for seed, importance in seed_importance.items():
        ordered = pd.Series(importance).rank(ascending=False, method="average")
        rank_rows.extend(
            {"seed": seed, "feature": feature, "seed_rank": float(rank_value)}
            for feature, rank_value in ordered.items()
        )
    ranks = pd.DataFrame(rank_rows).groupby("feature").seed_rank.agg(
        mean_seed_rank="mean",
        best_seed_rank="min",
        worst_seed_rank="max",
    )
    top_five = (
        pd.DataFrame(rank_rows)
        .assign(in_top_five=lambda frame: frame.seed_rank <= 5)
        .groupby("feature")
        .in_top_five.sum()
        .rename("top_five_seed_count")
    )
    return result.merge(ranks, on="feature").merge(top_five, on="feature")


def direction_summary(subject_values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in ALL:
        values = subject_values[f"value__{feature}"].to_numpy()
        shap_values = subject_values[f"shap_mean__{feature}"].to_numpy()
        correlation, p_value = spearmanr(values, shap_values)
        negative = subject_values.y == 0
        positive = subject_values.y == 1
        rows.append(
            {
                "feature": feature,
                "spearman_value_vs_shap": float(correlation),
                "spearman_p_value_descriptive": float(p_value),
                "value_median_cn": float(np.median(values[negative])),
                "value_median_mci_dementia": float(np.median(values[positive])),
                "shap_mean_cn": float(np.mean(shap_values[negative])),
                "shap_mean_mci_dementia": float(np.mean(shap_values[positive])),
            }
        )
    return pd.DataFrame(rows)


def binned_direction(subject_values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in ALL:
        values = subject_values[f"value__{feature}"]
        bins = pd.qcut(values, q=5, duplicates="drop")
        frame = pd.DataFrame(
            {
                "value": values,
                "shap": subject_values[f"shap_mean__{feature}"],
                "bin": bins,
            }
        )
        for order, (_, group) in enumerate(frame.groupby("bin", observed=True), start=1):
            rows.append(
                {
                    "feature": feature,
                    "bin_order_low_to_high": order,
                    "n_subjects": int(len(group)),
                    "feature_min": float(group.value.min()),
                    "feature_median": float(group.value.median()),
                    "feature_max": float(group.value.max()),
                    "mean_shap": float(group.shap.mean()),
                    "median_shap": float(group.shap.median()),
                }
            )
    return pd.DataFrame(rows)


def correlation_pairs(subject_values: pd.DataFrame) -> pd.DataFrame:
    raw = subject_values[[f"value__{feature}" for feature in ALL]].copy()
    raw.columns = ALL
    correlation = raw.corr(method="spearman")
    rows = []
    for left_index, left in enumerate(ALL):
        for right in ALL[left_index + 1 :]:
            rows.append(
                {
                    "feature_a": left,
                    "feature_b": right,
                    "spearman_correlation": float(correlation.loc[left, right]),
                    "absolute_correlation": float(abs(correlation.loc[left, right])),
                }
            )
    return pd.DataFrame(rows).sort_values("absolute_correlation", ascending=False)


def plot_importance(importance: pd.DataFrame, output: Path) -> None:
    ordered = importance.sort_values("mean_abs_shap", ascending=True)
    figure, axis = plt.subplots(figsize=(9.5, 6.8))
    axis.barh(ordered.feature, ordered.mean_abs_shap, color="#356B8C")
    axis.errorbar(
        ordered.mean_abs_shap,
        ordered.feature,
        xerr=np.vstack(
            [
                ordered.mean_abs_shap - ordered.ci_95_low,
                ordered.ci_95_high - ordered.mean_abs_shap,
            ]
        ),
        fmt="none",
        ecolor="#263238",
        elinewidth=1,
        capsize=2,
    )
    axis.set_xlabel("Mean absolute SHAP value (log-odds scale)")
    axis.set_title("Held-out SHAP importance of the frozen XGBoost model")
    axis.grid(axis="x", color="#D9DEE2", linewidth=0.7)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_beeswarm(
    subject_values: pd.DataFrame, importance: pd.DataFrame, output: Path
) -> None:
    top_features = importance.head(10).feature.tolist()
    rng = np.random.default_rng(1847)
    figure, axis = plt.subplots(figsize=(10.5, 7.0))
    for row, feature in enumerate(reversed(top_features)):
        shap_values = subject_values[f"shap_mean__{feature}"].to_numpy()
        feature_values = subject_values[f"value__{feature}"].to_numpy()
        percentiles = rankdata(feature_values, method="average") / len(feature_values)
        jitter = rng.uniform(-0.28, 0.28, size=len(subject_values))
        scatter = axis.scatter(
            shap_values,
            row + jitter,
            c=percentiles,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            s=17,
            alpha=0.72,
            linewidths=0,
        )
    axis.axvline(0, color="#263238", linewidth=1)
    axis.set_yticks(range(len(top_features)), labels=list(reversed(top_features)))
    axis.set_xlabel("SHAP value: left pushes CN, right pushes MCI/Dementia")
    axis.set_title("Direction of the ten most influential features")
    axis.grid(axis="x", color="#E1E5E8", linewidth=0.7)
    axis.set_axisbelow(True)
    colorbar = figure.colorbar(scatter, ax=axis, pad=0.02)
    colorbar.set_label("Feature value percentile (low to high)")
    colorbar.set_ticks([0, 1], labels=["Low", "High"])
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPETITIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output directory to preserve previous analyses")
    args.output.mkdir(parents=True)

    features, labels, data_audit = load_data(args.data)
    seeds = tuple(int(seed) for seed in args.seeds)
    if seeds != DEFAULT_SEEDS:
        raise ValueError(f"This frozen analysis expects seeds {DEFAULT_SEEDS}")

    oof_values = calculate_oof_shap(features, labels, seeds)
    subject_values = aggregate_subjects(oof_values)
    importance = bootstrap_importance(subject_values, oof_values, args.bootstrap)
    direction = direction_summary(subject_values)
    binned = binned_direction(subject_values)
    correlations = correlation_pairs(subject_values)

    oof_values.to_csv(args.output / "oof_shap_values.csv", index=False)
    subject_values.to_csv(args.output / "subject_shap_values.csv", index=False)
    importance.to_csv(args.output / "shap_importance.csv", index=False)
    direction.to_csv(args.output / "shap_direction_summary.csv", index=False)
    binned.to_csv(args.output / "shap_binned_direction.csv", index=False)
    correlations.to_csv(args.output / "feature_correlations.csv", index=False)
    plot_importance(importance, args.output / "shap_importance.png")
    plot_beeswarm(subject_values, importance, args.output / "shap_direction.png")

    details = {
        "method": "exact XGBoost TreeSHAP on held-out outer-fold subjects",
        "shap_scale": "raw margin (log-odds); positive pushes label 1 (MCI/Dementia)",
        "subject_count": int(len(subject_values)),
        "oof_rows": int(len(oof_values)),
        "seeds": list(seeds),
        "folds_per_seed": 5,
        "bootstrap_repetitions": args.bootstrap,
        "bootstrap_unit": "subject",
        "data_audit": data_audit,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {
            package: importlib.metadata.version(package)
            for package in [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "xgboost",
                "matplotlib",
            ]
        },
    }
    (args.output / "details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "COMPLETE.txt").write_text(
        "Completed held-out SHAP analysis.\n", encoding="utf-8"
    )
    print("\nSHAP importance")
    print(
        importance[
            [
                "rank",
                "feature",
                "mean_abs_shap",
                "ci_95_low",
                "ci_95_high",
                "importance_share",
                "mean_seed_rank",
                "top_five_seed_count",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
