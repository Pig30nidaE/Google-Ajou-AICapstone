"""
LightGBM + RandomForest classification for the RF/LGBM discrete datasets.

이 파일이 기존에 안 돌던 주된 이유:
- Windows 절대경로(C:/...)가 하드코딩되어 있었음
- 현재 산출 CSV는 target_class가 아니라 binary_class를 사용함
- 로컬에 xai/shap 패키지가 없음
- 로컬에 imbalanced-learn(imblearn)이 없어 SMOTE import에서 바로 종료됨

아래 코드는 현재 프로젝트 구조의 CSV를 기본값으로 읽고, 설치되지 않은 선택
의존성은 자동으로 건너뛸 수 있게 만든 실행용 버전입니다.
"""

from __future__ import annotations

import argparse
# ✏️ CHANGE B-5: serialize best parameter summaries for saved results
import json
import os
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib-cache"))

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, ParameterSampler, StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(PROJECT_DIR))
try:
    from xai import ShapAnalyzer
except ImportError:
    ShapAnalyzer = None

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None

try:
    # ✏️ CHANGE C-1: use tqdm progress bars in Colab/Jupyter/terminal when available
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


DEFAULT_ACTIVITY_FILE = PROJECT_DIR / "rf_lgbm_activity_discrete.csv"
DEFAULT_SLEEP_FILE = PROJECT_DIR / "rf_lgbm_sleep_discrete.csv"
META_COLS = {"patient_id", "sample_date", "split", "binary_class", "target_class"}
GRID_SEED_OFFSETS = {
    "lgbm_no_smote": 11,
    "rf_no_smote": 23,
    "lgbm_smote": 37,
    "rf_smote": 41,
}


class SimpleProgressBar:
    # ✏️ CHANGE C-1: fallback visible progress when tqdm is not installed
    def __init__(self, total: int, desc: str) -> None:
        self.total = max(1, int(total))
        self.desc = desc
        self.current = 0
        self.report_every = max(1, self.total // 20)
        self.postfix: dict[str, object] = {}
        print(f"{self.desc}: 0/{self.total} (0.0%)")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def update(self, n: int = 1) -> None:
        self.current = min(self.total, self.current + n)
        if self.current == self.total or self.current % self.report_every == 0:
            pct = self.current / self.total * 100
            postfix = ", ".join(f"{k}={v}" for k, v in self.postfix.items())
            suffix = f" | {postfix}" if postfix else ""
            print(f"{self.desc}: {self.current}/{self.total} ({pct:.1f}%){suffix}")

    def set_postfix(self, values: dict[str, object]) -> None:
        self.postfix = values

    def close(self) -> None:
        if self.current < self.total:
            print(f"{self.desc}: stopped at {self.current}/{self.total}")


class NullProgressBar:
    # ✏️ CHANGE C-1: no-op progress object for --no-progress
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def update(self, n: int = 1) -> None:
        return None

    def set_postfix(self, values: dict[str, object]) -> None:
        return None

    def close(self) -> None:
        return None


def make_progress_bar(total: int, desc: str, enabled: bool):
    # ✏️ CHANGE C-1: central progress factory for Colab/Jupyter/terminal output
    if not enabled:
        return NullProgressBar()
    if tqdm is not None:
        return tqdm(total=total, desc=desc, dynamic_ncols=True, leave=True)
    return SimpleProgressBar(total, desc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LightGBM and RandomForest on the discrete dementia datasets."
    )
    parser.add_argument("--activity-file", type=Path, default=DEFAULT_ACTIVITY_FILE)
    parser.add_argument("--sleep-file", type=Path, default=DEFAULT_SLEEP_FILE)
    parser.add_argument("--target-col", choices=["auto", "binary_class", "target_class"], default="auto")
    parser.add_argument("--mode", choices=["paper", "clinical"], default="paper")
    parser.add_argument("--source-split", choices=["train", "val", "all"], default="train")
    parser.add_argument("--random-search-iter", type=int, default=20)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--search-n-jobs", type=int, default=1)
    parser.add_argument("--model-n-jobs", type=int, default=1)
    # ✏️ CHANGE B-4: increase feature capacity for rolling features
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--max-grid-combinations",
        type=int,
        default=0,
        help="Limit each model's parameter grid with deterministic random sampling. 0 means full grid.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm/simple progress bars.",
    )
    parser.add_argument(
        "--colab",
        action="store_true",
        help="Enable Colab-friendly runtime logging and Drive output defaults.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for selected_features.csv, experiment_results.csv, scaler, and model pkl files.",
    )
    parser.add_argument(
        "--skip-smote",
        action="store_true",
        help="Skip SMOTE experiments even if imbalanced-learn is installed.",
    )
    return parser.parse_args()


def print_colab_runtime_info(colab: bool, output_dir: Path | None) -> None:
    # ✏️ CHANGE C-2: print Colab/Drive runtime context for Colab Pro runs
    if not colab:
        return
    print("\n[Colab Runtime]")
    print(f"COLAB_GPU: {os.environ.get('COLAB_GPU', 'not detected')}")
    print(f"COLAB_TPU_ADDR: {os.environ.get('COLAB_TPU_ADDR', 'not detected')}")
    drive_root = Path("/content/drive/MyDrive")
    print(f"Google Drive mounted: {drive_root.exists()} ({drive_root})")
    if output_dir is not None:
        print(f"Output directory: {output_dir}")
    print("Note: RandomForest is CPU-based; LightGBM uses CPU unless your Colab LightGBM build supports GPU.")


def resolve_output_dir(output_dir: Path | None, colab: bool) -> Path | None:
    # ✏️ CHANGE C-2: default Colab outputs to Google Drive when mounted
    if output_dir is not None:
        return output_dir
    if not colab:
        return None

    drive_root = Path("/content/drive/MyDrive")
    if drive_root.exists():
        return drive_root / "ML" / "lgbm_rf_model_outputs"
    return Path("/content/lgbm_rf_model_outputs")


def limit_parameter_grid(
    params_list: list[tuple],
    max_combinations: int,
    random_state: int,
    grid_name: str,
) -> list[tuple]:
    # ✏️ CHANGE C-3: deterministic grid budget to make Colab runs practical
    if max_combinations <= 0 or len(params_list) <= max_combinations:
        return params_list

    rng = np.random.default_rng(random_state + GRID_SEED_OFFSETS.get(grid_name, 0))
    selected = {0}
    if max_combinations > 1:
        candidates = np.arange(1, len(params_list))
        sampled = rng.choice(candidates, size=max_combinations - 1, replace=False)
        selected.update(int(i) for i in sampled)

    return [params_list[i] for i in sorted(selected)]


def limit_parameter_grids(
    grids: dict[str, list[tuple]],
    max_combinations: int,
    random_state: int,
) -> dict[str, list[tuple]]:
    # ✏️ CHANGE C-3: apply per-model grid budget and print visible run size
    limited = {}
    print("\n[Hyperparameter Grid]")
    for name, params_list in grids.items():
        limited[name] = limit_parameter_grid(params_list, max_combinations, random_state, name)
        if len(limited[name]) < len(params_list):
            print(f"  {name}: {len(params_list):,} → {len(limited[name]):,} combinations")
        else:
            print(f"  {name}: {len(params_list):,} combinations")
    if max_combinations <= 0:
        print("  mode: full grid search")
    else:
        print(f"  mode: deterministic sampled grid, max {max_combinations:,} combinations per model")
    return limited


def read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {path}")
    return pd.read_csv(path)


def choose_target_col(df_activity: pd.DataFrame, df_sleep: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        if requested not in df_activity.columns or requested not in df_sleep.columns:
            raise KeyError(f"요청한 target 컬럼이 양쪽 CSV에 없습니다: {requested}")
        return requested

    for col in ("target_class", "binary_class"):
        if col in df_activity.columns and col in df_sleep.columns:
            return col
    raise KeyError("target_class 또는 binary_class 컬럼을 찾지 못했습니다.")


def load_merged_data(activity_file: Path, sleep_file: Path, target_col: str) -> tuple[pd.DataFrame, str]:
    print("\n[데이터 로드] 활동 데이터와 수면 데이터를 병합합니다...")
    df_activity = read_dataset(activity_file)
    df_sleep = read_dataset(sleep_file)
    target_col = choose_target_col(df_activity, df_sleep, target_col)

    merge_keys = ["patient_id", "sample_date", "split", target_col]
    missing_keys = [c for c in merge_keys if c not in df_activity.columns or c not in df_sleep.columns]
    if missing_keys:
        raise KeyError(f"병합 키가 CSV에 없습니다: {missing_keys}")

    df = pd.merge(df_activity, df_sleep, on=merge_keys, how="inner", suffixes=("_activity", "_sleep"))
    if df.empty:
        raise ValueError("activity와 sleep을 병합한 결과가 비었습니다. patient_id/sample_date/split/label을 확인하세요.")

    print(f"activity shape: {df_activity.shape}")
    print(f"sleep shape: {df_sleep.shape}")
    print(f"merged shape: {df.shape}")
    print(f"target column: {target_col}")
    print("target distribution:")
    print(df[target_col].value_counts().sort_index().to_string())
    return df, target_col



def choose_target_col_single(df: pd.DataFrame, requested: str) -> str:
    """단일 activity/sleep CSV에서 정답 컬럼을 고릅니다."""
    # auto이면 CSV 안에 있는 target_class 또는 binary_class를 자동으로 찾습니다.
    if requested != "auto":
        if requested not in df.columns:
            raise KeyError(f"요청한 target 컬럼이 CSV에 없습니다: {requested}")
        return requested

    for col in ("target_class", "binary_class"):
        if col in df.columns:
            return col
    raise KeyError("target_class 또는 binary_class 컬럼을 찾지 못했습니다.")


def load_discrete_dataset(path: Path, target_col: str, source_split: str, dataset_name: str) -> tuple[pd.DataFrame, str]:
    """논문 RF/LGBM 재현용으로 activity 또는 sleep 비연속형 CSV 하나를 읽습니다."""
    df = read_dataset(path)
    target_col = choose_target_col_single(df, target_col)

    if source_split != "all":
        if "split" not in df.columns:
            raise KeyError("source_split을 사용하려면 split 컬럼이 필요합니다.")
        df = df[df["split"] == source_split].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"{dataset_name}: split={source_split!r} 데이터가 없습니다.")

    print(f"\n[Paper Data] {dataset_name}")
    print(f"file: {path}")
    print(f"source_split: {source_split}")
    print(f"shape: {df.shape}")
    print("target distribution:")
    print(df[target_col].value_counts().sort_index().to_string())
    return df, target_col


def paper_class_weights(labels: list[int]) -> list[object]:
    """논문 RF/LGBM 설명의 class_weight 탐색 범위를 binary label에 맞춰 구성합니다."""
    # class_weight는 정상/치매 데이터 개수가 불균형할 때 소수 클래스를 더 중요하게 보도록 하는 값입니다.
    if len(labels) != 2:
        return [None, "balanced"]
    negative, positive = labels[0], labels[-1]
    return [None, "balanced", *({negative: 1, positive: w} for w in (1, 5, 10, 20, 30))]


def paper_parameter_distributions(labels: list[int]) -> dict[str, dict[str, object]]:
    """논문에 명시된 Random Search 범위를 RF/LGBM용 분포로 만듭니다."""
    # max_depth, learning_rate, num_leaves, n_estimators 등 여러 모델 설정 후보를 준비합니다.
    if len(labels) == 2:
        negative, positive = labels[0], labels[-1]
        lgbm_weights = [None, "balanced", {negative: 1, positive: 10}, {negative: 1, positive: 20}]
    else:
        lgbm_weights = [None, "balanced"]

    return {
        "lgbm": {
            "max_depth": np.arange(3, 16),
            "learning_rate": np.linspace(0.01, 0.30, 30),
            "num_leaves": np.arange(20, 151),
            "class_weight": lgbm_weights,
            "n_estimators": [100, 200, 300, 500],
        },
        "rf": {
            "max_depth": list(range(1, 21)),
            "class_weight": paper_class_weights(labels),
            "n_estimators": [100, 200, 300, 500],
        },
    }



def fit_random_search_with_progress(
    model,
    param_distributions: dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int,
    cv_folds: int,
    random_state: int,
    desc: str,
    progress_enabled: bool,
):
    """RandomizedSearchCV와 같은 후보 샘플링을 하되, fold 단위 진행률을 보여줍니다."""
    # params_list는 이번에 시험할 모델 설정 조합 목록입니다.
    params_list = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=random_state))
    if not params_list:
        raise ValueError(f"{desc}: Random Search 후보가 없습니다.")

    # 각 검증 조각마다 정상/치매 비율이 비슷하도록 나눕니다.
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=False)
    # 지금까지 찾은 가장 좋은 정확도와 모델 설정을 기억합니다.
    best_score = -1.0
    best_params = None
    total_fits = len(params_list) * cv_folds

    with make_progress_bar(total_fits, f"{desc} CV", progress_enabled) as progress:
        for param_idx, params in enumerate(params_list, 1):
            fold_scores = []
            for train_idx, valid_idx in splitter.split(X_train, y_train):
                X_fold = X_train.iloc[train_idx]
                y_fold = y_train.iloc[train_idx]
                X_valid = X_train.iloc[valid_idx]
                y_valid = y_train.iloc[valid_idx]

                estimator = clone(model)
                estimator.set_params(**params)
                estimator.fit(X_fold, y_fold)
                pred = estimator.predict(X_valid)
                fold_scores.append(accuracy_score(y_valid, pred))
                progress.update(1)

            mean_score = float(np.mean(fold_scores))
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
            progress.set_postfix({
                "param": f"{param_idx}/{len(params_list)}",
                "mean_acc": f"{mean_score:.4f}",
                "best_acc": f"{best_score:.4f}",
            })

    print(f"[{desc}] Final fit with best parameters...")
    best_estimator = clone(model)
    best_estimator.set_params(**best_params)
    best_estimator.fit(X_train, y_train)
    return argparse.Namespace(
        best_estimator_=best_estimator,
        best_params_=best_params,
        best_score_=best_score,
    )



def paper_model_candidates(labels: list[int], random_state: int, model_n_jobs: int) -> dict[str, dict[str, object]]:
    """반복되는 모델 준비 코드를 한 곳에 모아 보일러플레이트를 줄입니다."""
    base_params = lgbm_base_params(labels, random_state)
    base_params["n_jobs"] = model_n_jobs
    distributions = paper_parameter_distributions(labels)
    return {
        "LightGBM": {
            "model": LGBMClassifier(**base_params),
            "params": distributions["lgbm"],
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=random_state, n_jobs=model_n_jobs, verbose=0),
            "params": distributions["rf"],
        },
    }


def fit_paper_base_models(
    dataset_name: str,
    labels: list[int],
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    random_search_iter: int,
    cv_folds: int,
    random_state: int,
    model_n_jobs: int,
    progress_enabled: bool,
) -> dict[str, argparse.Namespace]:
    """LightGBM/RF처럼 같은 방식으로 찾는 모델들을 반복문 하나로 학습합니다."""
    searches = {}
    for model_name, spec in paper_model_candidates(labels, random_state, model_n_jobs).items():
        print(f"\n[Paper Search] {dataset_name} {model_name} Random Search")
        searches[model_name] = fit_random_search_with_progress(
            spec["model"],
            spec["params"],
            X_train_scaled,
            y_train,
            random_search_iter,
            cv_folds,
            random_state,
            f"PAPER {dataset_name} {model_name}",
            progress_enabled,
        )
    return searches


def add_paper_ensemble_model(
    searches: dict[str, argparse.Namespace],
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, argparse.Namespace]:
    """개별 모델 결과에 RF+LGBM 앙상블 모델을 추가합니다."""
    ensemble = VotingClassifier(
        estimators=[
            ("lgbm", searches["LightGBM"].best_estimator_),
            ("rf", searches["RandomForest"].best_estimator_),
        ],
        voting="soft",
    )
    ensemble.fit(X_train_scaled, y_train)
    searches["RF_LGBM_Ensemble"] = argparse.Namespace(
        best_estimator_=ensemble,
        best_params_={
            "lgbm": searches["LightGBM"].best_params_,
            "rf": searches["RandomForest"].best_params_,
            "voting": "soft",
        },
        best_score_=max(searches["LightGBM"].best_score_, searches["RandomForest"].best_score_),
    )
    return searches


def evaluate_paper_models(
    searches: dict[str, argparse.Namespace],
    X_test_scaled: pd.DataFrame,
    y_test: pd.Series,
    labels: list[int],
) -> dict[str, tuple[object, dict[str, object]]]:
    """모든 paper 모델을 같은 방식으로 평가해 결과 구조를 통일합니다."""
    models = {}
    for model_name, search in searches.items():
        metrics = evaluate_model(search.best_estimator_, X_test_scaled, y_test, labels)
        metrics["best_cv_accuracy"] = float(search.best_score_)
        metrics["model"] = search.best_estimator_
        models[model_name] = (search.best_params_, metrics)
    return models


def fit_paper_dataset(
    dataset_name: str,
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
    random_search_iter: int,
    cv_folds: int,
    search_n_jobs: int,
    model_n_jobs: int,
    progress_enabled: bool = True,
) -> dict[str, object]:
    """논문 RF/LGBM 방식에 맞춰 단일 비연속형 데이터셋을 학습/평가합니다."""
    # X는 모델 입력값, y는 모델이 맞혀야 하는 정답입니다.
    features = feature_columns(df, target_col)
    X = df[features]
    y = df[target_col].astype(int)
    labels = class_labels(y)

    # 전체 데이터를 학습용 70%, 최종 확인용 30%로 나눕니다.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 숫자 범위를 0~1 사이로 맞춰 모델 학습을 안정적으로 만듭니다.
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features).reset_index(drop=True)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features).reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # 진행률을 정확히 보여주기 위해 Random Search 자체는 순서대로 돌립니다.
    # 대신 MODEL_N_JOBS=-1로 두면 각 모델 학습이 Colab CPU 코어를 활용합니다.
    if search_n_jobs != 1:
        print("[Paper Search] 진행률 표시 모드에서는 search_n_jobs 대신 model_n_jobs로 각 fit을 병렬화합니다.")

    # 보일러플레이트 방지: LightGBM/RF를 같은 helper로 처리하고, 앙상블/평가도 공통 함수로 묶습니다.
    searches = fit_paper_base_models(
        dataset_name,
        labels,
        X_train_scaled,
        y_train,
        random_search_iter,
        cv_folds,
        random_state,
        model_n_jobs,
        progress_enabled,
    )
    searches = add_paper_ensemble_model(searches, X_train_scaled, y_train)
    models = evaluate_paper_models(searches, X_test_scaled, y_test, labels)

    return {
        "dataset_name": dataset_name,
        "target_col": target_col,
        "features": features,
        "scaler": scaler,
        "labels": labels,
        "rows": len(df),
        "patients": df["patient_id"].nunique() if "patient_id" in df.columns else None,
        "models": models,
    }


def print_paper_result(dataset_name: str, model_name: str, best_params: object, metrics: dict[str, object], labels: list[int]) -> None:
    """논문 표와 같은 accuracy 중심으로 RF/LGBM 결과를 출력합니다."""
    print(f"\n[PAPER {dataset_name} - {model_name}]")
    print("Best Params:", best_params)
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"Macro F1 Score: {metrics['f1_macro']:.4f}")
    best_cv_accuracy = metrics.get("best_cv_accuracy")
    if best_cv_accuracy is not None:
        print(f"Best CV Accuracy: {best_cv_accuracy:.4f}")
    auc = metrics["auc_macro"]
    print(f"AUC Score: {auc:.4f}" if not np.isnan(auc) else "AUC Score: nan")
    print("Classification Report:\n", metrics["report"])
    print(f"Confusion Matrix (Row: True, Col: Pred - {class_names(labels)}):\n", metrics["cm"])


def save_paper_results(results: dict[str, dict[str, object]], output_dir: Path | None) -> None:
    """논문 재현 모드 결과를 CSV와 모델 파일로 저장합니다."""
    if output_dir is None:
        output_dir = SCRIPT_DIR / "model_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    import joblib

    summary = []
    for dataset_name, bundle in results.items():
        pd.Series(bundle["features"], name="feature").to_csv(
            output_dir / f"paper_{dataset_name}_selected_features.csv",
            index=False,
        )
        joblib.dump(bundle["scaler"], output_dir / f"paper_{dataset_name}_feature_scaler.pkl")

        for model_name, (best_params, metrics) in bundle["models"].items():
            safe_model = model_name.lower().replace(" ", "_")
            joblib.dump(metrics["model"], output_dir / f"paper_{dataset_name}_{safe_model}_model.pkl")
            summary.append({
                "dataset_type": f"PAPER_{dataset_name}",
                "model_name": model_name,
                "target_col": bundle["target_col"],
                "rows": bundle["rows"],
                "patients": bundle["patients"],
                "features": len(bundle["features"]),
                "accuracy": metrics["accuracy"],
                "accuracy_percent": metrics["accuracy"] * 100,
                "f1_macro": metrics["f1_macro"],
                "auc_macro": metrics["auc_macro"],
                "best_cv_accuracy": metrics.get("best_cv_accuracy", float("nan")),
                "best_params": json.dumps(best_params, default=str),
            })

    results_df = pd.DataFrame(summary)
    results_df.to_csv(output_dir / "paper_experiment_results.csv", index=False)
    best_idx = results_df["accuracy"].idxmax()
    best_model = results_df.loc[best_idx]
    print(f"\n[Paper 결과 저장 완료] → {output_dir}")
    print(
        f"[Paper 최고 성능] {best_model['dataset_type']} {best_model['model_name']}: "
        f"Accuracy={best_model['accuracy_percent']:.2f}%"
    )


def run_paper_experiment(
    activity_file: Path,
    sleep_file: Path,
    target_col: str = "auto",
    test_size: float = 0.30,
    random_state: int = 42,
    random_search_iter: int = 20,
    cv_folds: int = 3,
    source_split: str = "train",
    output_dir: Path | None = None,
    search_n_jobs: int = 1,
    model_n_jobs: int = -1,
    progress_enabled: bool = True,
) -> dict[str, object]:
    """논문 RF/LGBM 설정에 맞춰 activity와 sleep 비연속형 데이터를 각각 평가합니다."""
    start_time = datetime.now()
    print("Start:", start_time)
    print("\n[Paper Mode]")
    print("- 목표: 논문 Table 3의 Discrete Ensemble 조건 재현")
    print("- 데이터: activity/sleep 비연속형 CSV를 각각 사용")
    print("- 분할: row-level stratified 70/30 split")
    print("- 탐색: accuracy 기준 Random Search")
    print("- 주의: 논문 92.72%는 RF/LGBM이 아니라 discrete+continuous LSTM 성능입니다.")

    dataset_files = {"activity": Path(activity_file), "sleep": Path(sleep_file)}
    results: dict[str, dict[str, object]] = {}
    resolved_target = target_col
    for dataset_name, path in dataset_files.items():
        df, resolved_target = load_discrete_dataset(path, target_col, source_split, dataset_name)
        bundle = fit_paper_dataset(
            dataset_name=dataset_name,
            df=df,
            target_col=resolved_target,
            test_size=test_size,
            random_state=random_state,
            random_search_iter=random_search_iter,
            cv_folds=cv_folds,
            search_n_jobs=search_n_jobs,
            model_n_jobs=model_n_jobs,
            progress_enabled=progress_enabled,
        )
        results[dataset_name] = bundle

    save_paper_results(results, output_dir)

    print("\n====================================================")
    print("PAPER MODE FINAL RESULTS")
    print("====================================================")
    for dataset_name, bundle in results.items():
        for model_name, (best_params, metrics) in bundle["models"].items():
            print_paper_result(dataset_name, model_name, best_params, metrics, bundle["labels"])

    end_time = datetime.now()
    print("\nEnd:", end_time)
    print("Elapsed:", end_time - start_time)
    return {"results": results, "target_col": resolved_target, "output_dir": output_dir}


def feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    drop_cols = META_COLS | {target_col}
    features = [c for c in df.columns if c not in drop_cols]
    non_numeric = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise TypeError(f"숫자형이 아닌 feature가 남아 있습니다: {non_numeric[:10]}")
    if not features:
        raise ValueError("학습에 사용할 feature가 없습니다.")
    return features


def add_rolling_features(df: pd.DataFrame, target_col: str, window: int = 7) -> pd.DataFrame:
    # ✏️ CHANGE B-3: add patient-level rolling statistics for temporal patterns
    print(f"\n[Feature Engineering] Adding rolling features (window={window} days)...")
    required_cols = {"patient_id", "sample_date", target_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise KeyError(f"rolling feature 생성에 필요한 컬럼이 없습니다: {sorted(missing_cols)}")

    df = df.sort_values(["patient_id", "sample_date"]).copy()
    meta_cols = META_COLS | {target_col}
    numeric_cols = [
        c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_cols:
        print("  → No numeric base features found; skipping rolling feature generation.")
        return df

    rolling_mean = (
        df.groupby("patient_id", sort=False)[numeric_cols]
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
        .add_suffix(f"_roll{window}d_mean")
    )
    rolling_std = (
        df.groupby("patient_id", sort=False)[numeric_cols]
        .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        .add_suffix(f"_roll{window}d_std")
    )

    df_enhanced = pd.concat([df, rolling_mean, rolling_std], axis=1)
    added_features = len(rolling_mean.columns) + len(rolling_std.columns)
    print(f"  → Added {added_features} rolling features from {len(numeric_cols)} base features")
    return df_enhanced


def class_labels(y: pd.Series) -> list[int]:
    labels = sorted(pd.Series(y).dropna().astype(int).unique().tolist())
    if len(labels) < 2:
        raise ValueError(f"분류 클래스가 2개 미만입니다: {labels}")
    return labels


def class_names(labels: list[int]) -> list[str]:
    default = {0: "Normal(0)", 1: "MCI/Dementia(1)", 2: "Dementia(2)"}
    return [default.get(label, f"Class {label}") for label in labels]


def lgbm_base_params(labels: list[int], random_state: int) -> dict[str, object]:
    params: dict[str, object] = {
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }
    if len(labels) == 2:
        params["objective"] = "binary"
    else:
        params["objective"] = "multiclass"
        params["num_class"] = len(labels)
    return params


def make_lgbm_f1_eval(labels: list[int]):
    label_array = np.asarray(labels)

    def lgbm_f1_eval(y_true, y_pred):
        pred = np.asarray(y_pred)
        if len(labels) == 2:
            proba = pred[:, 1] if pred.ndim == 2 else pred
            y_pred_class = np.where(proba >= 0.5, labels[-1], labels[0])
        else:
            # 🔧 FIX 3: reshape multiclass predictions by sample count, not by a fixed class-major layout.
            if pred.ndim == 1:
                n_samples = len(y_true)
                n_classes = len(labels)
                pred = pred.reshape(n_samples, n_classes)
            y_pred_class = label_array[np.argmax(pred, axis=1)]

        f1 = f1_score(y_true, y_pred_class, average="macro", zero_division=0)
        return "f1_macro", f1, True

    return lgbm_f1_eval


def safe_auc(y_true: pd.Series, prob: np.ndarray, labels: list[int]) -> float:
    try:
        if len(labels) == 2:
            positive_label = labels[-1]
            positive_idx = list(labels).index(positive_label)
            return float(roc_auc_score(y_true, prob[:, positive_idx]))
        return float(roc_auc_score(y_true, prob, labels=labels, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, labels: list[int]) -> dict[str, object]:
    prob = model.predict_proba(X)
    pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "auc_macro": safe_auc(y, prob, labels),
        "report": classification_report(
            y,
            pred,
            labels=labels,
            target_names=class_names(labels),
            zero_division=0,
        ),
        "cm": confusion_matrix(y, pred, labels=labels),
    }


def select_top_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    labels: list[int],
    top_k: int,
    random_state: int,
) -> list[str]:
    print("\n[Feature Selection] Base 모델 학습 및 중요도 추출 중...")
    proxy_model = LGBMClassifier(
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    proxy_model.fit(X_train, y_train)

    if ShapAnalyzer is not None:
        print("[Feature Selection] xai.ShapAnalyzer를 사용합니다.")
        task = "multiclass" if len(labels) > 2 else "binary"
        kwargs = {
            "model": proxy_model,
            "feature_names": list(X_train.columns),
            "task": task,
        }
        if len(labels) > 2:
            kwargs["n_classes"] = len(labels)
            kwargs["class_names"] = class_names(labels)

        analyzer = ShapAnalyzer(**kwargs)
        analyzer.explain(X_train)
        shap_df = analyzer.to_dataframe(combine_classes=False)
        selected = shap_df.head(min(top_k, len(shap_df)))["feature"].tolist()
        print(f"\n[SHAP 추출 완료] Top {len(selected)} Features:")
        for i, feat in enumerate(selected, 1):
            print(f"  {i}. {feat}")
        return selected

    print("[Feature Selection] xai 패키지가 없어 LightGBM feature importance로 대체합니다.")
    # ✏️ CHANGE B-4: improved fallback feature ranking with normalized importance
    importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": proxy_model.feature_importances_,
        }
    )
    total_importance = importance["importance"].sum()
    if total_importance > 0:
        importance["importance_pct"] = importance["importance"] / total_importance * 100
    else:
        importance["importance_pct"] = 0
    importance = importance.sort_values(["importance", "feature"], ascending=[False, True])
    top_k = min(top_k, len(importance))
    selected = importance.head(top_k)["feature"].tolist()
    print(f"\n[Feature Selection 완료] Top {top_k} Features (LightGBM Importance):")
    for i, (_, row) in enumerate(importance.head(top_k).iterrows(), 1):
        print(f"  {i:2d}. {row['feature']:<30} ({row['importance_pct']:5.1f}%)")
    return selected


def class_weight_grid(labels: list[int]) -> list[object]:
    # 🔧 FIX 2: use class_weight dictionaries that contain every active class.
    if set(labels) == {0, 1, 2}:
        return [None, "balanced", {0: 1, 1: 5, 2: 5}, {0: 1, 1: 10, 2: 10}, {0: 1, 1: 20, 2: 20}]
    return [None, "balanced", {labels[0]: 1, labels[1]: 5}, {labels[0]: 1, labels[1]: 10}, {labels[0]: 1, labels[1]: 20}]


def parameter_grids(labels: list[int]) -> dict[str, list[tuple]]:
    # ✏️ CHANGE B-1: expanded grids optimized for medical tabular data
    class_weights = class_weight_grid(labels)
    lgbm_no_smote = list(product(
        [15, 31, 63],
        [0.03, 0.05, 0.08],
        [500, 1000, 1500],
        [20, 41, 80],
        [5, 8, 10, -1],
        class_weights,
        [0.7, 0.8, 1.0],
        [0.0, 0.1, 1.0],
        [0.0, 1.0, 5.0],
    ))
    rf_no_smote = list(product(
        [5, 10, 15, None],
        [300, 500, 1000],
        class_weights,
        [1, 2, 5],
        ["sqrt", "log2", 0.5],
    ))
    lgbm_smote = list(product(
        [15, 31, 63],
        [0.03, 0.05, 0.08],
        [500, 1000],
        [20, 41],
        [5, 8, 10],
        [None],
        [0.7, 0.8],
        [0.1, 1.0],
        [1.0, 5.0],
    ))
    rf_smote = list(product(
        [5, 10, 15],
        [300, 500],
        [None],
        [2, 5],
        ["sqrt", 0.5],
    ))
    return {
        "lgbm_no_smote": lgbm_no_smote,
        "rf_no_smote": rf_no_smote,
        "lgbm_smote": lgbm_smote,
        "rf_smote": rf_smote,
    }


def cv_splits(y: pd.Series, groups: np.ndarray) -> int:
    # ✏️ CHANGE A-1: use majority vote for patient representative labels
    group_df = pd.DataFrame({"group": groups, "label": y.to_numpy()})
    representative = (
        group_df.groupby("group")["label"]
        .agg(lambda x: x.value_counts().index[0])
    )
    class_counts = representative.value_counts()
    if class_counts.empty:
        raise ValueError("CV split 생성을 위한 patient representative label이 없습니다.")
    min_count = int(class_counts.min())
    n_splits = max(2, min(5, min_count))

    print(f"  [CV Splits] Patients per class: {class_counts.sort_index().to_dict()}")
    print(f"  [CV Splits] Using n_splits={n_splits}")
    if n_splits < 3:
        print("  ⚠️  WARNING: n_splits < 3. Consider more patient data for stable CV.")

    return n_splits


def save_results(
    results: dict,
    best_features: list[str],
    scaler: MinMaxScaler,
    target_col: str,
    output_dir: Path = None,
) -> None:
    # ✏️ CHANGE B-5: save models and results for reproducibility
    if output_dir is None:
        output_dir = SCRIPT_DIR / "model_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    import joblib

    pd.Series(best_features, name="feature").to_csv(output_dir / "selected_features.csv", index=False)
    joblib.dump(scaler, output_dir / "feature_scaler.pkl")

    summary = []
    for dataset_type, models in results.items():
        for model_name, (params, metrics) in models.items():
            safe_dataset = dataset_type.lower()
            safe_model = model_name.lower().replace(" ", "_")
            model = metrics.get("model")
            if model is not None:
                joblib.dump(model, output_dir / f"{safe_dataset}_{safe_model}_model.pkl")

            summary.append({
                "dataset_type": dataset_type,
                "model_name": model_name,
                "target_col": target_col,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "auc_macro": metrics["auc_macro"],
                "best_cv_f1": metrics.get("best_cv_f1", float("nan")),
                "cv_test_gap": metrics["f1_macro"] - metrics.get("best_cv_f1", metrics["f1_macro"]),
                "best_params": json.dumps(params, default=str),
            })

    if not summary:
        print(f"\n[모델 저장 건너뜀] 저장할 결과가 없습니다: {output_dir}")
        return

    results_df = pd.DataFrame(summary)
    results_df.to_csv(output_dir / "experiment_results.csv", index=False)
    best_idx = results_df["f1_macro"].idxmax()
    best_model = results_df.loc[best_idx]

    print(f"\n[모델 저장 완료] → {output_dir}")
    print(f"[최고 성능 모델] {best_model['dataset_type']} {best_model['model_name']}: F1={best_model['f1_macro']:.4f}")


def print_leakage_check(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    # 🔧 FIX 5: verify patient overlap immediately after patient-level train/test split.
    train_patients = set(train_df["patient_id"].unique())
    test_patients = set(test_df["patient_id"].unique())
    overlap = train_patients & test_patients

    print(f"\n[Leakage Check] Train patients: {len(train_patients)}")
    print(f"[Leakage Check] Test patients: {len(test_patients)}")
    print(f"[Leakage Check] Overlapping patients: {len(overlap)}")

    if len(overlap) > 0:
        print(f"  CRITICAL: {len(overlap)} patients in BOTH train and test!")
        print("  DATA LEAKAGE DETECTED - Results will be invalid!")
    else:
        print("  No patient overlap. Split is clinically valid.")
    print("=" * 65)


def fit_lgbm_cv(
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    X_test_scaled: pd.DataFrame,
    y_test: pd.Series,
    labels: list[int],
    params_list: list[tuple],
    train_groups: np.ndarray,
    random_state: int,
    smote=None,
    progress_enabled: bool = True,
    progress_desc: str = "LightGBM",
) -> tuple[tuple, dict[str, object]]:
    # 🔧 FIX 1: use patient-aware CV so a patient never appears in both fold train and validation.
    # ✏️ CHANGE B-2: fail clearly if LightGBM search space is empty
    if not params_list:
        raise ValueError("LightGBM parameter grid가 비어 있습니다.")
    n_splits = cv_splits(y_train, train_groups)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_score = -1.0
    best_params = None

    # ✏️ CHANGE C-1: show CV fit progress with current/best macro F1
    total_fits = len(params_list) * n_splits
    with make_progress_bar(total_fits, f"{progress_desc} CV", progress_enabled) as progress:
        for param_idx, params in enumerate(params_list, 1):
            fold_scores = []
            for train_idx, valid_idx in splitter.split(X_train_scaled, y_train, groups=train_groups):
                X_fold = X_train_scaled.iloc[train_idx]
                y_fold = y_train.iloc[train_idx]
                X_valid = X_train_scaled.iloc[valid_idx]
                y_valid = y_train.iloc[valid_idx]
                if smote is not None:
                    X_fold, y_fold = smote.fit_resample(X_fold, y_fold)

                model = LGBMClassifier(
                    # ✏️ CHANGE B-2: LightGBM with regularization parameters
                    **lgbm_base_params(labels, random_state),
                    num_leaves=params[0],
                    learning_rate=params[1],
                    n_estimators=params[2],
                    min_child_samples=params[3],
                    max_depth=params[4],
                    class_weight=params[5],
                    colsample_bytree=params[6],
                    reg_alpha=params[7],
                    reg_lambda=params[8],
                    subsample=0.8,
                    min_split_gain=0.01,
                )
                model.fit(
                    X_fold,
                    y_fold,
                    eval_set=[(X_valid, y_valid)],
                    eval_metric=make_lgbm_f1_eval(labels),
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
                )
                fold_scores.append(evaluate_model(model, X_valid, y_valid, labels)["f1_macro"])
                progress.update(1)

            mean_score = float(np.mean(fold_scores))
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
            progress.set_postfix({
                "param": f"{param_idx}/{len(params_list)}",
                "mean_f1": f"{mean_score:.4f}",
                "best_f1": f"{best_score:.4f}",
            })

    final_X, final_y = X_train_scaled, y_train
    if smote is not None:
        final_X, final_y = smote.fit_resample(final_X, final_y)

    print(f"[{progress_desc}] Final fit with best parameters...")
    final_model = LGBMClassifier(
        # ✏️ CHANGE B-2: LightGBM with regularization parameters
        **lgbm_base_params(labels, random_state),
        num_leaves=best_params[0],
        learning_rate=best_params[1],
        n_estimators=best_params[2],
        min_child_samples=best_params[3],
        max_depth=best_params[4],
        class_weight=best_params[5],
        colsample_bytree=best_params[6],
        reg_alpha=best_params[7],
        reg_lambda=best_params[8],
        subsample=0.8,
        min_split_gain=0.01,
    )
    final_model.fit(final_X, final_y)
    metrics = evaluate_model(final_model, X_test_scaled, y_test, labels)
    # 🔧 FIX 6: keep best CV F1 for final overfitting gap monitoring.
    metrics["best_cv_f1"] = best_score
    # ✏️ CHANGE B-5: keep fitted model available for persistence
    metrics["model"] = final_model
    return best_params, metrics


def fit_rf_cv(
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    X_test_scaled: pd.DataFrame,
    y_test: pd.Series,
    labels: list[int],
    params_list: list[tuple],
    train_groups: np.ndarray,
    random_state: int,
    smote=None,
    progress_enabled: bool = True,
    progress_desc: str = "RandomForest",
) -> tuple[tuple, dict[str, object]]:
    # 🔧 FIX 1: use patient-aware CV so a patient never appears in both fold train and validation.
    # ✏️ CHANGE B-2: fail clearly if RandomForest search space is empty
    if not params_list:
        raise ValueError("RandomForest parameter grid가 비어 있습니다.")
    n_splits = cv_splits(y_train, train_groups)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_score = -1.0
    best_params = None

    # ✏️ CHANGE C-1: show CV fit progress with current/best macro F1
    total_fits = len(params_list) * n_splits
    with make_progress_bar(total_fits, f"{progress_desc} CV", progress_enabled) as progress:
        for param_idx, params in enumerate(params_list, 1):
            fold_scores = []
            for train_idx, valid_idx in splitter.split(X_train_scaled, y_train, groups=train_groups):
                X_fold = X_train_scaled.iloc[train_idx]
                y_fold = y_train.iloc[train_idx]
                X_valid = X_train_scaled.iloc[valid_idx]
                y_valid = y_train.iloc[valid_idx]
                if smote is not None:
                    X_fold, y_fold = smote.fit_resample(X_fold, y_fold)

                model = RandomForestClassifier(
                    # ✏️ CHANGE B-2: RandomForest with enhanced regularization
                    max_depth=params[0],
                    n_estimators=params[1],
                    class_weight=params[2],
                    min_samples_leaf=params[3],
                    max_features=params[4],
                    random_state=random_state,
                    n_jobs=-1,
                    verbose=0,
                )
                model.fit(X_fold, y_fold)
                fold_scores.append(evaluate_model(model, X_valid, y_valid, labels)["f1_macro"])
                progress.update(1)

            mean_score = float(np.mean(fold_scores))
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
            progress.set_postfix({
                "param": f"{param_idx}/{len(params_list)}",
                "mean_f1": f"{mean_score:.4f}",
                "best_f1": f"{best_score:.4f}",
            })

    final_X, final_y = X_train_scaled, y_train
    if smote is not None:
        final_X, final_y = smote.fit_resample(final_X, final_y)

    print(f"[{progress_desc}] Final fit with best parameters...")
    final_model = RandomForestClassifier(
        # ✏️ CHANGE B-2: RandomForest with enhanced regularization
        max_depth=best_params[0],
        n_estimators=best_params[1],
        class_weight=best_params[2],
        min_samples_leaf=best_params[3],
        max_features=best_params[4],
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    final_model.fit(final_X, final_y)
    metrics = evaluate_model(final_model, X_test_scaled, y_test, labels)
    # 🔧 FIX 6: keep best CV F1 for final overfitting gap monitoring.
    metrics["best_cv_f1"] = best_score
    # ✏️ CHANGE B-5: keep fitted model available for persistence
    metrics["model"] = final_model
    return best_params, metrics


def print_result(dataset_type: str, model_name: str, best_params: tuple, metrics: dict[str, object], labels: list[int]) -> None:
    print(f"\n[{dataset_type} - {model_name}]")
    print("Best Params:", best_params)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1 Score: {metrics['f1_macro']:.4f}")
    # 🔧 FIX 6: show CV-vs-test gap to make overfitting visible.
    best_cv_f1 = metrics.get("best_cv_f1")
    if best_cv_f1 is not None:
        gap = metrics["f1_macro"] - best_cv_f1
        status = "Stable" if gap >= -0.05 else "Possible Overfit"
        print(f"Best CV F1 Score: {best_cv_f1:.4f}")
        print(f"Test F1 Score   : {metrics['f1_macro']:.4f}")
        print(f"Performance Gap : {gap:+.4f} ({status})")
    auc = metrics["auc_macro"]
    print(f"AUC Score: {auc:.4f}" if not np.isnan(auc) else "AUC Score: nan")
    print("Classification Report:\n", metrics["report"])
    print(f"Confusion Matrix (Row: True, Col: Pred - {class_names(labels)}):\n", metrics["cm"])


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir, args.colab)
    print_colab_runtime_info(args.colab, output_dir)
    progress_enabled = not args.no_progress

    if args.mode == "paper":
        run_paper_experiment(
            activity_file=args.activity_file,
            sleep_file=args.sleep_file,
            target_col=args.target_col,
            test_size=args.test_size,
            random_state=args.random_state,
            random_search_iter=args.random_search_iter,
            cv_folds=args.cv_folds,
            source_split=args.source_split,
            output_dir=output_dir,
            search_n_jobs=args.search_n_jobs,
            model_n_jobs=args.model_n_jobs,
            progress_enabled=progress_enabled,
        )
        return

    start_time = datetime.now()
    print("Start:", start_time)

    df, target_col = load_merged_data(args.activity_file, args.sleep_file, args.target_col)
    # ✏️ CHANGE B-3: apply rolling feature engineering
    df = add_rolling_features(df, target_col, window=7)
    all_features = feature_columns(df, target_col)
    labels = class_labels(df[target_col])

    # 🔧 FIX 1: split by patient_id to eliminate train/test patient leakage.
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    train_idx, test_idx = next(gss.split(df, df[target_col], groups=df["patient_id"]))
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    # ✏️ CHANGE A-2: verify all classes present in both train and test
    train_classes = set(train_df[target_col].unique())
    test_classes = set(test_df[target_col].unique())
    all_classes = set(df[target_col].unique())

    print(f"\n[클래스 분포 검증]")
    print(f"전체: {df[target_col].value_counts().sort_index().to_dict()}")
    print(f"Train: {train_df[target_col].value_counts().sort_index().to_dict()}")
    print(f"Test : {test_df[target_col].value_counts().sort_index().to_dict()}")

    missing_train = all_classes - train_classes
    missing_test = all_classes - test_classes
    if missing_train or missing_test:
        print(f"  ❌ Train 누락 클래스: {missing_train}")
        print(f"  ❌ Test 누락 클래스: {missing_test}")
        print(f"  → random_state를 변경하거나 데이터를 확인하세요.")
        sys.exit(1)
    else:
        print(f"  ✅ 모든 클래스가 Train/Test에 존재합니다.")

    print_leakage_check(train_df, test_df)

    X_train_full = train_df[all_features]
    y_train = train_df[target_col].astype(int)
    X_test_full = test_df[all_features]
    y_test = test_df[target_col].astype(int)

    best_features = select_top_features(X_train_full, y_train, labels, args.top_k, args.random_state)

    print("\n[전처리] 선택된 피처에 대해 MinMaxScaler를 적용합니다...")
    scaler = MinMaxScaler()
    # ✏️ CHANGE A-3: reset indices for safe iloc-based CV operations
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_full[best_features]),
        columns=best_features,
    ).reset_index(drop=True)
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_full[best_features]),
        columns=best_features,
    ).reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    train_groups = train_df["patient_id"].reset_index(drop=True).values

    grids = limit_parameter_grids(
        parameter_grids(labels),
        max_combinations=args.max_grid_combinations,
        random_state=args.random_state,
    )
    results: dict[str, dict[str, tuple[tuple, dict[str, object]]]] = {"NO_SMOTE": {}, "SMOTE": {}}

    print("\n====================================================")
    print("NO_SMOTE LightGBM CV")
    print("====================================================")
    results["NO_SMOTE"]["LightGBM"] = fit_lgbm_cv(
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        labels,
        grids["lgbm_no_smote"],
        train_groups,
        args.random_state,
        progress_enabled=progress_enabled,
        progress_desc="NO_SMOTE LightGBM",
    )

    print("\n====================================================")
    print("NO_SMOTE Random Forest CV")
    print("====================================================")
    results["NO_SMOTE"]["RandomForest"] = fit_rf_cv(
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        labels,
        grids["rf_no_smote"],
        train_groups,
        args.random_state,
        progress_enabled=progress_enabled,
        progress_desc="NO_SMOTE RandomForest",
    )

    run_smote = not args.skip_smote and SMOTE is not None
    if run_smote:
        smote = SMOTE(random_state=args.random_state)
        print("\n====================================================")
        print("SMOTE LightGBM CV")
        print("====================================================")
        results["SMOTE"]["LightGBM"] = fit_lgbm_cv(
            X_train_scaled,
            y_train,
            X_test_scaled,
            y_test,
            labels,
            grids["lgbm_smote"],
            train_groups,
            args.random_state,
            smote=smote,
            progress_enabled=progress_enabled,
            progress_desc="SMOTE LightGBM",
        )

        print("\n====================================================")
        print("SMOTE Random Forest CV")
        print("====================================================")
        results["SMOTE"]["RandomForest"] = fit_rf_cv(
            X_train_scaled,
            y_train,
            X_test_scaled,
            y_test,
            labels,
            grids["rf_smote"],
            train_groups,
            args.random_state,
            smote=smote,
            progress_enabled=progress_enabled,
            progress_desc="SMOTE RandomForest",
        )
    else:
        reason = "--skip-smote 옵션 사용" if args.skip_smote else "imbalanced-learn 패키지 미설치"
        print(f"\n[SMOTE 건너뜀] {reason}. NO_SMOTE 결과만 출력합니다.")

    # ✏️ CHANGE B-5: save results and models
    save_results(results, best_features, scaler, target_col, output_dir=output_dir)

    print("\n====================================================")
    print("FINAL RESULTS")
    print("====================================================")
    for dataset_type, by_model in results.items():
        for model_name, (best_params, metrics) in by_model.items():
            print_result(dataset_type, model_name, best_params, metrics, labels)

    end_time = datetime.now()
    print("\nEnd:", end_time)
    print("Elapsed:", end_time - start_time)


if __name__ == "__main__":
    main()
