# Auto-generated Python script converted from a Jupyter notebook.
# Source notebook: SangHyo/previous/Experiment3.ipynb
# Do not edit this generated file if you need exact notebook parity; edit the source notebook or copy this file first.

# Notebook compatibility helpers. Generated to keep notebook shell/magic cells runnable as Python.
import os as _NOTEBOOK_OS
import subprocess as _NOTEBOOK_SUBPROCESS
from pathlib import Path as _NOTEBOOK_PATH


def _NOTEBOOK_RUN_SHELL(command: str) -> None:
    _NOTEBOOK_SUBPROCESS.run(command, shell=True, check=True)


def _NOTEBOOK_RUN_BASH(script: str) -> None:
    _NOTEBOOK_SUBPROCESS.run(script, shell=True, executable="/bin/bash", check=True)


def _NOTEBOOK_CD(path: str) -> None:
    _NOTEBOOK_OS.chdir(_NOTEBOOK_OS.path.expanduser(path))
    print(_NOTEBOOK_PATH.cwd())


# %% [markdown] cell 1
# # 3-Class DataSanity LSTM (1487 integrated features)
#
# `CN=0`, `MCI=1`, `Dementia=2` 다중 분류를 위한 Colab GPU 학습 노트북입니다.
#
# binary `DataSanity` 실험의 장점을 유지하면서 두 가지 누수 가능성을 차단합니다.
#
# 1. RF feature selection은 각 CV fold의 train subset에서만 수행합니다.
# 2. 최종 모델은 CV에서 선택한 epoch 수로 전체 train pool을 refit합니다.
#
# 공식 평가는 subject-level Macro F1입니다. 최종 test의 Dementia subject는 2명뿐이므로
# Dementia 단독 지표는 임상적 일반화 성능으로 해석하지 않습니다.

# %% [markdown] cell 2
# ## 실행 순서와 test 격리
#
# - `0~5`: 환경, 입력 계약, train-only sanity
# - `6~10`: train-only RF ranking, random search, seed 안정성 CV
# - `11`: 전체 train refit
# - `12~13`: 격리 test 1회 평가와 plot 저장
# - `14`: SHAP 담당팀 인계 번들 저장
#
# `X_test_raw`, `y_test`, `groups_test`는 입력 계약 확인을 위해 로드하지만,
# **셀 12 이전에는 모델 선택, feature 선택, epoch 선택에 사용하지 않습니다.**

# %% [markdown] cell 3
# ## 0. 환경 설정

# %% cell 4
import gc
import json
import os
import pickle
import random
import re
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception as exc:
    print("Google Drive mount skipped:", exc)

import tensorflow as tf
from tensorflow.keras import mixed_precision

SEED = 42
N_CLASSES = 3
CLASS_NAMES = ["CN", "MCI", "Dementia"]
CLASS_NAME_BY_ID = {0: "CN", 1: "MCI", 2: "Dementia"}
PADDING_VALUE = -1.0
N_SPLITS = 5
N_TRIALS = 8
TRIAL_EPOCHS = 60
TOP_K_OPTIONS = [20, 30, 40]
STABILITY_SEEDS = [42, 7, 2024]
RUN_SEED_STABILITY = True

# Update only these overrides when Drive paths differ.
INTEGRATED_PKL_OVERRIDE = "/content/drive/MyDrive/3class_DataSanity_1487/Data/lstm_dataset_3class.pkl"
RESULT_DIR_OVERRIDE = None

INTEGRATED_PKL_CANDIDATES = [
    Path("/content/drive/MyDrive/3class_DataSanity_1487/Data/lstm_dataset_3class.pkl"),
    Path("/content/drive/MyDrive/lstm_dataset_3class.pkl"),
    Path("/content/lstm_dataset_3class.pkl"),
]
RESULT_DIR = Path(
    RESULT_DIR_OVERRIDE
    or "/content/drive/MyDrive/3class_DataSanity_1487"
)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_global_seed(SEED)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    mixed_precision.set_global_policy("mixed_float16")
    print("GPU:", [gpu.name for gpu in gpus])
    print("precision:", mixed_precision.global_policy().name)
else:
    print("WARNING: CPU runtime detected. Colab GPU runtime is recommended.")
print("TensorFlow:", tf.__version__)
print("RESULT_DIR:", RESULT_DIR)

# %% cell 5
def resolve_path(override, candidates, description):
    paths = ([Path(override)] if override else []) + list(candidates)
    for path in paths:
        if path.exists():
            print(f"{description}: {path}")
            return path
    raise FileNotFoundError(
        f"{description} not found. Set the corresponding *_OVERRIDE value.\n"
        + "\n".join(str(path) for path in paths)
    )


def _numpy_frombuffer_compat(buf, dtype, shape, order, axis_order=None):
    # Load NumPy 2.4 pickles on Colab runtimes with an older helper.
    array = np.frombuffer(buf, dtype=dtype)
    if order == "K" and axis_order is not None:
        return array.reshape(shape, order="C").transpose(axis_order)
    return array.reshape(shape, order=order)


class CompatibleNumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy._core.numeric" and name == "_frombuffer":
            return _numpy_frombuffer_compat
        return super().find_class(module, name)


def load_pickle_compat(path):
    with Path(path).open("rb") as file:
        return CompatibleNumpyUnpickler(file).load()


INTEGRATED_PKL_PATH = resolve_path(
    INTEGRATED_PKL_OVERRIDE,
    INTEGRATED_PKL_CANDIDATES,
    "3-class integrated lstm_dataset_3class.pkl",
)

# %% [markdown] cell 6
# ## 1. 입력 계약: integrated tensor + embedded 3-class subject labels

# %% cell 7
integrated = load_pickle_compat(INTEGRATED_PKL_PATH)

required_integrated = {
    "X_integrated_seq",
    "y",
    "patient_id",
    "window_start_date",
    "window_end_date",
    "split",
    "integrated_feature_names",
    "meta",
}
assert required_integrated.issubset(integrated), (
    f"integrated pickle missing: {sorted(required_integrated - set(integrated))}"
)

X_all = np.asarray(integrated["X_integrated_seq"], dtype=np.float32)
y_all_3 = np.asarray(integrated["y"], dtype=np.int64)
groups_all = np.asarray(integrated["patient_id"]).astype(str)
starts_all = np.asarray(integrated["window_start_date"]).astype(str)
ends_all = np.asarray(integrated["window_end_date"]).astype(str)
split_all = np.asarray(integrated["split"]).astype(str)
feature_names = list(integrated["integrated_feature_names"])
meta = integrated["meta"]

assert meta.get("task") == "multiclass_classification", meta
assert meta.get("binary_collapse_exact_match") is True, meta
assert X_all.ndim == 3 and X_all.shape[1:] == (7, 1487), X_all.shape
assert X_all.shape[0] == len(y_all_3) == len(groups_all) == len(starts_all) == len(ends_all) == len(split_all)
assert X_all.shape[-1] == len(feature_names) == 1487
assert np.isfinite(X_all).all(), "integrated tensor contains NaN/Inf"
assert set(np.unique(y_all_3)) == {0, 1, 2}

train_mask = split_all == "train"
test_mask = split_all == "val"
assert train_mask.sum() == 5500 and test_mask.sum() == 1440
assert not (set(groups_all[train_mask]) & set(groups_all[test_mask])), "train/test subject overlap"

for subject_id in np.unique(groups_all):
    assert len(set(y_all_3[groups_all == subject_id])) == 1, f"subject label conflict: {subject_id}"


def subject_label_counts(labels, groups):
    return Counter(int(labels[np.flatnonzero(groups == subject_id)[0]]) for subject_id in np.unique(groups))


assert Counter(y_all_3[train_mask]) == Counter({0: 3359, 1: 1900, 2: 241})
assert Counter(y_all_3[test_mask]) == Counter({0: 1118, 1: 247, 2: 75})
assert subject_label_counts(y_all_3[train_mask], groups_all[train_mask]) == Counter({0: 84, 1: 46, 2: 8})
assert subject_label_counts(y_all_3[test_mask], groups_all[test_mask]) == Counter({0: 26, 1: 4, 2: 2})

X_train_raw = X_all[train_mask]
y_train = y_all_3[train_mask]
groups_train = groups_all[train_mask]
starts_train = starts_all[train_mask]
ends_train = ends_all[train_mask]

# Keep official validation isolated until the final evaluation section.
X_test_raw = X_all[test_mask]
y_test = y_all_3[test_mask]
groups_test = groups_all[test_mask]
starts_test = starts_all[test_mask]
ends_test = ends_all[test_mask]

del integrated, X_all, y_all_3, groups_all, starts_all, ends_all, split_all
gc.collect()

print("Input contract passed with formal 3-class labels.")
print("X_train_raw:", X_train_raw.shape)
print("X_test_raw :", X_test_raw.shape, "(isolated until final evaluation)")

# %% [markdown] cell 8
# ## 2. Train-only sanity와 CV split 검증

# %% cell 9
from sklearn.model_selection import StratifiedKFold


def subject_labels(y_values, group_values):
    frame = pd.DataFrame({"subject_id": group_values, "y": y_values})
    assert frame.groupby("subject_id")["y"].nunique().max() == 1
    return frame.drop_duplicates("subject_id").reset_index(drop=True)


def class_counts(y_values):
    return {CLASS_NAME_BY_ID[i]: int(np.sum(np.asarray(y_values) == i)) for i in range(N_CLASSES)}


train_subjects = subject_labels(y_train, groups_train)
print("Train windows :", class_counts(y_train))
print("Train subjects:", class_counts(train_subjects["y"].to_numpy()))
print("Train padding ratio:", float((X_train_raw == PADDING_VALUE).mean()))

# Split unique subjects first. Splitting repeated windows directly can leave a
# validation fold without Dementia even when every fold could contain that class.
splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
window_indices = np.arange(len(groups_train))
CV_SPLITS = []
for subject_train_idx, subject_val_idx in splitter.split(
    train_subjects["subject_id"],
    train_subjects["y"],
):
    train_subject_ids = set(train_subjects.iloc[subject_train_idx]["subject_id"])
    val_subject_ids = set(train_subjects.iloc[subject_val_idx]["subject_id"])
    train_idx = window_indices[np.isin(groups_train, list(train_subject_ids))]
    val_idx = window_indices[np.isin(groups_train, list(val_subject_ids))]
    CV_SPLITS.append((train_idx, val_idx))

fold_sanity_rows = []
for fold_no, (train_idx, val_idx) in enumerate(CV_SPLITS, start=1):
    train_groups = set(groups_train[train_idx])
    val_groups = set(groups_train[val_idx])
    assert not (train_groups & val_groups), f"fold {fold_no}: subject overlap"
    train_classes = sorted(np.unique(y_train[train_idx]).tolist())
    val_classes = sorted(np.unique(y_train[val_idx]).tolist())
    assert train_classes == val_classes == [0, 1, 2], (
        f"fold {fold_no}: train={train_classes}, val={val_classes}"
    )
    val_subjects = subject_labels(y_train[val_idx], groups_train[val_idx])
    fold_sanity_rows.append(
        {
            "fold": fold_no,
            "train_windows": len(train_idx),
            "val_windows": len(val_idx),
            "train_subjects": len(train_groups),
            "val_subjects": len(val_groups),
            **{f"val_subject_{CLASS_NAME_BY_ID[i]}": int(np.sum(val_subjects["y"] == i)) for i in range(N_CLASSES)},
        }
    )

fold_sanity = pd.DataFrame(fold_sanity_rows)
display(fold_sanity)
fold_sanity.to_csv(RESULT_DIR / "fold_sanity.csv", index=False)
with (RESULT_DIR / "train_sanity.json").open("w", encoding="utf-8") as file:
    json.dump(
        {
            "train_shape": list(X_train_raw.shape),
            "train_window_labels": class_counts(y_train),
            "train_subject_labels": class_counts(train_subjects["y"].to_numpy()),
            "train_padding_ratio": float((X_train_raw == PADDING_VALUE).mean()),
            "n_cv_folds": N_SPLITS,
            "test_usage": "isolated until final evaluation",
        },
        file,
        indent=2,
        ensure_ascii=False,
    )

# %% [markdown] cell 10
# ## 3. 공용 유틸리티: feature mapping, RF ranking, z-score, subject weight

# %% cell 11
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def feature_metadata(feature_name):
    match = re.match(r"^(.*)_t(\d{3})$", feature_name)
    sensor_group = match.group(1) if match else feature_name
    timestep = int(match.group(2)) if match else None
    modality = "activity" if feature_name.startswith("activity_") else "sleep"
    return {
        "feature": feature_name,
        "sensor_group": sensor_group,
        "modality": modality,
        "is_timeslot_feature": timestep is not None,
        "timeslot_index": timestep,
        "minute_offset": None if timestep is None else timestep * 5,
    }


def enrich_feature_ranking(ranking):
    metadata = pd.DataFrame([feature_metadata(name) for name in ranking["feature"]])
    return pd.concat([ranking.reset_index(drop=True), metadata.drop(columns="feature")], axis=1)


def flatten_day_level(X, y):
    n_samples, n_days, n_features = X.shape
    X_flat = X.reshape(n_samples * n_days, n_features)
    y_flat = np.repeat(y, n_days)
    keep = ~(X_flat == PADDING_VALUE).all(axis=1)
    return X_flat[keep], y_flat[keep]


def fit_rf_feature_ranking(X, y, seed=SEED):
    X_day, y_day = flatten_day_level(X, y)
    X_day = np.where(X_day == PADDING_VALUE, 0.0, X_day)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(X_day, y_day)
    ranking = (
        pd.DataFrame(
            {
                "feature_index": np.arange(len(feature_names), dtype=int),
                "feature": feature_names,
                "rf_importance": rf.feature_importances_,
            }
        )
        .sort_values("rf_importance", ascending=False)
        .reset_index(drop=True)
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1, dtype=int))
    return enrich_feature_ranking(ranking)


def top_feature_indices(ranking, top_k):
    return ranking.head(int(top_k))["feature_index"].to_numpy(dtype=int)


def per_subject_zscore(X, groups):
    X = np.asarray(X, dtype=np.float32).copy()
    groups = np.asarray(groups).astype(str)
    for subject_id in np.unique(groups):
        selected = groups == subject_id
        block = X[selected]
        valid = block != PADDING_VALUE
        for feature_idx in range(X.shape[-1]):
            column = block[..., feature_idx]
            mask = valid[..., feature_idx]
            if mask.sum() < 2:
                normalized = np.zeros_like(column)
            else:
                normalized = (column - column[mask].mean()) / (column[mask].std() + 1e-8)
                normalized[~mask] = 0.0
            block[..., feature_idx] = normalized
        X[selected] = block
    assert np.isfinite(X).all()
    return X


def subject_balanced_sample_weights(y, groups):
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups).astype(str)
    unique_subjects = subject_labels(y, groups)
    classes = np.arange(N_CLASSES)
    class_values = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=unique_subjects["y"].to_numpy(dtype=int),
    )
    class_weight = dict(zip(classes, class_values))
    window_counts = Counter(groups.tolist())
    weights = np.asarray(
        [class_weight[int(label)] / window_counts[subject_id] for label, subject_id in zip(y, groups)],
        dtype=np.float32,
    )
    weights *= len(weights) / weights.sum()
    return weights, {int(key): float(value) for key, value in class_weight.items()}

# %% [markdown] cell 12
# ## 4. 공용 유틸리티: 3-class subject-level metrics와 bootstrap

# %% cell 13
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def safe_binary_auc(y_true, score):
    y_true = np.asarray(y_true, dtype=int)
    return float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) == 2 else np.nan


def safe_binary_auprc(y_true, score):
    y_true = np.asarray(y_true, dtype=int)
    return float(average_precision_score(y_true, score)) if len(np.unique(y_true)) == 2 else np.nan


def multiclass_metrics(y_true, proba):
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    assert proba.shape == (len(y_true), N_CLASSES), proba.shape
    pred = np.argmax(proba, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        pred,
        labels=np.arange(N_CLASSES),
        zero_division=0,
    )
    per_class = {}
    auc_values, auprc_values = [], []
    for class_id, class_name in enumerate(CLASS_NAMES):
        binary_true = (y_true == class_id).astype(int)
        auc = safe_binary_auc(binary_true, proba[:, class_id])
        auprc = safe_binary_auprc(binary_true, proba[:, class_id])
        auc_values.append(auc)
        auprc_values.append(auprc)
        per_class[class_name] = {
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1": float(f1[class_id]),
            "support": int(support[class_id]),
            "ovr_auroc": auc,
            "ovr_auprc": auprc,
        }
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "macro_ovr_auroc": float(np.nanmean(auc_values)),
        "macro_ovr_auprc": float(np.nanmean(auprc_values)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, pred, labels=np.arange(N_CLASSES)).tolist(),
    }


def aggregate_subject_probabilities(y_true, proba, groups):
    frame = pd.DataFrame({"subject_id": np.asarray(groups).astype(str), "y": y_true})
    for class_id in range(N_CLASSES):
        frame[f"p_{class_id}"] = np.asarray(proba)[:, class_id]
    assert frame.groupby("subject_id")["y"].nunique().max() == 1
    return (
        frame.groupby("subject_id", as_index=False)
        .agg(
            y=("y", "first"),
            **{f"p_{class_id}": (f"p_{class_id}", "mean") for class_id in range(N_CLASSES)},
        )
    )


def collapsed_binary_metrics(y_true, proba):
    y_binary = (np.asarray(y_true, dtype=int) != 0).astype(int)
    positive_score = np.asarray(proba, dtype=float)[:, 1:].sum(axis=1)
    pred = (positive_score >= 0.5).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_binary, pred, labels=[0, 1], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_binary, pred)),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "cn_recall": float(recall[0]),
        "mci_or_dementia_recall": float(recall[1]),
        "auroc": safe_binary_auc(y_binary, positive_score),
        "auprc": safe_binary_auprc(y_binary, positive_score),
        "support_cn": int(support[0]),
        "support_mci_or_dementia": int(support[1]),
        "confusion_matrix": confusion_matrix(y_binary, pred, labels=[0, 1]).tolist(),
    }


def subject_level_metrics(y_true, proba, groups):
    subject_df = aggregate_subject_probabilities(y_true, proba, groups)
    subject_proba = subject_df[[f"p_{class_id}" for class_id in range(N_CLASSES)]].to_numpy()
    return (
        multiclass_metrics(subject_df["y"].to_numpy(), subject_proba),
        collapsed_binary_metrics(subject_df["y"].to_numpy(), subject_proba),
        subject_df,
    )


def bootstrap_subject_confidence_intervals(subject_df, n_bootstrap=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    values = {
        "macro_f1": [],
        "balanced_accuracy": [],
        "macro_ovr_auroc": [],
        "macro_ovr_auprc": [],
        "collapsed_macro_f1": [],
    }
    y_values = subject_df["y"].to_numpy(dtype=int)
    proba = subject_df[[f"p_{class_id}" for class_id in range(N_CLASSES)]].to_numpy()
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(subject_df), len(subject_df))
        multi = multiclass_metrics(y_values[idx], proba[idx])
        collapsed = collapsed_binary_metrics(y_values[idx], proba[idx])
        values["macro_f1"].append(multi["macro_f1"])
        values["balanced_accuracy"].append(multi["balanced_accuracy"])
        values["macro_ovr_auroc"].append(multi["macro_ovr_auroc"])
        values["macro_ovr_auprc"].append(multi["macro_ovr_auprc"])
        values["collapsed_macro_f1"].append(collapsed["macro_f1"])
    return {
        key: {
            "lower_2_5": float(np.nanpercentile(items, 2.5)),
            "upper_97_5": float(np.nanpercentile(items, 97.5)),
        }
        for key, items in values.items()
    }

# %% [markdown] cell 14
# ## 5. 모델과 subject-level Macro F1 callback

# %% cell 15
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LSTM,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def build_model(input_shape, hp, seed=SEED):
    set_global_seed(seed)
    inputs = Input(shape=input_shape)
    x = Conv1D(
        int(hp["conv_filters"]),
        3,
        padding="same",
        kernel_regularizer=l2(float(hp["l2"])),
    )(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Bidirectional(
        LSTM(
            int(hp["lstm_units"]),
            return_sequences=True,
            kernel_regularizer=l2(float(hp["l2"])),
            recurrent_regularizer=l2(float(hp["l2"])),
        )
    )(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(
        int(hp["dense_units"]),
        activation="relu",
        kernel_regularizer=l2(float(hp["l2"])),
    )(x)
    x = Dropout(float(hp["dropout"]))(x)
    outputs = Dense(
        N_CLASSES,
        activation="softmax",
        dtype="float32",
        kernel_regularizer=l2(float(hp["l2"])),
    )(x)
    model = Model(inputs, outputs, name="multiclass_datasanity_lstm")
    model.compile(
        optimizer=Adam(float(hp["lr"])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class SubjectMacroF1Monitor(Callback):
    def __init__(self, X_val, y_val, groups_val, patience=10, restore_best_weights=True):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.groups_val = groups_val
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.best = -np.inf
        self.best_epoch = 0
        self.wait = 0
        self.best_weights = None
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        proba = self.model.predict(self.X_val, verbose=0)
        metrics, _, _ = subject_level_metrics(self.y_val, proba, self.groups_val)
        value = float(metrics["macro_f1"])
        logs["val_subject_macro_f1"] = value
        self.history.append({"epoch": epoch + 1, **metrics})
        if value > self.best:
            self.best = value
            self.best_epoch = epoch + 1
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.restore_best_weights and self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def train_one_fold(hp, fold_no, train_idx, val_idx, feature_ranking, model_seed=SEED, verbose=0):
    selected_indices = top_feature_indices(feature_ranking, hp["top_k"])
    X_fold_train = per_subject_zscore(
        X_train_raw[train_idx][:, :, selected_indices], groups_train[train_idx]
    )
    X_fold_val = per_subject_zscore(
        X_train_raw[val_idx][:, :, selected_indices], groups_train[val_idx]
    )
    sample_weight, class_weight = subject_balanced_sample_weights(
        y_train[train_idx], groups_train[train_idx]
    )

    tf.keras.backend.clear_session()
    model = build_model(X_fold_train.shape[1:], hp, seed=model_seed)
    monitor = SubjectMacroF1Monitor(
        X_fold_val, y_train[val_idx], groups_train[val_idx], patience=10
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
    )
    model.fit(
        X_fold_train,
        y_train[train_idx],
        validation_data=(X_fold_val, y_train[val_idx]),
        sample_weight=sample_weight,
        epochs=int(hp["epochs"]),
        batch_size=int(hp["batch_size"]),
        callbacks=[monitor, reduce_lr],
        verbose=verbose,
        shuffle=True,
    )
    proba = model.predict(X_fold_val, verbose=0)
    subject_metrics, collapsed_metrics, _ = subject_level_metrics(
        y_train[val_idx], proba, groups_train[val_idx]
    )
    sequence_metrics = multiclass_metrics(y_train[val_idx], proba)
    result = {
        "fold": int(fold_no),
        "model_seed": int(model_seed),
        "top_k": int(hp["top_k"]),
        "best_epoch": int(monitor.best_epoch),
        "subject_macro_f1": subject_metrics["macro_f1"],
        "subject_balanced_accuracy": subject_metrics["balanced_accuracy"],
        "subject_macro_ovr_auroc": subject_metrics["macro_ovr_auroc"],
        "subject_macro_ovr_auprc": subject_metrics["macro_ovr_auprc"],
        "subject_collapsed_macro_f1": collapsed_metrics["macro_f1"],
        "sequence_macro_f1": sequence_metrics["macro_f1"],
        "class_weight_subject_level": json.dumps(class_weight, sort_keys=True),
    }
    del model, X_fold_train, X_fold_val, proba
    tf.keras.backend.clear_session()
    gc.collect()
    return result

# %% [markdown] cell 16
# ## 6. Fold 내부 RF ranking 캐시
#
# RF ranking은 동일한 fold train subset에 대해 한 번만 계산합니다.
# 이후 각 trial은 `top_k=20/30/40`에 따라 ranking prefix만 사용합니다.

# %% cell 17
fold_feature_rankings = {}
ranking_frames = []
for fold_no, (train_idx, val_idx) in enumerate(CV_SPLITS, start=1):
    print(f"RF ranking fold {fold_no}/{N_SPLITS}")
    ranking = fit_rf_feature_ranking(X_train_raw[train_idx], y_train[train_idx], seed=SEED)
    fold_feature_rankings[fold_no] = ranking
    ranking_frames.append(ranking.assign(fold=fold_no))

fold_rankings_df = pd.concat(ranking_frames, ignore_index=True)
fold_rankings_df.to_csv(RESULT_DIR / "fold_rf_feature_importance.csv", index=False)
display(fold_feature_rankings[1].head(20))

# %% [markdown] cell 18
# ## 7. 5-fold x 8-trial random search

# %% cell 19
HP_SPACE = {
    "conv_filters": [8, 16, 32],
    "lstm_units": [4, 8, 16],
    "dense_units": [4, 8, 16],
    "dropout": [0.3, 0.4, 0.5, 0.6],
    "lr": [1e-3, 5e-4, 3e-4, 1e-4],
    "l2": [1e-4, 1e-3, 5e-3, 1e-2],
    "batch_size": [16, 32],
    "top_k": TOP_K_OPTIONS,
}


def sample_hp(rng):
    hp = {key: values[int(rng.randint(len(values)))] for key, values in HP_SPACE.items()}
    hp["epochs"] = TRIAL_EPOCHS
    return hp


def run_cv_for_hp(hp, model_seed=SEED, verbose=0):
    rows = []
    for fold_no, (train_idx, val_idx) in enumerate(CV_SPLITS, start=1):
        rows.append(
            train_one_fold(
                hp,
                fold_no,
                train_idx,
                val_idx,
                fold_feature_rankings[fold_no],
                model_seed=model_seed,
                verbose=verbose,
            )
        )
    return pd.DataFrame(rows)


rng = np.random.RandomState(SEED)
trial_summaries = []
trial_fold_frames = []
trial_hp_by_id = {}
for trial_id in range(1, N_TRIALS + 1):
    hp = sample_hp(rng)
    trial_hp_by_id[trial_id] = hp
    print(f"\nTrial {trial_id}/{N_TRIALS}: {hp}")
    fold_df = run_cv_for_hp(hp, model_seed=SEED, verbose=0).assign(trial=trial_id)
    trial_fold_frames.append(fold_df)
    summary = {
        "trial": trial_id,
        **hp,
        "cv_subject_macro_f1": fold_df["subject_macro_f1"].mean(),
        "cv_subject_macro_f1_std": fold_df["subject_macro_f1"].std(),
        "cv_subject_balanced_accuracy": fold_df["subject_balanced_accuracy"].mean(),
        "cv_subject_macro_ovr_auroc": fold_df["subject_macro_ovr_auroc"].mean(),
        "cv_subject_macro_ovr_auprc": fold_df["subject_macro_ovr_auprc"].mean(),
        "cv_subject_collapsed_macro_f1": fold_df["subject_collapsed_macro_f1"].mean(),
        "median_best_epoch": int(np.median(fold_df["best_epoch"])),
    }
    trial_summaries.append(summary)
    print(
        "  macro_f1={cv_subject_macro_f1:.4f} | "
        "balanced_acc={cv_subject_balanced_accuracy:.4f} | "
        "macro_ovr_auroc={cv_subject_macro_ovr_auroc:.4f}".format(**summary)
    )

random_search_df = (
    pd.DataFrame(trial_summaries)
    .sort_values(
        [
            "cv_subject_macro_f1",
            "cv_subject_balanced_accuracy",
            "cv_subject_macro_ovr_auroc",
        ],
        ascending=False,
    )
    .reset_index(drop=True)
)
random_search_folds_df = pd.concat(trial_fold_frames, ignore_index=True)
random_search_df.to_csv(RESULT_DIR / "random_search.csv", index=False)
random_search_folds_df.to_csv(RESULT_DIR / "random_search_fold_metrics.csv", index=False)

best_trial_id = int(random_search_df.iloc[0]["trial"])
best_hp = dict(trial_hp_by_id[best_trial_id])
best_cv_metrics = random_search_folds_df[
    random_search_folds_df["trial"] == best_trial_id
].copy()
best_cv_metrics.to_csv(RESULT_DIR / "cv_metrics.csv", index=False)
with (RESULT_DIR / "best_hp.json").open("w", encoding="utf-8") as file:
    json.dump(json_ready(best_hp), file, indent=2)

print("\nBest trial:", best_trial_id)
print(json.dumps(json_ready(best_hp), indent=2))
display(random_search_df)

# %% [markdown] cell 20
# ## 8. Best 설정 seed 안정성 CV
#
# test를 반복 조회하지 않습니다. 선택된 설정을 train-only CV에서 seed `42`, `7`, `2024`로
# 다시 학습하여 변동성을 기록합니다. seed `42` 결과는 random search 결과를 재사용합니다.

# %% cell 21
if RUN_SEED_STABILITY:
    stability_frames = [best_cv_metrics.assign(stability_seed=SEED)]
    for stability_seed in STABILITY_SEEDS:
        if stability_seed == SEED:
            continue
        print(f"\nSeed stability CV: {stability_seed}")
        frame = run_cv_for_hp(best_hp, model_seed=stability_seed, verbose=0)
        stability_frames.append(frame.assign(stability_seed=stability_seed, trial=best_trial_id))
    seed_stability_folds = pd.concat(stability_frames, ignore_index=True)
    seed_stability = (
        seed_stability_folds.groupby("stability_seed", as_index=False)
        .agg(
            subject_macro_f1=("subject_macro_f1", "mean"),
            subject_balanced_accuracy=("subject_balanced_accuracy", "mean"),
            subject_macro_ovr_auroc=("subject_macro_ovr_auroc", "mean"),
            subject_macro_ovr_auprc=("subject_macro_ovr_auprc", "mean"),
        )
    )
    seed_stability_mean = {
        "stability_seed": "mean",
        **{
            column: seed_stability[column].mean()
            for column in seed_stability.columns
            if column != "stability_seed"
        },
    }
    seed_stability_std = {
        "stability_seed": "std",
        **{
            column: seed_stability[column].std()
            for column in seed_stability.columns
            if column != "stability_seed"
        },
    }
    seed_stability = pd.concat(
        [seed_stability, pd.DataFrame([seed_stability_mean, seed_stability_std])],
        ignore_index=True,
    )
    seed_stability_folds.to_csv(RESULT_DIR / "seed_stability_folds.csv", index=False)
    seed_stability.to_csv(RESULT_DIR / "seed_stability.csv", index=False)
    display(seed_stability)
else:
    print("Seed stability skipped: RUN_SEED_STABILITY=False")

# %% [markdown] cell 22
# ## 9. 전체 train RF ranking과 final refit

# %% cell 23
print("Fit final RF ranking on the complete train pool only.")
final_rf_ranking = fit_rf_feature_ranking(X_train_raw, y_train, seed=SEED)
final_rf_ranking.to_csv(RESULT_DIR / "rf_feature_importance_full_train.csv", index=False)

final_top_k = int(best_hp["top_k"])
selected_features_df = final_rf_ranking.head(final_top_k).copy()
selected_features_df.to_csv(RESULT_DIR / "selected_features.csv", index=False)
selected_indices = selected_features_df["feature_index"].to_numpy(dtype=int)
selected_feature_names = selected_features_df["feature"].tolist()
display(selected_features_df)

# Test normalization starts only after every model-selection decision is fixed.
X_train_final = per_subject_zscore(X_train_raw[:, :, selected_indices], groups_train)
X_test_final = per_subject_zscore(X_test_raw[:, :, selected_indices], groups_test)
final_sample_weight, final_class_weight = subject_balanced_sample_weights(y_train, groups_train)
final_epochs = max(1, int(np.median(best_cv_metrics["best_epoch"])))

tf.keras.backend.clear_session()
final_model = build_model((7, final_top_k), best_hp, seed=SEED)
print("Final epochs:", final_epochs)
print("Subject-level class weights:", final_class_weight)
final_history = final_model.fit(
    X_train_final,
    y_train,
    sample_weight=final_sample_weight,
    epochs=final_epochs,
    batch_size=int(best_hp["batch_size"]),
    verbose=1,
    shuffle=True,
)
final_model.save(RESULT_DIR / "final_model.keras")
pd.DataFrame(final_history.history).to_csv(RESULT_DIR / "final_training_history.csv", index=False)

# %% [markdown] cell 24
# ## 10. 격리 test 1회 평가

# %% cell 25
from scipy.stats import ks_2samp

p_train = final_model.predict(X_train_final, verbose=0)
p_test = final_model.predict(X_test_final, verbose=0)
assert p_train.shape == (len(y_train), N_CLASSES)
assert p_test.shape == (len(y_test), N_CLASSES)
assert np.allclose(p_train.sum(axis=1), 1.0, atol=1e-5)
assert np.allclose(p_test.sum(axis=1), 1.0, atol=1e-5)

sequence_test_metrics = multiclass_metrics(y_test, p_test)
subject_test_metrics, subject_collapsed_metrics, subject_predictions = subject_level_metrics(
    y_test, p_test, groups_test
)
sequence_collapsed_metrics = collapsed_binary_metrics(y_test, p_test)
bootstrap_ci = bootstrap_subject_confidence_intervals(
    subject_predictions, n_bootstrap=2000, seed=SEED
)

sequence_predictions = pd.DataFrame(
    {
        "subject_id": groups_test,
        "window_start_date": starts_test,
        "window_end_date": ends_test,
        "true_label": y_test,
        "predicted_label": np.argmax(p_test, axis=1),
        **{f"p_{class_id}_{CLASS_NAMES[class_id]}": p_test[:, class_id] for class_id in range(N_CLASSES)},
    }
)
subject_predictions = subject_predictions.copy()
subject_predictions["predicted_label"] = np.argmax(
    subject_predictions[[f"p_{class_id}" for class_id in range(N_CLASSES)]].to_numpy(),
    axis=1,
)
sequence_predictions.to_csv(RESULT_DIR / "sequence_predictions.csv", index=False)
subject_predictions.to_csv(RESULT_DIR / "subject_predictions.csv", index=False)

ks_rows = []
for local_idx, feature_name in enumerate(selected_feature_names):
    train_values = X_train_raw[:, :, selected_indices[local_idx]].ravel()
    test_values = X_test_raw[:, :, selected_indices[local_idx]].ravel()
    train_values = train_values[train_values != PADDING_VALUE]
    test_values = test_values[test_values != PADDING_VALUE]
    statistic, p_value = ks_2samp(train_values, test_values)
    ks_rows.append(
        {
            "feature": feature_name,
            "ks_stat": float(statistic),
            "p_value": float(p_value),
            "mean_train": float(train_values.mean()),
            "mean_test": float(test_values.mean()),
        }
    )
selected_ks = pd.DataFrame(ks_rows).sort_values("ks_stat", ascending=False)
selected_ks.to_csv(RESULT_DIR / "selected_features_train_test_ks.csv", index=False)

test_report = {
    "official_metric": "subject_level_macro_f1",
    "limitations": [
        "Official test has only 2 Dementia subjects.",
        "Binary DataSanity previously found KS p<0.001 for 821/1487 integrated features.",
        "Subject z-score normalization targets offline subject analysis, not single-window realtime inference.",
    ],
    "best_trial": best_trial_id,
    "best_hp": best_hp,
    "final_epochs": final_epochs,
    "selected_feature_count": final_top_k,
    "sequence_level": sequence_test_metrics,
    "subject_level": subject_test_metrics,
    "sequence_level_cn_vs_impaired": sequence_collapsed_metrics,
    "subject_level_cn_vs_impaired": subject_collapsed_metrics,
    "subject_level_bootstrap_95_ci": bootstrap_ci,
    "test_padding_ratio": float((X_test_raw == PADDING_VALUE).mean()),
}
with (RESULT_DIR / "test_metrics.json").open("w", encoding="utf-8") as file:
    json.dump(json_ready(test_report), file, ensure_ascii=False, indent=2)

print(json.dumps(json_ready(test_report), ensure_ascii=False, indent=2))
display(selected_ks)

# %% [markdown] cell 26
# ## 11. Confusion matrix와 training curve

# %% cell 27
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.heatmap(
    np.asarray(sequence_test_metrics["confusion_matrix"]),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    ax=axes[0],
)
axes[0].set_title("Sequence-level confusion matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")

sns.heatmap(
    np.asarray(subject_test_metrics["confusion_matrix"]),
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    ax=axes[1],
)
axes[1].set_title("Subject-level confusion matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")
plt.tight_layout()
plt.savefig(RESULT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

history_df = pd.DataFrame(final_history.history)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(history_df["loss"], label="train")
axes[0].set_title("Final refit loss")
axes[0].set_xlabel("Epoch")
axes[0].grid(alpha=0.3)
axes[1].plot(history_df["accuracy"], label="train")
axes[1].set_title("Final refit accuracy")
axes[1].set_xlabel("Epoch")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULT_DIR / "training_curve.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown] cell 28
# ## 12. SHAP 담당팀 인계 번들
#
# 이 셀은 SHAP 값을 계산하지 않습니다. 학습된 모델, top-K 입력 배열, feature mapping,
# 예측 확률과 manifest만 저장합니다. 제외된 feature는 중요도 `0`이 아니라 분석 대상 밖입니다.

# %% cell 29
def subject_balanced_background_indices(groups, max_windows=200, seed=SEED):
    groups = np.asarray(groups).astype(str)
    rng = np.random.default_rng(seed)
    subject_ids = np.unique(groups)
    rng.shuffle(subject_ids)
    queues = {}
    for subject_id in subject_ids:
        indices = np.where(groups == subject_id)[0]
        rng.shuffle(indices)
        queues[subject_id] = list(indices)
    selected = []
    while len(selected) < min(max_windows, len(groups)):
        advanced = False
        for subject_id in subject_ids:
            if queues[subject_id]:
                selected.append(queues[subject_id].pop())
                advanced = True
                if len(selected) >= min(max_windows, len(groups)):
                    break
        if not advanced:
            break
    return np.asarray(selected, dtype=int)


background_idx = subject_balanced_background_indices(groups_train, max_windows=200, seed=SEED)
handoff_path = RESULT_DIR / "shap_handoff_arrays.npz"
np.savez_compressed(
    handoff_path,
    background_X=X_train_final[background_idx],
    background_y=y_train[background_idx],
    background_subject_id=groups_train[background_idx],
    X_test_explain=X_test_final,
    y_test=y_test,
    test_subject_id=groups_test,
    test_window_start_date=starts_test,
    test_window_end_date=ends_test,
)

manifest = {
    "task": "multiclass",
    "n_classes": N_CLASSES,
    "class_names": CLASS_NAMES,
    "class_mapping": CLASS_NAME_BY_ID,
    "model_file": "final_model.keras",
    "handoff_arrays_file": "shap_handoff_arrays.npz",
    "input_shape": [7, final_top_k],
    "original_integrated_feature_count": len(feature_names),
    "selected_feature_count": final_top_k,
    "selected_features_file": "selected_features.csv",
    "selected_features": selected_features_df.to_dict(orient="records"),
    "normalization": {
        "method": "per_subject_zscore",
        "padding_input_value": PADDING_VALUE,
        "padding_after_normalization": 0.0,
        "inference_scope": "offline subject analysis with multiple windows",
    },
    "background": {
        "source": "train only",
        "strategy": "subject-balanced round-robin",
        "seed": SEED,
        "max_windows": 200,
        "saved_windows": len(background_idx),
    },
    "shap_scope": (
        "Run SHAP only for selected top-K model inputs. "
        "Unselected features were not evaluated and must not be reported as zero importance."
    ),
    "timeslot_mapping": (
        "For *_tNNN features, timeslot_index is a 5-minute bin and minute_offset=timeslot_index*5."
    ),
}
with (RESULT_DIR / "shap_manifest.json").open("w", encoding="utf-8") as file:
    json.dump(json_ready(manifest), file, ensure_ascii=False, indent=2)

loaded_handoff = np.load(handoff_path, allow_pickle=True)
assert loaded_handoff["background_X"].shape[1:] == (7, final_top_k)
assert loaded_handoff["X_test_explain"].shape == (len(y_test), 7, final_top_k)
assert len(selected_features_df) == final_top_k
assert tuple(final_model.input_shape[1:]) == (7, final_top_k)
assert final_model.output_shape[-1] == N_CLASSES

print("SHAP handoff validation passed.")
print("\nArtifacts:")
for artifact in sorted(RESULT_DIR.iterdir()):
    print(" -", artifact.name)

# %% [markdown] cell 30
# ## 해석 시 주의사항
#
# - 공식 지표는 subject-level Macro F1입니다.
# - test Dementia subject가 2명이므로 Dementia recall은 탐색적 결과입니다.
# - subject별 z-score은 여러 window를 보유한 대상자의 오프라인 분석을 위한 정책입니다.
# - SHAP 담당팀은 `shap_manifest.json`의 top-K mapping과 `shap_handoff_arrays.npz`만 사용합니다.
# - 기존 binary DataSanity에서 `821/1487` feature의 train/test shift가 확인되었습니다.
