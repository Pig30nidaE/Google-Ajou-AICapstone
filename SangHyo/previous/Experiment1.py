# Auto-generated Python script converted from a Jupyter notebook.
# Source notebook: SangHyo/previous/Experiment1.ipynb
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
# # 0531 Paper-Aligned Integrated LSTM
#
# `docs/라이프로그 데이터를 활용한 LSTM 모델 기반의 치매 예측.pdf`의 LSTM 설명에 맞춘
# 재현형 학습 노트북입니다.
#
# 논문에서 확인되는 조건:
#
# - MinMax 정규화, 7일 시퀀스, 5분 로그 길이 288, 패딩 `-1`
# - 첫 번째 LSTM 32 units, 두 번째 LSTM 16 units
# - 각 은닉층 뒤 Dropout
# - sigmoid 출력, Adam, binary crossentropy
# - early stopping patience 5
# - discrete + continuous 통합 LSTM의 보고 accuracy: `92.72%`
#
# 논문은 정확한 Dropout 비율과 통합 텐서의 배치를 공개하지 않았습니다. 이 노트북은
# 전처리팀이 제공한 `X_integrated_seq` (`7 x 1487`)를 사용하며 Dropout은 `0.2`로 둡니다.
# official validation은 마지막 평가에서만 사용하고, 조기 종료 epoch와 threshold는 source train
# 내부의 환자 단위 holdout으로 선택합니다.

# %% [markdown] cell 2
# ## 1. Colab A100 Setup

# %% cell 3
import gc
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

import tensorflow as tf
from tensorflow.keras import mixed_precision

tf.random.set_seed(SEED)
gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    raise RuntimeError("GPU가 감지되지 않았습니다. Colab 런타임을 A100 GPU로 설정하세요.")

mixed_precision.set_global_policy("mixed_float16")
print("TensorFlow:", tf.__version__)
print("GPU:", gpus)
print("Mixed precision:", mixed_precision.global_policy().name)

try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception as exc:
    print("Google Drive mount를 건너뜁니다:", exc)

# 데이터 위치가 다르면 이 값만 수정하세요.
DATA_PATH_OVERRIDE = None
DATA_PATH_CANDIDATES = [
    Path("/content/drive/MyDrive/ML_preprocessing/O_0531/LSTM/lstm_dataset.pkl"),
    Path("/content/drive/MyDrive/ML_preprocessing/LSTM_preprocessing/0531/LSTM/lstm_dataset.pkl"),
    Path("/content/drive/MyDrive/TeamProject/training/LSTM/binary/Data/0531/LSTM/lstm_dataset.pkl"),
    Path("/content/lstm_dataset.pkl"),
]


def resolve_data_path():
    candidates = ([Path(DATA_PATH_OVERRIDE)] if DATA_PATH_OVERRIDE else []) + DATA_PATH_CANDIDATES
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "lstm_dataset.pkl을 찾지 못했습니다. DATA_PATH_OVERRIDE에 Google Drive 경로를 지정하세요.\n"
        + "\n".join(str(path) for path in candidates)
    )


DATA_PATH = resolve_data_path()
print("DATA_PATH:", DATA_PATH)


def _numpy_frombuffer_compat(buf, dtype, shape, order, axis_order=None):
    """Load NumPy 2.4 pickles on Colab runtimes whose private helper still takes four args."""
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
    with Path(path).open("rb") as f:
        return CompatibleNumpyUnpickler(f).load()

# %% [markdown] cell 4
# ## 2. Load 0531 Integrated Dataset

# %% cell 5
dataset = load_pickle_compat(DATA_PATH)

required = {
    "X_integrated_seq", "y", "patient_id", "split", "integrated_feature_names", "meta"
}
missing = required - set(dataset)
if missing:
    raise KeyError(f"필수 키가 없습니다: {sorted(missing)}")

X = np.asarray(dataset["X_integrated_seq"], dtype=np.float32)
y = np.asarray(dataset["y"], dtype=np.int64)
groups = np.asarray(dataset["patient_id"]).astype(str)
split = np.asarray(dataset["split"]).astype(str)
feature_names = list(dataset["integrated_feature_names"])
meta = dict(dataset["meta"])
del dataset
gc.collect()

train_mask = split == "train"
holdout_mask = split == "val"
assert X.shape == (len(y), 7, 1487), X.shape
assert X.shape[-1] == len(feature_names)
assert set(np.unique(y)).issubset({0, 1})
assert not (set(groups[train_mask]) & set(groups[holdout_mask]))
assert np.isnan(X).sum() == 0 and np.isinf(X).sum() == 0

X_train, y_train, groups_train = X[train_mask], y[train_mask], groups[train_mask]
X_holdout, y_holdout, groups_holdout = X[holdout_mask], y[holdout_mask], groups[holdout_mask]
del X, y, groups, split
gc.collect()


def split_summary(name, y_values, group_values):
    subject_frame = pd.DataFrame({"patient_id": group_values, "y": y_values}).drop_duplicates("patient_id")
    print(
        f"{name}: windows={len(y_values)}, patients={len(subject_frame)}, "
        f"window_labels={dict(zip(*np.unique(y_values, return_counts=True)))}, "
        f"patient_labels={subject_frame['y'].value_counts().sort_index().to_dict()}"
    )


split_summary("train", y_train, groups_train)
split_summary("official_val_holdout", y_holdout, groups_holdout)
print("input_shape:", X_train.shape[1:])
print("padding_value:", meta["padding_value"])

# %% [markdown] cell 6
# ## 3. Metrics

# %% cell 7
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float):
        return None if np.isnan(obj) else obj
    return obj


def safe_auc(y_true, proba):
    return float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) == 2 else np.nan


def safe_pr_auc(y_true, proba):
    return float(average_precision_score(y_true, proba)) if len(np.unique(y_true)) == 2 else np.nan


def binary_metrics(y_true, proba, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, pred, labels=[0, 1], zero_division=0
    )
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "cn_precision": float(precision[0]),
        "cn_recall": float(recall[0]),
        "cn_f1": float(f1[0]),
        "pos_precision": float(precision[1]),
        "pos_recall": float(recall[1]),
        "pos_f1": float(f1[1]),
        "auc": safe_auc(y_true, proba),
        "pr_auc": safe_pr_auc(y_true, proba),
        "support_cn": int(support[0]),
        "support_pos": int(support[1]),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }


def aggregate_subject_proba(proba, y_true, groups):
    frame = pd.DataFrame({"patient_id": groups, "y": y_true, "proba": proba})
    label_nunique = frame.groupby("patient_id")["y"].nunique()
    assert label_nunique.max() == 1, "한 환자에게 서로 다른 라벨이 있습니다."
    return frame.groupby("patient_id", as_index=False).agg(y=("y", "first"), proba=("proba", "mean"))


def subject_metrics(y_true, proba, groups, threshold=0.5):
    agg = aggregate_subject_proba(proba, y_true, groups)
    result = binary_metrics(agg["y"].to_numpy(), agg["proba"].to_numpy(), threshold)
    result["n_subjects"] = int(len(agg))
    return result, agg


def tune_subject_threshold(y_true, proba, groups):
    rows = []
    for threshold in np.round(np.linspace(0.05, 0.95, 181), 3):
        metrics, _ = subject_metrics(y_true, proba, groups, threshold)
        metrics["selection_score"] = (
            0.35 * metrics["macro_f1"]
            + 0.30 * metrics["balanced_accuracy"]
            + 0.20 * metrics["pos_recall"]
            + 0.15 * (0.0 if np.isnan(metrics["auc"]) else metrics["auc"])
        )
        rows.append(metrics)
    grid = pd.DataFrame(rows).sort_values(
        ["selection_score", "macro_f1", "balanced_accuracy", "pos_recall", "auc"],
        ascending=False,
    )
    return float(grid.iloc[0]["threshold"]), grid


def print_metrics(title, metrics):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print(json.dumps(json_ready(metrics), ensure_ascii=False, indent=2))

# %% [markdown] cell 8
# ## 4. Train Paper-Aligned Architecture

# %% cell 9
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

RESULT_DIR = Path("/content/drive/MyDrive/0531_Result/01_paper_aligned")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 논문은 dropout 비율을 공개하지 않았습니다. 0.2는 재현을 위한 명시적 구현 선택입니다.
DROPOUT = 0.2
BATCH_SIZE = 32
MAX_EPOCHS = 100


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def subject_holdout_indices(y_values, group_values, val_size=0.20, seed=SEED):
    subject_frame = pd.DataFrame({"patient_id": group_values, "y": y_values}).drop_duplicates("patient_id")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr_pos, va_pos = next(splitter.split(subject_frame["patient_id"], subject_frame["y"]))
    train_subjects = set(subject_frame.iloc[tr_pos]["patient_id"])
    val_subjects = set(subject_frame.iloc[va_pos]["patient_id"])
    tr_idx = np.where(np.isin(group_values, list(train_subjects)))[0]
    va_idx = np.where(np.isin(group_values, list(val_subjects)))[0]
    assert not (set(group_values[tr_idx]) & set(group_values[va_idx]))
    return tr_idx, va_idx


def build_paper_lstm(input_shape, seed=SEED):
    set_seed(seed)
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(32, return_sequences=True),
            Dropout(DROPOUT),
            LSTM(16),
            Dropout(DROPOUT),
            Dense(1, activation="sigmoid", dtype="float32"),
        ],
        name="paper_aligned_integrated_lstm",
    )
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# official validation은 마지막 평가 전까지 사용하지 않습니다.
tr_idx, early_val_idx = subject_holdout_indices(y_train, groups_train)
paper_model = build_paper_lstm(X_train.shape[1:])
paper_history = paper_model.fit(
    X_train[tr_idx],
    y_train[tr_idx],
    validation_data=(X_train[early_val_idx], y_train[early_val_idx]),
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            min_delta=1e-4,
            restore_best_weights=True,
        )
    ],
    verbose=1,
)

best_epoch = int(np.argmin(paper_history.history["val_loss"]) + 1)
p_early_val = paper_model.predict(X_train[early_val_idx], batch_size=256, verbose=0).ravel()
tuned_threshold, threshold_grid = tune_subject_threshold(
    y_train[early_val_idx], p_early_val, groups_train[early_val_idx]
)

print("best_epoch:", best_epoch)
print("subject threshold selected from train-only internal validation:", tuned_threshold)
threshold_grid.to_csv(RESULT_DIR / "paper_aligned_internal_threshold_grid.csv", index=False)
pd.DataFrame(paper_history.history).to_csv(RESULT_DIR / "paper_aligned_internal_history.csv", index=False)

# %% [markdown] cell 10
# ## 5. Refit and Evaluate Official Validation

# %% cell 11
# 조기 종료에서 선택한 epoch 수만 사용하여 전체 train source로 다시 학습합니다.
# official validation은 이 모델 선택 과정에 관여하지 않습니다.
tf.keras.backend.clear_session()
final_model = build_paper_lstm(X_train.shape[1:], seed=SEED)
final_history = final_model.fit(
    X_train,
    y_train,
    epochs=best_epoch,
    batch_size=BATCH_SIZE,
    verbose=1,
)

p_holdout = final_model.predict(X_holdout, batch_size=256, verbose=0).ravel()
sequence_metrics_05 = binary_metrics(y_holdout, p_holdout, threshold=0.5)
subject_metrics_05, subject_predictions = subject_metrics(
    y_holdout, p_holdout, groups_holdout, threshold=0.5
)
subject_metrics_tuned, _ = subject_metrics(
    y_holdout, p_holdout, groups_holdout, threshold=tuned_threshold
)

print_metrics("official validation - sequence metrics @ 0.5", sequence_metrics_05)
print_metrics("official validation - subject metrics @ 0.5 (paper comparison primary)", subject_metrics_05)
print_metrics("official validation - subject metrics @ train-only tuned threshold", subject_metrics_tuned)

subject_predictions["pred_05"] = (subject_predictions["proba"] >= 0.5).astype(int)
subject_predictions["pred_tuned"] = (subject_predictions["proba"] >= tuned_threshold).astype(int)
sequence_predictions = pd.DataFrame(
    {
        "patient_id": groups_holdout,
        "y": y_holdout,
        "proba": p_holdout,
        "pred_05": (p_holdout >= 0.5).astype(int),
        "pred_tuned": (p_holdout >= tuned_threshold).astype(int),
    }
)

final_metrics = {
    "method": "paper_aligned_integrated_lstm",
    "paper_reported_accuracy": 0.9272,
    "implementation_choices_not_disclosed_by_paper": {
        "dropout": DROPOUT,
        "integrated_tensor": "X_integrated_seq: 7 days x (1440 continuous flattened + 47 discrete)",
        "evaluation_split": "official source train/val split with patient overlap 0",
    },
    "best_epoch_from_train_only_internal_validation": best_epoch,
    "train_only_tuned_subject_threshold": tuned_threshold,
    "official_validation_sequence_metrics_05": sequence_metrics_05,
    "official_validation_subject_metrics_05": subject_metrics_05,
    "official_validation_subject_metrics_tuned": subject_metrics_tuned,
}

final_model.save(str(RESULT_DIR / "paper_aligned_integrated_lstm.keras"))
pd.DataFrame(final_history.history).to_csv(RESULT_DIR / "paper_aligned_refit_history.csv", index=False)
subject_predictions.to_csv(RESULT_DIR / "paper_aligned_subject_predictions.csv", index=False)
sequence_predictions.to_csv(RESULT_DIR / "paper_aligned_sequence_predictions.csv", index=False)
with (RESULT_DIR / "paper_aligned_final_metrics.json").open("w", encoding="utf-8") as f:
    json.dump(json_ready(final_metrics), f, ensure_ascii=False, indent=2)

print("saved:", RESULT_DIR)
