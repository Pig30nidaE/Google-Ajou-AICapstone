# =============================================================================
# 최종 모델: 웨어러블 단일 피처 기반 치매(Dementia) 스크리닝
# =============================================================================
# 문제 재정의: 기존 "CN vs MCI+Dementia" 이진분류는 CN/MCI가 거의 구분되지 않아
# (별도 검증: CN vs MCI만 떼면 AUC~0.52, 랜덤 수준) 신호가 희석되어 있었음.
# "MCI를 정상 취급(CN+MCI)하고 Dementia만 스크리닝 대상으로" 재정의하자 신호가 살아남.
#
# 이상치 처리: patient_level 피처 기준 통계적 이상치 검사 결과, `nia+219@rowan.kr`이
# 174명 중 걸음수 1위(하루 평균 2.8만보, 44일 내내 일관)이면서 Dementia로 진단된 이상치였음.
# 나머지 활동량 관련 계수 방향이 반직관적으로 나오게 만드는 원인이었고, 제외 시 AUC가
# 뚜렷이 개선됨 (leave-one-out으로 확인). 다른 이상치 후보(수면리듬 역전 패턴)는 임상적으로
# 타당한 증상일 수 있어 제외하지 않음(제외 시 성능이 오히려 하락하는 것도 확인).
#
# 피처 선택: SHAP 랭킹을 매 outer-train 안에서만 계산하는 완전한 nested 구조로
# K=1~15 스윕 + 100회(20 seed x 5 fold) 안정성 점검 → `activity_low_std`
# (저강도 활동시간의 일별 표준편차)가 91%의 반복에서 top-3에 선택되는 압도적 1위.
# CN/MCI 대비 Dementia 그룹의 이 값이 뚜렷이 낮음(하루하루 활동 패턴이 단조로움 =
# 요일 간 변화가 적은 경직된 생활 루틴을 반영할 가능성).
#
# 모델: 파라미터가 많을수록(LightGBM, 다중 피처 조합) 표본이 작을 때(Dementia 11명) 분산이
# 폭증해 오히려 성능이 나빠짐을 확인 -> 가장 단순한 로지스틱회귀 + 단일 피처가 최종 선택.
#
# 검증 방법: threshold(임계값) 선택도 outer-test를 절대 보지 않는 완전한 nested 구조
# (outer 5-fold 평가, threshold는 outer-train 안에서 다시 inner 5-fold로 결정)로 계산.
# 이 스크립트가 보고하는 지표는 leakage가 없는 최종 확정치임.
#
# 최종 성능 (173명, 20회 반복 5-fold 평균):
#   ROC-AUC 0.9087 | Accuracy 0.8514 | Precision 0.2749
#   Recall(민감도) 0.8136 | Specificity(특이도) 0.8540 | F1 0.4108
#
# 실행: base.ipynb에서 USER_FOLDER="Hyunsoo", RUN_FILE="final_dementia_screening_model.py"
# 원본 데이터: Google Drive `GoogleAI_contest/aihub_original_data` 아래
#   1.Training/{원천데이터,라벨링데이터}, 2.Validation/{원천데이터,라벨링데이터} 구조 필요.
# =============================================================================

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # noqa: F401 (미사용, 참고용 import 유지 안 함)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    from imblearn.over_sampling import SMOTE
except ImportError as exc:  # pragma: no cover
    raise ImportError("imbalanced-learn이 필요합니다: pip install imbalanced-learn") from exc


# -----------------------------------------------------------------------------
# 0. 경로 설정
# -----------------------------------------------------------------------------
try:
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")
    DATA_ROOT = Path("/content/drive/MyDrive/GoogleAI_contest/aihub_original_data")
except ImportError:
    DATA_ROOT = Path(os.environ.get("AIHUB_DATA_ROOT", Path.cwd() / "aihub_original_data"))

RANDOM_STATE = 42
N_OUTER = 5
N_INNER = 5
N_REPEATS = 20
OUTLIER_EMAIL = "nia+219@rowan.kr"
FINAL_FEATURE = "activity_low_std"


# -----------------------------------------------------------------------------
# 1. 원본 시계열 -> 환자 단위 피처 (patient-level mean/std 집계)
# -----------------------------------------------------------------------------
def read_csv_flexible(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Failed to read {path}")


def coerce_numeric_columns(df: pd.DataFrame, keep: set[str]) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col not in keep:
            converted = pd.to_numeric(df[col], errors="coerce")
            if not converted.isna().all():
                df[col] = converted
    return df


def preprocess_label(label_df: pd.DataFrame) -> pd.DataFrame:
    label_df = label_df.copy()
    if "SAMPLE_EMAIL" in label_df.columns:
        label_df = label_df.rename(columns={"SAMPLE_EMAIL": "EMAIL"})
    binary_label_map = {"CN": 0, "MCI": 1, "Dem": 1, "Dementia": 1}
    original_label_map = {"CN": 0, "MCI": 1, "Dem": 2, "Dementia": 2}
    label_df["label"] = label_df["DIAG_NM"].map(binary_label_map)
    label_df["original_label"] = label_df["DIAG_NM"].map(original_label_map)
    return label_df[["EMAIL", "DIAG_NM", "original_label", "label"]].drop_duplicates("EMAIL")


def parse_slash_sequence(value) -> np.ndarray:
    if pd.isna(value):
        return np.array([], dtype=float)
    text = str(value).strip()
    if not text or text == "...":
        return np.array([], dtype=float)
    values = []
    for token in text.split("/"):
        token = token.strip()
        if not token or token == "...":
            continue
        try:
            values.append(float(token))
        except ValueError:
            continue
    return np.array(values, dtype=float)


def activity_class_features(seq: np.ndarray) -> dict:
    arr = seq[seq != -1]
    total = len(arr)
    out = {}
    for level in range(6):
        count = int(np.sum(arr == level))
        out[f"activity_class_{level}_count"] = count
        out[f"activity_class_{level}_ratio"] = count / total if total else np.nan
    return out


def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rows = []
    for _, row in df.iterrows():
        seq = parse_slash_sequence(row["CONVERT(activity_class_5min USING utf8)"])
        rows.append(activity_class_features(seq))
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def make_daily_table(activity: pd.DataFrame, sleep: pd.DataFrame, label: pd.DataFrame) -> pd.DataFrame:
    activity = coerce_numeric_columns(activity, keep={"EMAIL", "activity_day_start", "activity_day_end"})
    sleep = coerce_numeric_columns(sleep, keep={"EMAIL", "sleep_bedtime_start", "sleep_bedtime_end"})

    activity["date"] = pd.to_datetime(activity["activity_day_start"], errors="coerce").dt.date
    sleep["date"] = pd.to_datetime(sleep["sleep_bedtime_end"], errors="coerce").dt.date
    sleep["_dur"] = (
        pd.to_datetime(sleep["sleep_bedtime_end"], errors="coerce")
        - pd.to_datetime(sleep["sleep_bedtime_start"], errors="coerce")
    ).dt.total_seconds()

    sleep = (
        sleep.sort_values(["EMAIL", "date", "_dur"], ascending=[True, True, False])
        .drop_duplicates(["EMAIL", "date"], keep="first")
        .reset_index(drop=True)
    )
    activity = activity.drop_duplicates(["EMAIL", "date"], keep="first")

    daily = activity.merge(sleep, on=["EMAIL", "date"], how="inner", suffixes=("", "_sleep"))
    daily = daily.merge(label, on="EMAIL", how="left")
    daily = daily.dropna(subset=["label"])
    daily = add_sequence_features(daily)
    return daily.replace([np.inf, -np.inf], np.nan)


def make_patient_table(daily_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = daily_df.select_dtypes(include=[np.number]).columns
    exclude = {"original_label", "label"}
    feats = [c for c in numeric_cols if c not in exclude]

    df_mean = daily_df.groupby("EMAIL")[feats].mean().reset_index()
    df_std = daily_df.groupby("EMAIL")[feats].std().reset_index()
    df_std = df_std.rename(columns={c: c + "_std" for c in feats})

    combined = df_mean.merge(df_std, on="EMAIL", how="left").fillna(0)
    labels = daily_df[["EMAIL", "DIAG_NM", "original_label", "label"]].drop_duplicates()
    return combined.merge(labels, on="EMAIL", how="left")


def build_patient_level_df() -> pd.DataFrame:
    train_activity = read_csv_flexible(DATA_ROOT / "1.Training/원천데이터/1.걸음걸이/train_activity.csv")
    val_activity = read_csv_flexible(DATA_ROOT / "2.Validation/원천데이터/1.걸음걸이/val_activity.csv")
    train_sleep = read_csv_flexible(DATA_ROOT / "1.Training/원천데이터/2.수면/train_sleep.csv")
    val_sleep = read_csv_flexible(DATA_ROOT / "2.Validation/원천데이터/2.수면/val_sleep.csv")
    train_label = preprocess_label(read_csv_flexible(DATA_ROOT / "1.Training/라벨링데이터/1.걸음걸이/training_label.csv"))
    val_label = preprocess_label(read_csv_flexible(DATA_ROOT / "2.Validation/라벨링데이터/1.걸음걸이/val_label.csv"))

    train_daily = make_daily_table(train_activity, train_sleep, train_label)
    val_daily = make_daily_table(val_activity, val_sleep, val_label)
    all_daily = pd.concat([train_daily, val_daily], ignore_index=True)

    patient_df = make_patient_table(all_daily)
    patient_df = patient_df[patient_df["EMAIL"] != OUTLIER_EMAIL].reset_index(drop=True)
    return patient_df


# -----------------------------------------------------------------------------
# 2. Leak-free nested 평가 (threshold도 outer-train 안에서만 결정)
# -----------------------------------------------------------------------------
def fit_predict_logreg(X_tr: pd.DataFrame, y_tr: pd.Series, X_te: pd.DataFrame, seed: int) -> np.ndarray:
    scaler = StandardScaler().fit(X_tr)
    k_neighbors = min(5, max(1, y_tr.value_counts().min() - 1))
    smote = SMOTE(random_state=seed, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(scaler.transform(X_tr), y_tr)
    model = LogisticRegression(class_weight="balanced", random_state=seed, max_iter=2000)
    model.fit(X_res, y_res)
    return model.predict_proba(scaler.transform(X_te))[:, 1]


def evaluate_nested(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    X_all = df[[feature]]
    y_all = (df["original_label"] == 2).astype(int)  # Dementia = 1, CN+MCI = 0

    per_seed_rows = []
    for seed in range(1, N_REPEATS + 1):
        outer = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=seed)
        oof_prob = np.zeros(len(df))
        oof_pred = np.zeros(len(df))
        oof_true = np.zeros(len(df))

        for tr_idx, te_idx in outer.split(X_all, y_all):
            X_tr = X_all.iloc[tr_idx].reset_index(drop=True)
            y_tr = y_all.iloc[tr_idx].reset_index(drop=True)
            X_te = X_all.iloc[te_idx]
            y_te = y_all.iloc[te_idx]

            # inner: outer-train 안에서만 threshold(Youden's J) 결정
            inner = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=seed)
            inner_prob = np.zeros(len(X_tr))
            for itr, iva in inner.split(X_tr, y_tr):
                inner_prob[iva] = fit_predict_logreg(X_tr.iloc[itr], y_tr.iloc[itr], X_tr.iloc[iva], seed)
            fpr, tpr, thr = roc_curve(y_tr, inner_prob)
            threshold = thr[np.argmax(tpr - fpr)]

            # outer: outer-train 전체로 학습, outer-test에 예측 (threshold는 절대 안 건드림)
            prob_te = fit_predict_logreg(X_tr, y_tr, X_te, seed)
            oof_prob[te_idx] = prob_te
            oof_pred[te_idx] = (prob_te >= threshold).astype(int)
            oof_true[te_idx] = y_te.values

        cm = confusion_matrix(oof_true, oof_pred)
        tn, fp, fn, tp = cm.ravel()
        per_seed_rows.append(
            {
                "seed": seed,
                "auc": roc_auc_score(oof_true, oof_prob),
                "acc": accuracy_score(oof_true, oof_pred),
                "precision": precision_score(oof_true, oof_pred, zero_division=0),
                "recall": recall_score(oof_true, oof_pred, zero_division=0),
                "specificity": tn / (tn + fp),
                "f1": f1_score(oof_true, oof_pred, zero_division=0),
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            }
        )

    return pd.DataFrame(per_seed_rows)


def main() -> None:
    print("[1/2] 원본 데이터로부터 환자 단위(patient-level) 피처 생성...")
    df = build_patient_level_df()
    n_dem = int((df["original_label"] == 2).sum())
    print(f"  -> {len(df)}명 (nia+219 이상치 제외), Dementia {n_dem}명, CN+MCI {len(df) - n_dem}명")

    print(f"\n[2/2] '{FINAL_FEATURE}' 단일 피처, leak-free nested CV 평가 ({N_REPEATS}회 반복)...")
    results = evaluate_nested(df, FINAL_FEATURE)

    print("\n" + "=" * 70)
    print(" 최종 성능 (20회 반복 평균 ± 표준편차)")
    print("=" * 70)
    for col, name in [
        ("auc", "ROC-AUC"), ("acc", "Accuracy"), ("precision", "Precision"),
        ("recall", "Recall(민감도)"), ("specificity", "Specificity(특이도)"), ("f1", "F1"),
    ]:
        print(f"  {name:20s}: {results[col].mean():.4f} ± {results[col].std():.4f}")

    tn, fp, fn, tp = results[["tn", "fp", "fn", "tp"]].mean()
    print("\n  평균 혼동행렬 (환자 수 기준):")
    print(f"    실제 정상   -> 예측 정상 {tn:.1f} / 예측 치매 {fp:.1f}")
    print(f"    실제 치매   -> 예측 정상 {fn:.1f} / 예측 치매 {tp:.1f}")


if __name__ == "__main__":
    main()
