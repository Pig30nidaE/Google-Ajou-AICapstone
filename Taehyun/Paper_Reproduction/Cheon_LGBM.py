import os
import sys
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier
import shap

warnings.filterwarnings('ignore')

# ==========================================
# 1. Raw Data Preprocessing
# ==========================================
# Taehyun 폴더 바로 아래의 data 폴더 경로 설정 (하위 서브폴더 미사용)
CURRENT_FILE_DIR = Path(__file__).resolve().parent
TAEHYUN_DIR = CURRENT_FILE_DIR.parent
DATA_DIR = TAEHYUN_DIR / "data"

if not DATA_DIR.exists():
    if (Path.cwd().parent / "data").exists():
        DATA_DIR = Path.cwd().parent / "data"
    elif Path(r"c:\ML4\Google-Ajou-AICapstone\Taehyun\data").exists():
        DATA_DIR = Path(r"c:\ML4\Google-Ajou-AICapstone\Taehyun\data")
    else:
        DATA_DIR = Path("./data")

def find_csv(filename: str) -> Path:
    return DATA_DIR / filename

def read_csv_flexible(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str, low_memory=False)
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    return pd.DataFrame()

def coerce_numeric_columns(df: pd.DataFrame, keep: set) -> pd.DataFrame:
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
    if label_df.empty: return label_df
    binary_label_map = {"CN": 0, "MCI": 1, "Dem": 1, "Dementia": 1}
    label_df["label"] = label_df["DIAG_NM"].map(binary_label_map)
    return label_df[["EMAIL", "DIAG_NM", "label"]].drop_duplicates("EMAIL")

def parse_slash_sequence(value, dtype=float) -> np.ndarray:
    if pd.isna(value): return np.array([], dtype=float)
    text = str(value).strip()
    if not text or text == "...": return np.array([], dtype=float)
    values = []
    for token in text.split("/"):
        token = token.strip()
        if not token or token == "...": continue
        try: values.append(dtype(token))
        except ValueError: continue
    return np.array(values, dtype=float)

def numeric_sequence_stats(seq: np.ndarray, prefix: str, remove_zero: bool = False) -> dict:
    arr = np.asarray(seq, dtype=float)
    arr = arr[arr != -1]
    if remove_zero: arr = arr[arr != 0]
    if len(arr) == 0:
        return {f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan, f"{prefix}_var": np.nan, f"{prefix}_min": np.nan, f"{prefix}_max": np.nan, f"{prefix}_median": np.nan, f"{prefix}_q25": np.nan, f"{prefix}_q75": np.nan, f"{prefix}_iqr": np.nan, f"{prefix}_valid_count": 0}
    q25, q75 = np.percentile(arr, 25), np.percentile(arr, 75)
    return {f"{prefix}_mean": float(np.mean(arr)), f"{prefix}_std": float(np.std(arr)), f"{prefix}_var": float(np.var(arr)), f"{prefix}_min": float(np.min(arr)), f"{prefix}_max": float(np.max(arr)), f"{prefix}_median": float(np.median(arr)), f"{prefix}_q25": float(q25), f"{prefix}_q75": float(q75), f"{prefix}_iqr": float(q75 - q25), f"{prefix}_valid_count": int(len(arr))}

def activity_class_features(seq: np.ndarray) -> dict:
    arr = np.asarray(seq, dtype=float)
    arr = arr[arr != -1]
    total = len(arr)
    out = {}
    for level in range(6):
        count = int(np.sum(arr == level))
        out[f"activity_class_{level}_count"] = count
        out[f"activity_class_{level}_ratio"] = count / total if total else np.nan
    out["activity_rest_ratio"] = out.get("activity_class_1_ratio", np.nan)
    out["activity_inactive_ratio"] = out.get("activity_class_2_ratio", np.nan)
    out["activity_active_ratio"] = out.get("activity_class_3_ratio", 0) + out.get("activity_class_4_ratio", 0) + out.get("activity_class_5_ratio", 0) if total else np.nan
    out["activity_class_valid_count"] = total
    return out

def sleep_hypnogram_features(seq: np.ndarray) -> dict:
    arr = np.asarray(seq, dtype=float)
    arr = arr[(arr != -1) & (arr != 0)]
    total = len(arr)
    out = {}
    stage_map = {1: "deep", 2: "light", 3: "rem", 4: "awake"}
    for level, name in stage_map.items():
        count = int(np.sum(arr == level))
        out[f"sleep_{name}_count_5min"] = count
        out[f"sleep_{name}_ratio_5min"] = count / total if total else np.nan
    out["sleep_stage_transition_count"] = int(np.sum(arr[1:] != arr[:-1])) if total > 1 else 0
    out["sleep_hypnogram_valid_count"] = total
    return out

def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rows = []
    for _, row in df.iterrows():
        feats = {}
        if "CONVERT(activity_class_5min USING utf8)" in row:
            feats.update(activity_class_features(parse_slash_sequence(row["CONVERT(activity_class_5min USING utf8)"], dtype=float)))
        if "CONVERT(activity_met_1min USING utf8)" in row:
            feats.update(numeric_sequence_stats(parse_slash_sequence(row["CONVERT(activity_met_1min USING utf8)"], dtype=float), "activity_met_1min", remove_zero=False))
        if "CONVERT(sleep_hr_5min USING utf8)" in row:
            feats.update(numeric_sequence_stats(parse_slash_sequence(row["CONVERT(sleep_hr_5min USING utf8)"], dtype=float), "sleep_hr_5min", remove_zero=True))
        if "CONVERT(sleep_rmssd_5min USING utf8)" in row:
            feats.update(numeric_sequence_stats(parse_slash_sequence(row["CONVERT(sleep_rmssd_5min USING utf8)"], dtype=float), "sleep_rmssd_5min", remove_zero=True))
        if "CONVERT(sleep_hypnogram_5min USING utf8)" in row:
            feats.update(sleep_hypnogram_features(parse_slash_sequence(row["CONVERT(sleep_hypnogram_5min USING utf8)"], dtype=float)))
        rows.append(feats)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

def make_daily_table(activity: pd.DataFrame, sleep: pd.DataFrame, label: pd.DataFrame) -> pd.DataFrame:
    if activity.empty or sleep.empty: return pd.DataFrame()
    activity = coerce_numeric_columns(activity, keep={"EMAIL", "activity_day_start", "activity_day_end"})
    sleep = coerce_numeric_columns(sleep, keep={"EMAIL", "sleep_bedtime_start", "sleep_bedtime_end"})
    
    activity["activity_day_start_dt"] = pd.to_datetime(activity.get("activity_day_start"), errors="coerce")
    activity["activity_day_end_dt"] = pd.to_datetime(activity.get("activity_day_end"), errors="coerce")
    sleep["sleep_bedtime_start_dt"] = pd.to_datetime(sleep.get("sleep_bedtime_start"), errors="coerce")
    sleep["sleep_bedtime_end_dt"] = pd.to_datetime(sleep.get("sleep_bedtime_end"), errors="coerce")
    
    activity["date"] = activity["activity_day_start_dt"].dt.date
    sleep["date"] = sleep["sleep_bedtime_end_dt"].dt.date
    
    sleep_start = sleep["sleep_bedtime_start_dt"]
    sleep_end = sleep["sleep_bedtime_end_dt"]
    sleep["_sleep_duration_seconds"] = (sleep_end - sleep_start).dt.total_seconds()
    sleep = sleep.sort_values(["EMAIL", "date", "_sleep_duration_seconds"], ascending=[True, True, False]).drop_duplicates(["EMAIL", "date"], keep="first").reset_index(drop=True)
    
    daily = activity.merge(sleep, on=["EMAIL", "date"], how="inner", suffixes=("", "_sleep"))
    if not label.empty:
        daily = daily.merge(label, on="EMAIL", how="left")
    
    daily = add_sequence_features(daily)
    
    drop_cols = ["activity_class_5min", "activity_met_1min", "sleep_hr_5min", "sleep_hypnogram_5min", "sleep_rmssd_5min", "CONVERT(activity_class_5min USING utf8)", "CONVERT(activity_met_1min USING utf8)", "CONVERT(sleep_hr_5min USING utf8)", "CONVERT(sleep_hypnogram_5min USING utf8)", "CONVERT(sleep_rmssd_5min USING utf8)", "activity_day_start", "activity_day_end", "activity_day_start_dt", "activity_day_end_dt", "sleep_bedtime_start", "sleep_bedtime_end", "sleep_bedtime_start_dt", "sleep_bedtime_end_dt", "_sleep_duration_seconds", "date", "DIAG_NM"]
    daily = daily.drop(columns=[c for c in drop_cols if c in daily.columns], errors="ignore")
    daily = daily.replace([np.inf, -np.inf], np.nan)
    return daily

def process_data():
    print("Loading raw CSV files from data directory...")
    train_act = read_csv_flexible(find_csv("train_activity.csv"))
    val_act = read_csv_flexible(find_csv("val_activity.csv"))
    train_slp = read_csv_flexible(find_csv("train_sleep.csv"))
    val_slp = read_csv_flexible(find_csv("val_sleep.csv"))
    
    train_lbl_name = "training_label.csv" if (DATA_DIR / "training_label.csv").exists() else "training_label_activity.csv"
    val_lbl_name = "val_label.csv" if (DATA_DIR / "val_label.csv").exists() else "val_label_activity.csv"
    
    train_lbl = preprocess_label(read_csv_flexible(find_csv(train_lbl_name)))
    val_lbl = preprocess_label(read_csv_flexible(find_csv(val_lbl_name)))
    
    print("Processing daily tables...")
    train_daily = make_daily_table(train_act, train_slp, train_lbl)
    val_daily = make_daily_table(val_act, val_slp, val_lbl)
    
    all_daily = pd.concat([train_daily, val_daily], ignore_index=True)
    
    TARGET_COL = "label"
    DROP_COLS = ["EMAIL", TARGET_COL, "DIAG_NM", "date"]
    features = [c for c in all_daily.columns if c not in DROP_COLS and pd.api.types.is_numeric_dtype(all_daily[c])]
    
    # Paper: "결측치 처리가 불가능한 로그 데이터는 제거"
    # To try and hit exactly 7737 / 4446, we drop NaNs instead of median imputation.
    print(f"Data shape before dropping NaNs: {all_daily.shape}")
    all_daily = all_daily.dropna(subset=features)
    print(f"Data shape after dropping NaNs: {all_daily.shape}")
    
    y = all_daily[TARGET_COL].astype(int).values
    X = all_daily[features].values
    
    print(f"Label Distribution -> CN(0): {np.sum(y==0)}, MCI(1): {np.sum(y==1)}")
    
    return X, y, features

# ==========================================
# 2. Evaluation Pipeline (5 Repeats x 5-Fold CV)
# ==========================================
def evaluate_model(model, X, y, n_repeats=5):
    aucs = []
    accs = []
    precs = []
    recs = []
    f1s = []
    
    for repeat in range(n_repeats):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + repeat)
        for tr, va in cv.split(X, y):
            model.fit(X[tr], y[tr])
            preds = model.predict_proba(X[va])[:, 1]
            pred_labels = np.where(preds >= 0.5, 1, 0)
            
            aucs.append(roc_auc_score(y[va], preds))
            accs.append(accuracy_score(y[va], pred_labels))
            precs.append(precision_score(y[va], pred_labels, zero_division=0))
            recs.append(recall_score(y[va], pred_labels, zero_division=0))
            f1s.append(f1_score(y[va], pred_labels, zero_division=0))
            
    def compute_stats(arr):
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        n = len(arr)
        ci = 1.96 * std_val / np.sqrt(n)
        return mean_val, std_val, mean_val - ci, mean_val + ci

    stats = {
        "Accuracy": compute_stats(accs),
        "ROC-AUC": compute_stats(aucs),
        "Precision": compute_stats(precs),
        "Recall": compute_stats(recs),
        "F1-Score": compute_stats(f1s)
    }
    
    for name, (mean_val, std_val, ci_low, ci_high) in stats.items():
        if name == "ROC-AUC":
            print(f"  {name:10s}: {mean_val:.4f} ± {std_val:.4f} (95% CI: {ci_low:.4f}-{ci_high:.4f})")
        else:
            print(f"  {name:10s}: {mean_val*100:.2f} ± {std_val*100:.2f}% (95% CI: {ci_low*100:.2f}-{ci_high*100:.2f})")
            
    return stats

def main():
    print("======================================================")
    print(" Cheon et al. (2025) Exact Reproduction (5 Repeats x 5-Fold)")
    print("======================================================\n")
    
    X, y, feature_names = process_data()
    feature_names = np.array(feature_names)
    print(f"\nTotal Features: {len(feature_names)}")
    
    # ----------------------------------------------------
    # Step 1: Baseline Evaluation (Table 1 reproduction)
    # ----------------------------------------------------
    print("\n[Step 1] Baseline Evaluation (Table 1: All features, Default Hyperparams)")
    baseline_model = LGBMClassifier(random_state=42, verbose=-1)
    evaluate_model(baseline_model, X, y)
    
    # ----------------------------------------------------
    # Step 2: Feature Selection using SHAP (Figure 4 reproduction)
    # ----------------------------------------------------
    print("\n[Step 2] Feature Selection (Figure 4: SHAP Top 40 Features)")
    explainer_model = LGBMClassifier(random_state=42, verbose=-1)
    explainer_model.fit(X, y)
    
    explainer = shap.TreeExplainer(explainer_model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    
    top_40_idx = sorted_idx[:40]
    X_top40 = X[:, top_40_idx]
    
    fs_model = LGBMClassifier(random_state=42, verbose=-1)
    evaluate_model(fs_model, X_top40, y)
    
    # ----------------------------------------------------
    # Step 3: Hyperparameter Tuning (Final Result reproduction)
    # ----------------------------------------------------
    print("\n[Step 3] Hyperparameter Tuning on Top 40 Features (Table 2 & Final Result)")
    tuned_model = LGBMClassifier(
        min_child_samples=41,
        num_leaves=330,
        n_estimators=1000,
        learning_rate=0.08,
        random_state=42,
        verbose=-1
    )
    evaluate_model(tuned_model, X_top40, y)
    
    print("\n======================================================")
    print(" Reproduction Complete.")
    print("======================================================")

if __name__ == "__main__":
    main()
