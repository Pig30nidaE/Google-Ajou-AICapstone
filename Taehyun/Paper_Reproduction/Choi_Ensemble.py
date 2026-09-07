import os
import sys
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import scipy.stats as st
import optuna

warnings.filterwarnings('ignore')

# ==========================================
# 1. Raw Data Preprocessing (from AI Hub)
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
    original_label_map = {"CN": 0, "MCI": 1, "Dem": 2, "Dementia": 2}
    binary_label_map = {"CN": 0, "MCI": 1, "Dem": 1, "Dementia": 1}
    label_df["original_label"] = label_df["DIAG_NM"].map(original_label_map)
    label_df["label"] = label_df["DIAG_NM"].map(binary_label_map)
    return label_df[["EMAIL", "DIAG_NM", "original_label", "label"]].drop_duplicates("EMAIL")

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
    out["activity_not_worn_ratio"] = out.get("activity_class_0_ratio", np.nan)
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

def process_raw_to_daily_level():
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
    DROP_COLS = ["EMAIL", "original_label", TARGET_COL, "DIAG_NM", "date"]
    features = [c for c in all_daily.columns if c not in DROP_COLS and pd.api.types.is_numeric_dtype(all_daily[c])]
    
    all_daily[features] = all_daily[features].fillna(all_daily[features].median())
    
    X = all_daily[features].values
    y = all_daily[TARGET_COL].astype(int).values
    groups = all_daily["EMAIL"].values
    
    return X, y, groups, features

# ==========================================
# 2. Main Model Execution
# ==========================================
def calc_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return acc, auc, prec, rec, f1, sens, spec

def get_stats(arr):
    arr = np.array(arr)
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0 or len(arr) == 1:
        return mean, std, mean, mean
    ci = st.t.interval(0.95, len(arr)-1, loc=mean, scale=st.sem(arr))
    return mean, std, ci[0], ci[1]

CHOI_PAPER_FEATURES = [
    "activity_high", "activity_steps", "activity_daily_movement", "activity_non_wear", 
    "activity_inactivity_alerts", "activity_score_training_frequency", "activity_score_training_volume", 
    "active_low", "activity_medium", "activity_met_min_inactive", "activity_met_min_low",
    "activity_met_min_medium", "activity_average_met", "activity_cal_active", 
    "activity_score_meet_daily_targets", "activity_score_stay_active", "activity_score",
    "activity_total", "activity_score_recovery_time", "activity_rest",
    "sleep_light", "sleep_deep", "sleep_score_deep", "sleep_awake", "sleep_restless", 
    "sleep_rem", "sleep_score_rem", "sleep_hr_lowest", "sleep_hr_average", "sleep_breath_average",
    "sleep_score_disturbances", "sleep_total", "sleep_score_alignment", "sleep_period_id", 
    "sleep_onset_latency", "sleep_score_latency", "sleep_score", "sleep_midpoint_time", 
    "sleep_midpoint_at_delta", "sleep_score_efficiency", "sleep_duration", "sleep_temperature_deviation",
    # Mapped raw Oura column names if different (activity_active_low -> active_low in text)
    "activity_active_low" 
]

def choi_feature_selection(X_train, y_train, feature_names):
    print("  [Feature Selection] Running Pearson > 0.9 & RF Importance >= Median...")
    df = pd.DataFrame(X_train, columns=feature_names)
    
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    median_imp = np.median(importances)
    
    corr_matrix = df.corr().abs()
    
    to_drop = []
    for i in range(len(feature_names)):
        has_high_corr = False
        for j in range(len(feature_names)):
            if i != j and corr_matrix.iloc[i, j] >= 0.9:
                has_high_corr = True
                break
        
        if has_high_corr and importances[i] < median_imp:
            to_drop.append(i)
            
    keep_indices = [i for i in range(len(feature_names)) if i not in to_drop]
    print(f"  [Feature Selection] Dropped {len(to_drop)} features. Kept {len(keep_indices)} features.")
    return keep_indices

def run_scenario(X, y, groups, feature_names, scenario_name, leakage_allowed, use_nested_cv, n_repeats=5):
    print(f"\n=============================================")
    print(f"Starting Scenario: {scenario_name}")
    print(f"Leakage Allowed: {leakage_allowed}, Nested CV: {use_nested_cv}, Repeats: {n_repeats}")
    print(f"=============================================")
    
    all_acc, all_auc, all_prec, all_rec, all_f1, all_sens, all_spec = [], [], [], [], [], [], []
    all_y_true, all_y_pred = [], []
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    for repeat in range(n_repeats):
        if leakage_allowed:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + repeat)
            cv_split = list(cv.split(X, y))
            X_used = X
        else:
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42 + repeat)
            cv_split = list(cv.split(X, y, groups))
            X_used = X

        for fold, (train_idx, test_idx) in enumerate(cv_split, 1):
            if leakage_allowed:
                # Leakage: Use paper's hardcoded features (already selected on whole dataset)
                keep_indices = [i for i, f in enumerate(feature_names) if f in CHOI_PAPER_FEATURES or f.replace("activity_active_low", "active_low") in CHOI_PAPER_FEATURES]
                if len(keep_indices) == 0: keep_indices = list(range(len(feature_names))) # fallback
                X_tr = X_used[train_idx][:, keep_indices]
                X_te = X_used[test_idx][:, keep_indices]
            else:
                # No Leakage: Select features using ONLY training data dynamically
                X_tr_raw = X_used[train_idx]
                X_te_raw = X_used[test_idx]
                keep_indices = choi_feature_selection(X_tr_raw, y[train_idx], feature_names)
                X_tr = X_tr_raw[:, keep_indices]
                X_te = X_te_raw[:, keep_indices]
                
            y_tr, y_te = y[train_idx], y[test_idx]
            
            if use_nested_cv:
                def objective(trial):
                    max_depth = trial.suggest_int('max_depth', 5, 20)
                    lr = trial.suggest_float('lr', 0.05, 0.2)
                    
                    rf = RandomForestClassifier(n_estimators=50, max_depth=max_depth, class_weight='balanced', random_state=42)
                    gbm = GradientBoostingClassifier(n_estimators=50, max_depth=max_depth//2, learning_rate=lr, random_state=42)
                    svm = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='auto', class_weight='balanced', C=10, probability=True, random_state=42))
                    xgb = XGBClassifier(n_estimators=50, max_depth=max_depth//2, learning_rate=lr, eval_metric='logloss', random_state=42)
                    
                    model = VotingClassifier(estimators=[('RF', rf), ('GBM', gbm), ('SVM', svm), ('XGB', xgb)], voting='soft')
                    
                    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                    inner_aucs = []
                    for inner_tr, inner_val in inner_cv.split(X_tr, y_tr):
                        model.fit(X_tr[inner_tr], y_tr[inner_tr])
                        preds = model.predict_proba(X_tr[inner_val])[:, 1]
                        inner_aucs.append(roc_auc_score(y_tr[inner_val], preds))
                    return np.mean(inner_aucs)
                    
                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=5) # 앙상블은 매우 느리므로 5회로 축소
                best_depth = study.best_params['max_depth']
                best_lr = study.best_params['lr']
                
                rf = RandomForestClassifier(n_estimators=200, max_depth=best_depth, class_weight='balanced', random_state=42)
                gbm = GradientBoostingClassifier(n_estimators=100, max_depth=best_depth//2, learning_rate=best_lr, random_state=42)
                svm = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='auto', class_weight='balanced', C=50, probability=True, random_state=42))
                xgb = XGBClassifier(n_estimators=100, max_depth=best_depth//2, learning_rate=best_lr, eval_metric='logloss', random_state=42)
                
                model = VotingClassifier(estimators=[('RF', rf), ('GBM', gbm), ('SVM', svm), ('XGB', xgb)], voting='soft')
            else:
                rf = RandomForestClassifier(n_estimators=200, max_depth=30, class_weight='balanced', random_state=42)
                gbm = GradientBoostingClassifier(n_estimators=100, max_depth=7, learning_rate=0.2, random_state=42)
                svm = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='auto', class_weight='balanced', C=50, probability=True, random_state=42))
                xgb = XGBClassifier(n_estimators=100, max_depth=7, learning_rate=0.2, eval_metric='logloss', random_state=42)
                model = VotingClassifier(estimators=[('RF', rf), ('GBM', gbm), ('SVM', svm), ('XGB', xgb)], voting='soft')
                
            model.fit(X_tr, y_tr)
            prob = model.predict_proba(X_te)[:, 1]
            pred = np.where(prob >= 0.5, 1, 0)
            
            acc, auc, prec, rec, f1, sens, spec = calc_metrics(y_te, pred, prob)
            all_acc.append(acc)
            all_auc.append(auc)
            all_prec.append(prec)
            all_rec.append(rec)
            all_f1.append(f1)
            all_sens.append(sens)
            all_spec.append(spec)
            
            all_y_true.extend(y_te)
            all_y_pred.extend(pred)
            
            print(f"[{scenario_name}] R{repeat+1}-F{fold} Completed.")
            
    acc_m, acc_s, acc_l, acc_u = get_stats(all_acc)
    auc_m, auc_s, auc_l, auc_u = get_stats(all_auc)
    prec_m, prec_s, prec_l, prec_u = get_stats(all_prec)
    rec_m, rec_s, rec_l, rec_u = get_stats(all_rec)
    f1_m, f1_s, f1_l, f1_u = get_stats(all_f1)
    sens_m, sens_s, sens_l, sens_u = get_stats(all_sens)
    spec_m, spec_s, spec_l, spec_u = get_stats(all_spec)
    
    print(f"\n[{scenario_name}] Final Statistical Results (5 Repeats x 5 Folds):")
    print(f"Accuracy   : {acc_m*100:.2f} ± {acc_s*100:.2f}% (95% CI: {acc_l*100:.2f}-{acc_u*100:.2f})")
    print(f"ROC-AUC    : {auc_m:.4f} ± {auc_s:.4f} (95% CI: {auc_l:.4f}-{auc_u:.4f})")
    print(f"Precision  : {prec_m*100:.2f} ± {prec_s*100:.2f}% (95% CI: {prec_l*100:.2f}-{prec_u*100:.2f})")
    print(f"Recall     : {rec_m*100:.2f} ± {rec_s*100:.2f}% (95% CI: {rec_l*100:.2f}-{rec_u*100:.2f})")
    print(f"F1-Score   : {f1_m*100:.2f} ± {f1_s*100:.2f}% (95% CI: {f1_l*100:.2f}-{f1_u*100:.2f})")
    print(f"Sensitivity: {sens_m*100:.2f} ± {sens_s*100:.2f}% (95% CI: {sens_l*100:.2f}-{sens_u*100:.2f})")
    print(f"Specificity: {spec_m*100:.2f} ± {spec_s*100:.2f}% (95% CI: {spec_l*100:.2f}-{spec_u*100:.2f})")
    
    cm = confusion_matrix(all_y_true, all_y_pred)
    os.makedirs('outputs', exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Choi Ensemble - {scenario_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    filename = f"Choi_Ensemble_{scenario_name.replace(' ', '_')}.png"
    plt.savefig(f'outputs/{filename}', dpi=300)
    plt.close()

def main():
    X, y, groups, feature_names = process_raw_to_daily_level()
    print(f"\nData Loaded. Shape: {X.shape}, Features: {len(feature_names)}")
    
    scenarios = [
        {"name": "Scenario1_Leakage_SingleCV", "leak": True, "nested": False},
        {"name": "Scenario2_Leakage_NestedCV", "leak": True, "nested": True},
        {"name": "Scenario3_NoLeakage_SingleCV", "leak": False, "nested": False},
        {"name": "Scenario4_NoLeakage_NestedCV", "leak": False, "nested": True},
    ]
    
    for sc in scenarios:
        run_scenario(X, y, groups, feature_names, sc["name"], sc["leak"], sc["nested"], n_repeats=5)
        
    print("\nAll scenarios completed. Confusion matrices saved in 'outputs' folder.")

if __name__ == '__main__':
    main()
