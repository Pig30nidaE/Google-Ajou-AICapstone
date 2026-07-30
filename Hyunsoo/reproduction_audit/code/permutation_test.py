"""
CN+MCI vs Dementia (서캐디안 피처 5개) leak-free nested CV 결과가 우연이 아닌지
label permutation test로 검증한다.

방법: dementia_label을 무작위로 섞고(피처/CV fold 구조는 그대로 고정), 동일한
파이프라인(SHAP 랭킹 -> inner CV forward selection -> 5개 모델 학습 -> outer-test 예측)을
그대로 재실행해서 OOF AUC를 얻는다. 이를 N번 반복해 귀무분포(null distribution)를 만들고,
관측된 AUC가 그 분포에서 몇 %ile인지로 p-value를 계산한다.

CV fold split(random_state=42)은 모든 permutation에서 고정 -> "라벨-피처 관계가 진짜인가"만
분리해서 검증 (fold 분할 자체의 변동성은 이 테스트의 관심사가 아님).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import run_dementia_vs_rest_nested as base  # noqa: E402

RANDOM_STATE = base.RANDOM_STATE
N_OUTER = base.N_OUTER
CANDIDATE_FEATURES = base.CANDIDATE_FEATURES
MODEL_NAMES = ["LightGBM", "CatBoost", "XGBoost", "RandomForest", "LogisticRegression", "Ensemble"]

N_PERM = 1000
OUT_PARTIAL = Path(__file__).parent / "permutation_null_partial.json"
OUT_FINAL = Path(__file__).parent / "permutation_test_results.json"


def run_once(X_all, y_all) -> dict:
    outer_skf = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = {m: np.zeros(len(y_all)) for m in MODEL_NAMES}

    for tr_idx, te_idx in outer_skf.split(X_all, y_all):
        X_tr = X_all.iloc[tr_idx].reset_index(drop=True)
        X_te = X_all.iloc[te_idx].reset_index(drop=True)
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]

        sel_feats, _, _, _ = base.inner_select_features(X_tr, y_tr)
        models = base.train_models(X_tr, y_tr, sel_feats)
        probs = base.predict_all(models, X_te, sel_feats)
        for m in MODEL_NAMES:
            oof_preds[m][te_idx] = probs[m]

    aucs = {}
    for m in MODEL_NAMES:
        try:
            aucs[m] = float(roc_auc_score(y_all, oof_preds[m]))
        except ValueError:
            aucs[m] = float("nan")
    return aucs


def main():
    df = base.load_data()
    y_true = df["dementia_label"].astype(int).values
    X_all = df[CANDIDATE_FEATURES]

    print("[관측값] 실제 라벨로 파이프라인 재실행 (재현성 확인용)...")
    observed = run_once(X_all, y_true)
    print("Observed AUCs:", {k: round(v, 4) for k, v in observed.items()})

    rng = np.random.default_rng(12345)
    null_aucs = {m: [] for m in MODEL_NAMES}

    print(f"\n[Permutation Test] N={N_PERM}회 라벨 셔플 후 동일 파이프라인 재실행...")
    t0 = time.time()
    for i in range(N_PERM):
        y_perm = rng.permutation(y_true)
        aucs = run_once(X_all, y_perm)
        for m in MODEL_NAMES:
            null_aucs[m].append(aucs[m])

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * N_PERM
            ens_arr = np.array([v for v in null_aucs["Ensemble"] if not np.isnan(v)])
            p_ens = (1 + np.sum(ens_arr >= observed["Ensemble"])) / (len(ens_arr) + 1) if len(ens_arr) else float("nan")
            print(f"  [{i+1:4d}/{N_PERM}] elapsed={elapsed:6.1f}s ETA_total={eta:6.1f}s | "
                  f"Ensemble null mean={np.mean(ens_arr):.4f} current_p={p_ens:.4f}")
            OUT_PARTIAL.write_text(json.dumps(
                {"n_done": i + 1, "observed": observed, "null_aucs": null_aucs}, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print(" Permutation Test 최종 결과")
    print("=" * 80)
    results = {}
    for m in MODEL_NAMES:
        arr = np.array([v for v in null_aucs[m] if not np.isnan(v)])
        obs = observed[m]
        p = (1 + np.sum(arr >= obs)) / (len(arr) + 1)
        results[m] = {
            "observed_auc": obs,
            "null_mean": float(np.mean(arr)),
            "null_std": float(np.std(arr)),
            "null_p95": float(np.percentile(arr, 95)),
            "null_p99": float(np.percentile(arr, 99)),
            "p_value": float(p),
            "n_perm": int(len(arr)),
        }
        r = results[m]
        print(f"[{m:20s}] observed={r['observed_auc']:.4f}  null={r['null_mean']:.4f}±{r['null_std']:.4f} "
              f"(95%ile={r['null_p95']:.4f}, 99%ile={r['null_p99']:.4f})  p-value={r['p_value']:.4f}")

    OUT_FINAL.write_text(json.dumps({"observed": observed, "results": results, "n_perm": N_PERM},
                                     indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {OUT_FINAL}")


if __name__ == "__main__":
    main()
