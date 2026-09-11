#!/usr/bin/env python
"""Markdown tables from results/<experiment>/ artifacts (used verbatim in RESULT_COMPARISON.md / FINAL_REPORT.md).

python report_tables.py [--suffix _smoke] > results/summary_tables.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parent
PAPER = {"baseline_lgbm_default_all72": {"roc_auc": 0.9010, "accuracy": 0.8262, "precision_macro": 0.8276, "recall_macro": 0.7904, "f1_macro": 0.8025},
         "fs_default_selected_k": {"roc_auc": 0.9037}, "fs_default_k40": {"roc_auc": 0.9037},
         "final_tuned_selected_k": {"roc_auc": 0.9492}, "final_tuned_k40": {"roc_auc": 0.9492}}
TABLE1_PAPER = {"LightGBM": (0.8262, 0.9010, 0.8276, 0.7904, 0.8025), "Random forest": (0.8055, 0.8835, 0.8325, 0.7491, 0.7659),
                "Decision tree": (0.7041, 0.6806, 0.6808, 0.6806, 0.6807), "K-Nearest Neighbor": (0.6572, 0.6595, 0.6229, 0.6111, 0.6136),
                "Multi-Layer Perceptron": (0.5953, 0.6348, 0.6188, 0.5634, 0.5142), "Support vector machine": (0.6393, 0.6249, 0.6609, 0.5083, 0.4114),
                "Logistic regression": (0.6457, 0.6067, 0.6113, 0.5331, 0.4830)}
METRICS = ["roc_auc", "accuracy", "precision_macro", "recall_macro", "f1_macro", "recall_pos_sensitivity", "specificity", "balanced_accuracy", "pr_auc"]


def load(name: str, suffix: str):
    d = EXP / "results" / f"{name}{suffix}"
    if not (d / "metrics.json").exists():
        return None
    out = {"metrics": json.loads((d / "metrics.json").read_text()), "dir": d,
           "fold": pd.read_csv(d / "fold_metrics.csv")}
    if (d / "run_manifest.json").exists():
        out["manifest"] = json.loads((d / "run_manifest.json").read_text())
    for f in ("selected_hyperparameters_by_fold.csv", "selected_features.json", "feature_selection_diagnostic.json", "split_summary.csv"):
        if (d / f).exists():
            out[f] = pd.read_csv(d / f) if f.endswith(".csv") else json.loads((d / f).read_text())
    return out


def f4(x, nd=4):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def prim(ex, stage, metric):
    if ex is None or stage not in ex["metrics"]["stages"]:
        return None
    return ex["metrics"]["stages"][stage][metric]["primary_repeat_mean"]


def pooled(ex, stage, metric):
    if ex is None or stage not in ex["metrics"]["stages"]:
        return None
    return ex["metrics"]["stages"][stage][metric]["sensitivity_pooled_folds"]


def cell(p):
    if p is None:
        return "—"
    s = f4(p["mean"])
    if p.get("sd_across_repeats") is not None:
        s += f" ± {f4(p['sd_across_repeats'])} [{f4(p['ci95_low'])}, {f4(p['ci95_high'])}]"
    return s


def seed0(ex, stage, metric):
    p = prim(ex, stage, metric)
    return None if p is None else p["per_repeat"][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    R, G, N = (load(n, a.suffix) for n in ("reproduction", "subject_independent", "nested_cv"))
    L = []
    P = L.append
    P("## T1. Experiment R vs paper (ROC-AUC unless stated; primary = mean of per-repeat fold means ± SD [95 % t-CI]; seed 42 = single paper-style run)\n")
    P("| Stage | Paper | R primary | R seed-42 run | R − paper | \\|diff\\| |")
    P("|---|---|---|---|---|---|")
    for stage, pv in PAPER.items():
        p = prim(R, stage, "roc_auc")
        if p is None:
            continue
        P(f"| `{stage}` | {pv['roc_auc']:.4f} | {cell(p)} | {f4(seed0(R, stage, 'roc_auc'))} | {f4(p['mean'] - pv['roc_auc'])} | {f4(abs(p['mean'] - pv['roc_auc']))} |")
    p = prim(R, "final_tuned_all72", "roc_auc")
    if p:
        P(f"| `final_tuned_all72` | (not reported) | {cell(p)} | {f4(seed0(R, 'final_tuned_all72', 'roc_auc'))} | — | — |")
    if R and "selected_features.json" in R:
        ks = [d["k_selected"] for d in R["selected_features.json"]]
        aucs = [d["cv_auc_at_k"] for d in R["selected_features.json"]]
        P(f"\nForward-selection k* per repeat (paper: 40): {ks}; CV AUC at k*: {[round(x, 4) for x in aucs]}")
    P("\n### T1b. Baseline LightGBM, all five Table-1 metrics (R, all 72 features, default params)\n")
    P("| Metric | Paper | R primary | R − paper |")
    P("|---|---|---|---|")
    for m in ("accuracy", "roc_auc", "precision_macro", "recall_macro", "f1_macro"):
        p = prim(R, "baseline_lgbm_default_all72", m)
        if p:
            P(f"| {m} | {PAPER['baseline_lgbm_default_all72'][m]:.4f} | {cell(p)} | {f4(p['mean'] - PAPER['baseline_lgbm_default_all72'][m])} |")

    P("\n## T2. Table 1 analogue: seven base models, library defaults (ROC-AUC / Accuracy / F1-macro)\n")
    P("| Model | Paper AUC | R AUC | G AUC | Paper Acc | R Acc | G Acc | Paper F1m | R F1m | G F1m |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for model, pv in TABLE1_PAPER.items():
        st = f"table1_{model}"
        r = [prim(R, st, m) for m in ("roc_auc", "accuracy", "f1_macro")]
        g = [prim(G, st, m) for m in ("roc_auc", "accuracy", "f1_macro")]
        P(f"| {model} | {pv[1]:.4f} | {f4(r[0] and r[0]['mean'])} | {f4(g[0] and g[0]['mean'])} | {pv[0]:.4f} | {f4(r[1] and r[1]['mean'])} | {f4(g[1] and g[1]['mean'])} | {pv[4]:.4f} | {f4(r[2] and r[2]['mean'])} | {f4(g[2] and g[2]['mean'])} |")

    P("\n## T3. R → G: same stage, only the split unit (and the scope of the learned SHAP ranking) changes\n")
    P("| Stage | R (record-level) | G (subject-level) | G − R | R pooled-OOF AUC | G pooled-OOF AUC |")
    P("|---|---|---|---|---|---|")
    for stage in ("baseline_lgbm_default_all72", "fs_default_k40", "final_tuned_k40", "final_tuned_all72"):
        r, g = prim(R, stage, "roc_auc"), prim(G, stage, "roc_auc")
        if r is None or g is None:
            continue
        ro = R["metrics"]["stages"][stage]["pooled_oof_auc_per_repeat"]["mean"]
        go = G["metrics"]["stages"][stage]["pooled_oof_auc_per_repeat"]["mean"]
        P(f"| `{stage}` | {cell(r)} | {cell(g)} | {f4(g['mean'] - r['mean'])} | {f4(ro)} | {f4(go)} |")
    if G and "feature_selection_diagnostic.json" in G:
        P(f"\nG diagnostic forward curve (fold-local ranking, scored on the test fold, NOT used): best k per repeat = {[d['diagnostic_best_k_NOT_USED'] for d in G['feature_selection_diagnostic.json']]}, AUC = {[round(d['diagnostic_auc'], 4) for d in G['feature_selection_diagnostic.json']]}; k actually used = {G['feature_selection_diagnostic.json'][0]['k_used']}")

    P("\n## T4. G → N: selection moved inside the outer training subjects\n")
    P("| Condition | ROC-AUC primary | pooled-OOF AUC | Accuracy | F1-macro | Balanced acc. |")
    P("|---|---|---|---|---|---|")
    rows = [("G `final_tuned_k40` (k = 40 fixed, Table-2 params)", G, "final_tuned_k40"),
            ("N-A `nested_selected_k_paper_params` (k inner-selected, Table-2 params)", N, "nested_selected_k_paper_params"),
            ("N `nested_selected_k_selected_params` (k and params inner-selected) **primary**", N, "nested_selected_k_selected_params"),
            ("N `nested_selected_k_default_params` (k inner-selected, library defaults)", N, "nested_selected_k_default_params"),
            ("G `baseline_lgbm_default_all72` (no selection, defaults)", G, "baseline_lgbm_default_all72")]
    for label, ex, st in rows:
        p = prim(ex, st, "roc_auc")
        if p is None:
            continue
        oo = ex["metrics"]["stages"][st]["pooled_oof_auc_per_repeat"]["mean"]
        P(f"| {label} | {cell(p)} | {f4(oo)} | {f4(prim(ex, st, 'accuracy')['mean'])} | {f4(prim(ex, st, 'f1_macro')['mean'])} | {f4(prim(ex, st, 'balanced_accuracy')['mean'])} |")
    if N and "selected_hyperparameters_by_fold.csv" in N:
        hp = N["selected_hyperparameters_by_fold.csv"]
        P(f"\nN inner-selected k per outer fold (repeat×fold = {len(hp)}): min {hp.k_selected.min()}, median {hp.k_selected.median():.0f}, max {hp.k_selected.max()}; values = {hp.k_selected.tolist()}")
        P(f"\nN inner-selected hyperparameters (counts over {len(hp)} outer folds):\n")
        P("| selected_params | n | mean inner AUC | mean outer AUC (selected) |")
        P("|---|---|---|---|")
        outer = N["fold"][N["fold"]["stage"] == "nested_selected_k_selected_params"][["repeat", "fold", "roc_auc"]].rename(columns={"fold": "outer_fold"})
        hp2 = hp.merge(outer, on=["repeat", "outer_fold"], how="left")
        for params, g in hp2.groupby("selected_params"):
            P(f"| `{params}` | {len(g)} | {g.inner_mean_auc_at_selected_params.mean():.4f} | {g.roc_auc.mean():.4f} |")
        P(f"\nInner-CV mean AUC at k* (selection optimism check): mean over outer folds = {hp.inner_mean_auc_at_k.mean():.4f}; Table-2 params inner AUC mean = {hp.inner_mean_auc_paper_params.mean():.4f}; default inner AUC mean = {hp.inner_mean_auc_default_params.mean():.4f}")

    P("\n## T5. All metrics, all stages (primary mean ± SD across repeats)\n")
    P("| Experiment | Stage | " + " | ".join(METRICS) + " |")
    P("|---|---|" + "---|" * len(METRICS))
    for name, ex in (("R", R), ("G", G), ("N", N)):
        if ex is None:
            continue
        for stage in ex["metrics"]["stages"]:
            if stage.startswith("table1_"):
                continue
            vals = []
            for m in METRICS:
                p = prim(ex, stage, m)
                vals.append(f"{f4(p['mean'])}" + (f" ± {f4(p['sd_across_repeats'])}" if p.get("sd_across_repeats") is not None else ""))
            P(f"| {name} | `{stage}` | " + " | ".join(vals) + " |")

    P("\n## T6. Sensitivity: pooled fold scores (n = repeats × folds) vs primary repeat-level statistic (ROC-AUC)\n")
    P("| Experiment | Stage | primary mean ± SD (repeats) | pooled mean ± SD (folds) | n fold scores |")
    P("|---|---|---|---|---|")
    for name, ex in (("R", R), ("G", G), ("N", N)):
        if ex is None:
            continue
        for stage in ex["metrics"]["stages"]:
            if stage.startswith("table1_"):
                continue
            p, q = prim(ex, stage, "roc_auc"), pooled(ex, stage, "roc_auc")
            P(f"| {name} | `{stage}` | {f4(p['mean'])} ± {f4(p.get('sd_across_repeats'))} | {f4(q['mean'])} ± {f4(q.get('sd'))} | {q['n_fold_scores']} |")

    P("\n## T7. Splits and runtime\n")
    P("| Experiment | seeds | folds | subject overlap (max over folds) | test subjects per fold | runtime (wall) | processes |")
    P("|---|---|---|---|---|---|---|")
    for name, ex in (("R", R), ("G", G), ("N", N)):
        if ex is None:
            continue
        man = ex.get("manifest", {})
        fd = ex["fold"]
        ov = int(fd["subject_overlap"].max())
        ts = fd.groupby(["repeat", "fold"])["n_test_subjects"].first()
        rt = man.get("runtime_seconds")
        P(f"| {name} | {man.get('seeds')} | {man.get('folds_run_per_repeat')} | {ov} | {int(ts.min())}–{int(ts.max())} | {f4(rt / 60 if rt else None, 1)} min | {len(man.get('per_seed_runs', [])) or 1} |")
    print("\n".join(L))


if __name__ == "__main__":
    main()
