"""One shared pipeline for the three experiments (R / G / N). See PROTOCOL_COMPARISON.md."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from .data import cohort_summary, data_fingerprint, load_daily_records
from .evaluation import ALL_METRICS, aggregate, compute_metrics, pooled_oof_auc
from .features import PAPER_FEATURES, build_features, feature_matrix
from .manifest import build_manifest, environment_text, utc_now, write_json
from .model import (LGBM_DEFAULT, PAPER_TABLE2, make_base_models, make_lgbm, paper_anchored_grid, params_key,
                    score_and_predict)
from .selection import best_k, forward_path, rank_features
from .splits import assert_subject_disjoint, record_folds, record_overlap_subjects, subject_folds

SMOKE_MAX_K = 6


class Log:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        line = f"[{utc_now()}] {msg}"
        print(line, flush=True)
        self._f.write(line + "\n")
        self._f.flush()


class Experiment:
    def __init__(self, cfg: dict, cfg_text: str, exp_dir: Path, repo_root: Path, data_root: Path, smoke: bool = False):
        self.cfg, self.cfg_text = cfg, cfg_text
        self.exp_dir, self.repo_root, self.data_root, self.smoke = exp_dir, repo_root, data_root, smoke
        out = exp_dir / cfg["output_dir"]
        self.out = out.with_name(out.name + "_smoke") if smoke else out
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = Log(self.out / "stdout.log")
        self.name = cfg["experiment"]
        self.seeds = [int(cfg["seeds"][0])] if smoke else [int(s) for s in cfg["seeds"]]
        self.n_splits = int(cfg["n_splits"])
        self.n_threads = int(cfg.get("n_threads", 8))
        self.max_k = SMOKE_MAX_K if smoke else int(cfg["feature_selection"]["max_k"])
        self.folds_to_run = 1 if smoke else None
        self.inner_folds_to_run = 1 if smoke else None
        self.fold_rows, self.oof_rows, self.split_rows = [], [], []
        self.timing: dict[str, list[float]] = {}
        self.name_to_col = {n: i for i, n in enumerate(PAPER_FEATURES)}
        self.all_cols = list(range(len(PAPER_FEATURES)))

    # ------------------------------------------------------------------ data
    def load(self) -> None:
        t0 = time.time()
        raw = load_daily_records(self.data_root)
        self.data_fp = data_fingerprint(self.data_root)
        feat = build_features(raw, sleep_time_definition=self.cfg.get("sleep_time_definition", "clock_difference"))
        self.feat = feat
        self.X = feature_matrix(feat)
        self.y = feat["y"].to_numpy(dtype=int)
        self.groups = feat["subject_id"].to_numpy()
        self.record_id = feat["record_id"].to_numpy()
        n_nan = int(np.isnan(self.X).sum())
        summ = cohort_summary(feat)
        summ["n_features"] = int(self.X.shape[1])
        summ["nan_cells_in_feature_matrix"] = n_nan
        write_json(self.out / "cohort_summary.json", summ)
        fs = pd.DataFrame({"feature": PAPER_FEATURES, "mean": np.nanmean(self.X, 0), "sd": np.nanstd(self.X, 0, ddof=1),
                           "min": np.nanmin(self.X, 0), "max": np.nanmax(self.X, 0), "n_nan": np.isnan(self.X).sum(0)})
        fs.to_csv(self.out / "feature_summary.csv", index=False)
        self.log(f"[{self.name}] data loaded: X={self.X.shape}, subjects={summ['n_subjects']}, records={summ['n_records']}, "
                 f"y1={int(self.y.sum())}, NaN cells={n_nan} ({time.time() - t0:.1f}s)" + ("  [SMOKE]" if self.smoke else ""))

    # ------------------------------------------------------------------ folds
    def folds_for(self, seed: int):
        if self.cfg["split_unit"] == "record":
            folds = record_folds(self.y, self.n_splits, seed, shuffle=bool(self.cfg.get("shuffle", True)))
        else:
            folds = subject_folds(self.y, self.groups, self.n_splits, seed)
        return folds[: self.folds_to_run] if self.folds_to_run else folds

    def split_info(self, tr, te) -> dict:
        if self.cfg["split_unit"] == "subject":
            info = assert_subject_disjoint(self.groups, tr, te)
        else:
            info = {"n_train_subjects": len(set(self.groups[tr])), "n_test_subjects": len(set(self.groups[te])),
                    "subject_overlap": record_overlap_subjects(self.groups, tr, te),
                    "n_train_records": int(len(tr)), "n_test_records": int(len(te))}
        return info

    def record_split(self, repeat, fold, tr, te, level="outer", outer_fold=None) -> dict:
        info = self.split_info(tr, te)
        row = {"experiment": self.name, "repeat": repeat, "seed": self.seeds[repeat], "level": level, "outer_fold": outer_fold, "fold": fold, **info,
               "n_test_pos_records": int(self.y[te].sum())}
        self.split_rows.append(row)
        return info

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, stage: str, repeat: int, folds, model_factory, cols, fold_ids=None, extra: dict | None = None):
        per_fold_cols = cols if (len(cols) > 0 and isinstance(cols[0], (list, tuple, np.ndarray))) else [cols] * len(folds)
        fold_ids = list(range(len(folds))) if fold_ids is None else fold_ids
        for f, (tr, te) in zip(fold_ids, folds):
            c = list(per_fold_cols[fold_ids.index(f)])
            t0 = time.time()
            m = model_factory(f).fit(self.X[np.ix_(tr, c)], self.y[tr])
            fit_s = time.time() - t0
            score, pred = score_and_predict(m, self.X[np.ix_(te, c)])
            met = compute_metrics(self.y[te], score, pred)
            info = self.split_info(tr, te)
            row = {"experiment": self.name, "stage": stage, "repeat": repeat, "seed": self.seeds[repeat], "fold": f, "n_features": len(c),
                   "fit_seconds": round(fit_s, 3), **info, **met, **(extra or {})}
            self.fold_rows.append(row)
            self.oof_rows.append(pd.DataFrame({"experiment": self.name, "stage": stage, "repeat": repeat, "seed": self.seeds[repeat], "fold": f,
                                               "record_id": self.record_id[te], "subject_id": self.groups[te],
                                               "y": self.y[te], "score": score, "pred": pred}))
            self.timing.setdefault(stage, []).append(fit_s)
            self.log(f"  [{stage}] repeat {repeat} fold {f}: AUC={met['roc_auc']:.4f} acc={met['accuracy']:.4f} "
                     f"f1m={met['f1_macro']:.4f} | {len(c)} feats, fit {fit_s:.1f}s, test n={met['n_test']} "
                     f"subj={info['n_test_subjects']} overlap={info['subject_overlap']}")

    def timed(self, key: str, seconds: float) -> None:
        self.timing.setdefault(key, []).append(seconds)

    def top_cols(self, ranking: list[str], k: int) -> list[int]:
        return [self.name_to_col[n] for n in ranking[:k]]

    # ------------------------------------------------------------------ R
    def run_reproduction(self) -> None:
        fs_rows, imp_rows, selected = [], [], []
        base_cfg = self.cfg.get("base_models", {})
        for r, seed in enumerate(self.seeds):
            self.log(f"=== R repeat {r} (seed {seed}) ===")
            folds = self.folds_for(seed)
            for f, (tr, te) in enumerate(folds):
                self.record_split(r, f, tr, te)
            if base_cfg.get("enabled", True) and r < int(base_cfg.get("repeats", 1)):
                for name, model in make_base_models(seed, self.n_threads).items():
                    self.evaluate(f"table1_{name}", r, folds, lambda f, m=model: clone(m), self.all_cols)
            self.evaluate("baseline_lgbm_default_all72", r, folds, lambda f: make_lgbm(LGBM_DEFAULT, seed, self.n_threads), self.all_cols)

            t0 = time.time()
            rank = rank_features(self.X, self.y, PAPER_FEATURES, LGBM_DEFAULT, seed, self.n_threads)  # dataset-wide (A15)
            self.timed("shap_ranking_all_records", time.time() - t0)
            ranking = rank["feature"].tolist()
            imp_rows.append(rank.assign(repeat=r, seed=seed))
            self.log(f"  SHAP ranking (all records, {time.time() - t0:.1f}s); top-5: {ranking[:5]}")
            path = forward_path(self.X, self.y, folds, ranking, PAPER_FEATURES, LGBM_DEFAULT, seed, self.n_threads,
                                self.max_k, log=self.log)
            fs_rows.append(path.assign(repeat=r, seed=seed))
            k_star, auc_star = best_k(path)
            selected.append({"repeat": r, "seed": seed, "k_selected": k_star, "cv_auc_at_k": auc_star,
                             "selected_features": ranking[:k_star]})
            self.log(f"  forward selection: k*={k_star} (fold-mean AUC {auc_star:.4f})")
            tuned = lambda f: make_lgbm(PAPER_TABLE2, seed, self.n_threads)
            dflt = lambda f: make_lgbm(LGBM_DEFAULT, seed, self.n_threads)
            self.evaluate("fs_default_selected_k", r, folds, dflt, self.top_cols(ranking, k_star), extra={"k": k_star})
            self.evaluate("final_tuned_selected_k", r, folds, tuned, self.top_cols(ranking, k_star), extra={"k": k_star})
            kp = self.cfg["final_model"].get("also_evaluate_paper_k")
            if kp and kp <= self.max_k:
                self.evaluate("fs_default_k40", r, folds, dflt, self.top_cols(ranking, kp), extra={"k": kp})
                self.evaluate("final_tuned_k40", r, folds, tuned, self.top_cols(ranking, kp), extra={"k": kp})
            self.evaluate("final_tuned_all72", r, folds, tuned, self.all_cols)
        pd.concat(fs_rows).to_csv(self.out / "feature_selection.csv", index=False)
        pd.concat(imp_rows).to_csv(self.out / "shap_importance.csv", index=False)
        write_json(self.out / "selected_features.json", selected)
        self.finish()

    # ------------------------------------------------------------------ G
    def run_subject_independent(self) -> None:
        fs_rows, rank_rows, diag = [], [], []
        k_fixed = int(self.cfg["feature_selection"]["k_fixed"])
        k_use = min(k_fixed, self.max_k)
        base_cfg = self.cfg.get("base_models", {})
        for r, seed in enumerate(self.seeds):
            self.log(f"=== G repeat {r} (seed {seed}) ===")
            folds = self.folds_for(seed)
            for f, (tr, te) in enumerate(folds):
                self.record_split(r, f, tr, te)
            if base_cfg.get("enabled", True) and r < int(base_cfg.get("repeats", 1)):
                for name, model in make_base_models(seed, self.n_threads).items():
                    self.evaluate(f"table1_{name}", r, folds, lambda f, m=model: clone(m), self.all_cols)
            self.evaluate("baseline_lgbm_default_all72", r, folds, lambda f: make_lgbm(LGBM_DEFAULT, seed, self.n_threads), self.all_cols)

            rankings = []
            for f, (tr, te) in enumerate(folds):
                t0 = time.time()
                rank = rank_features(self.X[tr], self.y[tr], PAPER_FEATURES, LGBM_DEFAULT, seed, self.n_threads)  # train fold only (A17)
                self.timed("shap_ranking_train_fold", time.time() - t0)
                rankings.append(rank["feature"].tolist())
                rank_rows.append(rank.assign(repeat=r, seed=seed, fold=f, selected_top_k=rank["rank"] <= k_use))
                self.log(f"  fold {f}: train-fold SHAP ranking ({time.time() - t0:.1f}s); top-5: {rankings[-1][:5]}")
            path = forward_path(self.X, self.y, folds, rankings, PAPER_FEATURES, LGBM_DEFAULT, seed, self.n_threads,
                                self.max_k, log=self.log)
            fs_rows.append(path.assign(repeat=r, seed=seed))
            kd, aucd = best_k(path)
            diag.append({"repeat": r, "seed": seed, "diagnostic_best_k_NOT_USED": kd, "diagnostic_auc": aucd, "k_used": k_use})
            self.log(f"  diagnostic curve best k={kd} (AUC {aucd:.4f}) -- NOT used; k fixed at {k_use} (paper)")
            tuned = lambda f: make_lgbm(PAPER_TABLE2, seed, self.n_threads)
            dflt = lambda f: make_lgbm(LGBM_DEFAULT, seed, self.n_threads)
            topk = [self.top_cols(rk, k_use) for rk in rankings]
            self.evaluate("fs_default_k40", r, folds, dflt, topk, extra={"k": k_use})
            self.evaluate("final_tuned_k40", r, folds, tuned, topk, extra={"k": k_use})
            if self.cfg["final_model"].get("also_evaluate_all_features", True):
                self.evaluate("final_tuned_all72", r, folds, tuned, self.all_cols)
        pd.concat(fs_rows).to_csv(self.out / "feature_selection.csv", index=False)
        pd.concat(rank_rows).to_csv(self.out / "selected_features_by_fold.csv", index=False)
        write_json(self.out / "feature_selection_diagnostic.json", diag)
        self.finish()

    # ------------------------------------------------------------------ N
    def run_nested(self) -> None:
        grid = paper_anchored_grid()
        inner_n = int(self.cfg["inner_splits"])
        inner_rows, feat_rows, hp_rows, leak = [], [], [], []
        for r, seed in enumerate(self.seeds):
            self.log(f"=== N repeat {r} (seed {seed}) ===")
            outer = self.folds_for(seed)
            for f, (otr, ote) in enumerate(outer):
                self.record_split(r, f, otr, ote, level="outer")
                test_subjects = set(self.groups[ote].tolist())
                inner_rel = subject_folds(self.y[otr], self.groups[otr], inner_n, seed * 1000 + f + 1)
                if self.inner_folds_to_run:
                    inner_rel = inner_rel[: self.inner_folds_to_run]
                inner = [(otr[itr], otr[iva]) for itr, iva in inner_rel]
                for g, (itr, iva) in enumerate(inner):
                    assert_subject_disjoint(self.groups, itr, iva)
                    seen = set(self.groups[itr].tolist()) | set(self.groups[iva].tolist())
                    if seen & test_subjects:
                        raise AssertionError("outer-test subject appeared inside the inner loop")
                    self.record_split(r, g, itr, iva, level="inner", outer_fold=f)
                leak.append({"repeat": r, "seed": seed, "outer_fold": f, "outer_train_test_overlap": 0,
                             "n_inner_folds": len(inner), "inner_train_val_overlap": 0,
                             "outer_test_subjects_in_inner": 0, "n_outer_test_subjects": len(test_subjects)})
                # --- inner stage A: feature count -------------------------------------------------
                t0 = time.time()
                rankings_inner = []
                for g, (itr, iva) in enumerate(inner):
                    rk = rank_features(self.X[itr], self.y[itr], PAPER_FEATURES, LGBM_DEFAULT, seed, self.n_threads)
                    rankings_inner.append(rk["feature"].tolist())
                self.timed("shap_ranking_inner_train", (time.time() - t0) / len(inner))
                pathA = forward_path(self.X, self.y, inner, rankings_inner, PAPER_FEATURES, LGBM_DEFAULT, seed,
                                     self.n_threads, self.max_k, log=self.log)
                k_star, aucA = best_k(pathA)
                inner_rows.append(pathA.assign(repeat=r, seed=seed, outer_fold=f, stage="k_selection").rename(columns={"fold": "inner_fold"}))
                self.log(f"  outer {f}: inner k*={k_star} (inner mean AUC {aucA:.4f}) [{time.time() - t0:.0f}s]")
                # --- inner stage B: hyperparameters at k* ------------------------------------------
                t0 = time.time()
                rowsB = []
                for g, (itr, iva) in enumerate(inner):
                    c = self.top_cols(rankings_inner[g], k_star)
                    for hi, H in enumerate(grid):
                        t1 = time.time()
                        m = make_lgbm(H, seed, self.n_threads).fit(self.X[np.ix_(itr, c)], self.y[itr])
                        self.timed(f"inner_fit_n_estimators_{H.get('n_estimators', 100)}", time.time() - t1)
                        auc = roc_auc_score(self.y[iva], m.predict_proba(self.X[np.ix_(iva, c)])[:, 1])
                        rowsB.append({"repeat": r, "seed": seed, "outer_fold": f, "inner_fold": g, "stage": "hp_selection",
                                      "grid_index": hi, "params": params_key(H), **H, "k": k_star, "auc": float(auc)})
                B = pd.DataFrame(rowsB)
                meanB = B.groupby("grid_index")["auc"].mean().reindex(range(len(grid)))
                h_star = int(meanB.to_numpy().argmax())  # first max in grid order (all-default first) (A20)
                H_star = grid[h_star]
                inner_rows.append(B)
                self.log(f"  outer {f}: inner H*={params_key(H_star)} (inner mean AUC {meanB.iloc[h_star]:.4f}) "
                         f"[{time.time() - t0:.0f}s]; paper-params inner AUC={meanB.iloc[len(grid) - 1]:.4f}, "
                         f"default inner AUC={meanB.iloc[0]:.4f}")
                # --- outer evaluation ---------------------------------------------------------------
                t0 = time.time()
                rk_outer = rank_features(self.X[otr], self.y[otr], PAPER_FEATURES, LGBM_DEFAULT, seed, self.n_threads)
                self.timed("shap_ranking_outer_train", time.time() - t0)
                ranking_outer = rk_outer["feature"].tolist()
                c_star = self.top_cols(ranking_outer, k_star)
                feat_rows.append(rk_outer.assign(repeat=r, seed=seed, outer_fold=f, k_selected=k_star, selected=rk_outer["rank"] <= k_star))
                extra = {"k": k_star, "inner_auc_at_k": aucA}
                one = [(otr, ote)]
                self.evaluate("nested_selected_k_selected_params", r, one, lambda _: make_lgbm(H_star, seed, self.n_threads),
                              c_star, fold_ids=[f], extra={**extra, "params": params_key(H_star), "inner_auc_at_params": float(meanB.iloc[h_star])})
                if self.cfg["hyperparameters"].get("also_report_fixed_paper_params", True):
                    self.evaluate("nested_selected_k_paper_params", r, one, lambda _: make_lgbm(PAPER_TABLE2, seed, self.n_threads),
                                  c_star, fold_ids=[f], extra={**extra, "params": params_key(PAPER_TABLE2)})
                self.evaluate("nested_selected_k_default_params", r, one, lambda _: make_lgbm(LGBM_DEFAULT, seed, self.n_threads),
                              c_star, fold_ids=[f], extra={**extra, "params": "default"})
                hp_rows.append({"repeat": r, "seed": seed, "outer_fold": f, "k_selected": k_star, "inner_mean_auc_at_k": aucA,
                                **{f"selected_{k}": v for k, v in H_star.items()}, "selected_params": params_key(H_star),
                                "inner_mean_auc_at_selected_params": float(meanB.iloc[h_star]),
                                "inner_mean_auc_paper_params": float(meanB.iloc[len(grid) - 1]),
                                "inner_mean_auc_default_params": float(meanB.iloc[0]),
                                "n_outer_train_subjects": len(set(self.groups[otr])), "n_outer_test_subjects": len(test_subjects)})
        pd.concat(inner_rows).to_csv(self.out / "inner_selection.csv", index=False)
        pd.concat(feat_rows).to_csv(self.out / "selected_features_by_fold.csv", index=False)
        pd.DataFrame(hp_rows).to_csv(self.out / "selected_hyperparameters_by_fold.csv", index=False)
        write_json(self.out / "leakage_assertions.json", leak)
        self.finish()

    # ------------------------------------------------------------------ artifacts
    def finish(self) -> None:
        write_results(self.out, self.name, self.smoke, pd.DataFrame(self.fold_rows), pd.concat(self.oof_rows, ignore_index=True),
                      pd.DataFrame(self.split_rows), self.cfg)
        write_json(self.out / "timing.json", {k: {"n": len(v), "mean_s": float(np.mean(v)), "total_s": float(np.sum(v))} for k, v in self.timing.items()})
        self.log(f"[{self.name}] artifacts written to {self.out}")

    def write_manifest(self, start: str, end: str) -> None:
        code_paths = sorted((self.exp_dir / "src" / "cheon").glob("*.py")) + [self.exp_dir / "run.py", self.exp_dir / "merge_runs.py"]
        man = build_manifest(experiment=self.name, config=self.cfg, config_text=self.cfg_text, repo_root=self.repo_root,
                             code_paths=[p for p in code_paths if p.exists()], data_fp=self.data_fp, seeds=self.seeds, start=start, end=end,
                             extra={"smoke": self.smoke, "output_dir": str(self.out), "n_features": len(PAPER_FEATURES),
                                    "max_k": self.max_k, "folds_run_per_repeat": self.folds_to_run or self.n_splits})
        write_json(self.out / "run_manifest.json", man)
        (self.out / "environment.txt").write_text(environment_text())


def write_results(out: Path, name: str, smoke: bool, fold_df: pd.DataFrame, oof: pd.DataFrame, split_df: pd.DataFrame, cfg: dict) -> dict:
    """Write fold_metrics.csv, oof_predictions.csv, split_summary.csv, metrics.json (shared by run and merge)."""
    out.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(out / "fold_metrics.csv", index=False)
    if name == "nested_cv":
        fold_df[fold_df["stage"].str.startswith("nested")].to_csv(out / "outer_fold_metrics.csv", index=False)
    oof.to_csv(out / "oof_predictions.csv", index=False)
    split_df.to_csv(out / "split_summary.csv", index=False)
    metrics = {"experiment": name, "smoke": smoke, "stages": {}, "primary_statistic":
               "per-repeat mean over folds (paper Figure 3), then mean/SD/95% t-CI across repeats; pooled-fold and pooled-OOF given as sensitivity"}
    for stage, g in fold_df.groupby("stage", sort=False):
        entry = aggregate(g, ALL_METRICS)
        entry["n_folds_per_repeat"] = int(g.groupby("repeat").size().iloc[0])
        entry["seeds"] = sorted(int(s) for s in g["seed"].unique())
        entry["pooled_oof_auc_per_repeat"] = pooled_oof_auc(oof[oof["stage"] == stage])
        entry["subject_overlap_max"] = int(g["subject_overlap"].max())
        if "k" in g.columns:
            entry["k_per_repeat"] = g.groupby("repeat")["k"].first().dropna().astype(int).tolist()
        metrics["stages"][stage] = entry
    if name == "reproduction":
        metrics["paper_comparison"] = paper_comparison(cfg, metrics["stages"])
    write_json(out / "metrics.json", metrics)
    return metrics


def paper_comparison(cfg: dict, stages: dict) -> dict:
        P = cfg.get("paper_reference", {})
        comp = {}
        def cmp(stage, paper_vals, note):
            if stage not in stages:
                return
            d = {}
            for m, pv in paper_vals.items():
                if pv is not None and m in stages[stage]:
                    ours = stages[stage][m]["primary_repeat_mean"]["mean"]
                    d[m] = {"paper": pv, "reproduced": ours, "difference": ours - pv, "abs_difference": abs(ours - pv)}
            comp[stage] = {"note": note, **d}
        cmp("baseline_lgbm_default_all72", P.get("baseline_lightgbm", {}), "Table 1 LightGBM row (all 72 features, default params)")
        cmp("table1_LightGBM", P.get("baseline_lightgbm", {}), "Table 1 LightGBM row, first seed only")
        fsr = P.get("feature_selection", {})
        cmp("fs_default_selected_k", {"roc_auc": fsr.get("roc_auc")}, f"forward selection best (paper k={fsr.get('k')})")
        cmp("fs_default_k40", {"roc_auc": fsr.get("roc_auc")}, "default LightGBM at the paper's k=40")
        cmp("final_tuned_selected_k", {"roc_auc": P.get("tuned_final", {}).get("roc_auc")}, "Table 2 params at our selected k")
        cmp("final_tuned_k40", {"roc_auc": P.get("tuned_final", {}).get("roc_auc")}, "Table 2 params at the paper's k=40")
        return comp


def run(cfg: dict, cfg_text: str, exp_dir: Path, repo_root: Path, data_root: Path, smoke: bool = False) -> Path:
    ex = Experiment(cfg, cfg_text, exp_dir, repo_root, data_root, smoke)
    start = utc_now()
    ex.log(f"start {start} experiment={ex.name} smoke={smoke} seeds={ex.seeds} out={ex.out}")
    ex.load()
    {"reproduction": ex.run_reproduction, "subject_independent": ex.run_subject_independent,
     "nested_cv": ex.run_nested}[ex.name]()
    end = utc_now()
    ex.write_manifest(start, end)
    ex.log(f"end {end}")
    return ex.out
