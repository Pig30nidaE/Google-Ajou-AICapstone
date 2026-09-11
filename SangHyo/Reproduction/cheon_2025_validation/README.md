# Cheon et al. (2025) — paper-first validation: R / G / N

Independent re-implementation of 천희웅 외 (2025), 「설명가능 인공지능을 활용한 라이프로그 기반 치매 위험도
산정 방법에 관한 연구」, JKIIE 51(2):161–170, built from the PDF only, then run under three validation
designs on one shared pipeline:

| Arm | Question | Split | Learned steps |
|---|---|---|---|
| R `reproduction` | Does the paper reproduce? | record-level StratifiedKFold(5) (paper) | SHAP ranking on all records (paper-faithful) |
| G `subject_independent` | Same recipe, evaluation subjects excluded from training | StratifiedGroupKFold(5) by subject | SHAP ranking on the training fold; k = 40 fixed (paper) |
| N `nested_cv` | … and excluded from feature/model selection too | outer StratifiedGroupKFold(5) + inner StratifiedGroupKFold(5) | k and hyperparameters chosen on inner folds |

Documents (read in this order): `PAPER_PROTOCOL.md` → `ASSUMPTIONS.md` → `PROTOCOL_COMPARISON.md` →
`RESULT_COMPARISON.md` → `EXISTING_CODE_COMPARISON.md` → `FINAL_REPORT.md`.

## Layout

```
configs/{reproduction,subject_independent,nested_cv}.yaml
src/cheon/{data,features,splits,model,selection,evaluation,manifest,pipeline}.py
run.py               # python run.py --config configs/<exp>.yaml [--smoke] [--seeds 42] [--output-dir ...]
merge_runs.py        # merge single-seed runs (used for N)
launch_all.sh        # the exact command set used for the full runs
report_tables.py     # markdown tables from results/
tests/               # leakage / scope / forbidden-feature tests (pytest)
results/<exp>/       # metrics.json, fold_metrics.csv, oof_predictions.csv (hashed ids), split_summary.csv,
                     # run_manifest.json, environment.txt, stdout.log, timing.json, + per-experiment extras
```

## Environment

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python lightgbm shap scikit-learn pandas numpy scipy pyyaml pytest psutil
.venv/bin/python -m pytest tests -q
.venv/bin/python run.py --config configs/reproduction.yaml --smoke     # ~1 min sanity run
./launch_all.sh                                                        # full R, G, N (≈30 min wall on 8 cores)
.venv/bin/python report_tables.py > results/summary_tables.md
```

Raw data (`<repo>/Data`) is read-only; MMSE / cognitive-test files are never opened; subject identifiers
are hashed at load and never written to any artifact.
