# XAI Paper Reproduction

Target paper:

`docs/설명가능 인공지능을 활용한 라이프로그 기반 치매 위험도 산정 방법에 관한 연구.pdf`

This folder implements a paper-first reproduction pipeline from the raw AI-Hub sample data under
`128.치매 고위험군 라이프로그/`.

## Paper Targets

| Item | Paper setting |
| --- | --- |
| Dataset | AI-Hub dementia high-risk wearable lifelog, 174 subjects |
| Unit | Daily lifelog row |
| Target | Binary: `CN=0`, `MCI/Dem=1` |
| Class count | CN `7,737`, impaired `4,446` daily rows |
| Validation | 5-fold cross validation |
| Model comparison | Logistic Regression, Decision Tree, KNN, SVM, MLP, Random Forest, LightGBM |
| Selection metric | ROC-AUC for impaired class |
| Best model in paper | LightGBM |
| Feature selection | SHAP importance forward selection, best at top 40 features |
| Tuned LightGBM | `min_data_in_leaf=41`, `num_leaves=330`, `n_estimators=1000`, `learning_rate=0.08` |
| Risk score | Sum of positive SHAP values per daily row |
| Risk validation | One-sided one-sample t-test: impaired DRS mean > CN DRS mean |

## Folder Structure

```text
training/XAI_Paper_Reproduction/
├── XAI_Paper_Reproduction_Colab.ipynb
├── README.md
├── REPRODUCTION_DETAILS_KO.md
├── requirements.txt
├── src/
│   └── xai_paper_reproduction.py
└── scripts/
    ├── 00_validate_environment.py
    ├── 01_preprocess_daily_binary.py
    ├── 02_compare_baseline_models.py
    ├── 03_shap_forward_selection.py
    ├── 04_final_lgbm_shap_drs.py
    ├── 05_make_reproduction_report.py
    └── 06_audit_reproduction_outputs.py
```

Outputs are written to `training/XAI_Paper_Reproduction/outputs/` by default.

For Colab, open `XAI_Paper_Reproduction_Colab.ipynb` and run the notebook cells in order.

For a detailed Korean record of preprocessing, training, paper-specified settings, unspecified settings,
and reproduction decisions, see `REPRODUCTION_DETAILS_KO.md`.

If Colab sessions may run in separate runtimes, mount Google Drive and pass the same persistent
`--output-dir` to scripts `01` through `06`, for example:

```bash
XAI_OUTPUT_DIR=/content/drive/MyDrive/xai_paper_reproduction_outputs
```

## Colab Pro+ Run Plan

Use up to three sessions:

### Session 1 - Preprocess and model comparison

```bash
pip install -r training/XAI_Paper_Reproduction/requirements.txt
python training/XAI_Paper_Reproduction/scripts/00_validate_environment.py --run-preprocess-smoke --strict
python training/XAI_Paper_Reproduction/scripts/01_preprocess_daily_binary.py
python training/XAI_Paper_Reproduction/scripts/02_compare_baseline_models.py
```

With a persistent output directory:

```bash
python training/XAI_Paper_Reproduction/scripts/01_preprocess_daily_binary.py --output-dir "$XAI_OUTPUT_DIR"
python training/XAI_Paper_Reproduction/scripts/02_compare_baseline_models.py --output-dir "$XAI_OUTPUT_DIR"
```

Expected key outputs:

- `outputs/data/daily_binary_lifelog.csv`
- `outputs/data/feature_columns.json`
- `outputs/baselines/model_comparison.csv`
- `outputs/baselines/model_comparison_against_paper.csv`

Fast smoke test before the full seven-model run:

```bash
python training/XAI_Paper_Reproduction/scripts/02_compare_baseline_models.py --models LightGBM --n-splits 2
```

### Session 2 - SHAP forward feature selection

```bash
pip install -r training/XAI_Paper_Reproduction/requirements.txt
python training/XAI_Paper_Reproduction/scripts/00_validate_environment.py --strict
python training/XAI_Paper_Reproduction/scripts/03_shap_forward_selection.py --max-features 80
```

With a persistent output directory:

```bash
python training/XAI_Paper_Reproduction/scripts/03_shap_forward_selection.py --output-dir "$XAI_OUTPUT_DIR" --max-features 80
```

Expected key outputs:

- `outputs/feature_selection/shap_importance_full.csv`
- `outputs/feature_selection/forward_selection_metrics.csv`
- `outputs/feature_selection/selected_features.json`
- `outputs/feature_selection/forward_selection_auc.png`

### Session 3 - Final LightGBM, SHAP, and Dementia Risk Score

```bash
pip install -r training/XAI_Paper_Reproduction/requirements.txt
python training/XAI_Paper_Reproduction/scripts/00_validate_environment.py --strict
python training/XAI_Paper_Reproduction/scripts/04_final_lgbm_shap_drs.py --use-paper-params
python training/XAI_Paper_Reproduction/scripts/05_make_reproduction_report.py
python training/XAI_Paper_Reproduction/scripts/06_audit_reproduction_outputs.py
```

With a persistent output directory:

```bash
python training/XAI_Paper_Reproduction/scripts/04_final_lgbm_shap_drs.py --output-dir "$XAI_OUTPUT_DIR" --use-paper-params
python training/XAI_Paper_Reproduction/scripts/05_make_reproduction_report.py --output-dir "$XAI_OUTPUT_DIR"
python training/XAI_Paper_Reproduction/scripts/06_audit_reproduction_outputs.py --output-dir "$XAI_OUTPUT_DIR"
```

Optional grid search:

```bash
python training/XAI_Paper_Reproduction/scripts/04_final_lgbm_shap_drs.py --grid-search
```

Fast smoke test before the full final run:

```bash
python training/XAI_Paper_Reproduction/scripts/04_final_lgbm_shap_drs.py --use-paper-params --n-splits 2 --n-estimators-override 50 --max-drs-rows 500
python training/XAI_Paper_Reproduction/scripts/05_make_reproduction_report.py
python training/XAI_Paper_Reproduction/scripts/06_audit_reproduction_outputs.py --allow-smoke
```

Expected key outputs:

- `outputs/final/final_cv_metrics.json`
- `outputs/final/dementia_risk_scores.csv`
- `outputs/final/dementia_risk_score_summary.json`
- `outputs/final/dementia_risk_score_histogram.png`
- `outputs/final/shap_summary_positive.png`
- `outputs/final/shap_importance_positive.csv`
- `outputs/reproduction_report.md`
- `outputs/reproduction_audit.json`
- `outputs/reproduction_audit.md`

## Important Reproduction Assumptions

- The paper reports `7,737 + 4,446 = 12,183` daily rows. This exactly matches the activity daily rows in the provided sample. Sleep rows are 12 fewer, so preprocessing uses activity rows as the base and left-joins sleep features. Missing sleep values are imputed inside each CV fold.
- The paper describes 5-fold CV on daily rows and later notes the limitation that the model is daily-row based because only 174 subjects are available. Therefore row-level `StratifiedKFold` is the default. The code also supports subject-group folds for diagnostic experiments.
- The paper does not disclose all preprocessing details or untuned model hyperparameters. The scripts keep paper-stated values fixed where available and expose unclear choices as CLI options.
