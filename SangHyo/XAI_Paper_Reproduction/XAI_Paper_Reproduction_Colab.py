# Auto-generated Python script converted from a Jupyter notebook.
# Source notebook: SangHyo/XAI_Paper_Reproduction/XAI_Paper_Reproduction_Colab.ipynb
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
# # XAI Paper Reproduction - Colab Runner
#
# This notebook runs the reproduction pipeline for `설명가능 인공지능을 활용한 라이프로그 기반 치매 위험도 산정 방법에 관한 연구.pdf`.
#
# Expected Google Drive layout:
#
# ```text
# MyDrive/
# └── TeamProject/
#     ├── 128.치매 고위험군 라이프로그/
#     ├── docs/
#     └── training/
#         └── XAI_Paper_Reproduction/
# ```
#
# Run the cells in order. Each bash cell redeclares paths intentionally because Colab sessions and cells can lose shell variables.

# %% [markdown] cell 2
# ## 0. Mount Google Drive

# %% cell 3
from google.colab import drive
drive.mount("/content/drive")

# %% [markdown] cell 4
# ## 1. Check Paths
#
# Run this first. If any `ls` command fails, fix the Google Drive folder layout before continuing.

# %% cell 5
_NOTEBOOK_RUN_BASH('set -e\n\nPROJECT_DIR="/content/drive/MyDrive/TeamProject"\nRAW_DIR="$PROJECT_DIR/128.치매 고위험군 라이프로그"\nXAI_DIR="$PROJECT_DIR/training/XAI_Paper_Reproduction"\nXAI_OUTPUT_DIR="$XAI_DIR/outputs"\n\necho "PROJECT_DIR=$PROJECT_DIR"\necho "RAW_DIR=$RAW_DIR"\necho "XAI_DIR=$XAI_DIR"\necho "XAI_OUTPUT_DIR=$XAI_OUTPUT_DIR"\n\nls -la "$PROJECT_DIR"\nls -la "$RAW_DIR"\nls -la "$XAI_DIR"')

# %% [markdown] cell 6
# ## Session 1. Install, Validate, Preprocess, and Compare Baseline Models
#
# This is the longest baseline step because it evaluates all seven paper models.

# %% cell 7
_NOTEBOOK_RUN_BASH('set -e\n\nPROJECT_DIR="/content/drive/MyDrive/TeamProject"\nRAW_DIR="$PROJECT_DIR/128.치매 고위험군 라이프로그"\nXAI_DIR="$PROJECT_DIR/training/XAI_Paper_Reproduction"\nXAI_OUTPUT_DIR="$XAI_DIR/outputs"\n\ncd "$PROJECT_DIR"\nmkdir -p "$XAI_OUTPUT_DIR"\n\npip install -r "$XAI_DIR/requirements.txt"\n\npython "$XAI_DIR/scripts/00_validate_environment.py" \\\n  --raw-dir "$RAW_DIR" \\\n  --output-dir "$XAI_OUTPUT_DIR" \\\n  --run-preprocess-smoke \\\n  --strict\n\npython "$XAI_DIR/scripts/01_preprocess_daily_binary.py" \\\n  --raw-dir "$RAW_DIR" \\\n  --output-dir "$XAI_OUTPUT_DIR"\n\npython "$XAI_DIR/scripts/02_compare_baseline_models.py" \\\n  --output-dir "$XAI_OUTPUT_DIR"')

# %% [markdown] cell 8
# ### Optional Session 1 Smoke Test
#
# Use this instead of the full Session 1 model comparison only when checking that the notebook runs. It runs LightGBM only.

# %% cell 9
_NOTEBOOK_RUN_BASH('set -e\n\nPROJECT_DIR="/content/drive/MyDrive/TeamProject"\nRAW_DIR="$PROJECT_DIR/128.치매 고위험군 라이프로그"\nXAI_DIR="$PROJECT_DIR/training/XAI_Paper_Reproduction"\nXAI_OUTPUT_DIR="$XAI_DIR/outputs_smoke"\n\ncd "$PROJECT_DIR"\nmkdir -p "$XAI_OUTPUT_DIR"\n\npip install -r "$XAI_DIR/requirements.txt"\n\npython "$XAI_DIR/scripts/00_validate_environment.py" \\\n  --raw-dir "$RAW_DIR" \\\n  --output-dir "$XAI_OUTPUT_DIR" \\\n  --run-preprocess-smoke \\\n  --strict\n\npython "$XAI_DIR/scripts/01_preprocess_daily_binary.py" \\\n  --raw-dir "$RAW_DIR" \\\n  --output-dir "$XAI_OUTPUT_DIR"\n\npython "$XAI_DIR/scripts/02_compare_baseline_models.py" \\\n  --output-dir "$XAI_OUTPUT_DIR" \\\n  --models LightGBM \\\n  --n-splits 2')

# %% [markdown] cell 10
# ## Session 2. SHAP Importance and Forward Feature Selection

# %% cell 11
_NOTEBOOK_RUN_BASH('set -e\n\nPROJECT_DIR="/content/drive/MyDrive/TeamProject"\nRAW_DIR="$PROJECT_DIR/128.치매 고위험군 라이프로그"\nXAI_DIR="$PROJECT_DIR/training/XAI_Paper_Reproduction"\nXAI_OUTPUT_DIR="$XAI_DIR/outputs"\n\ncd "$PROJECT_DIR"\n\npip install -r "$XAI_DIR/requirements.txt"\n\npython "$XAI_DIR/scripts/00_validate_environment.py" \\\n  --raw-dir "$RAW_DIR" \\\n  --output-dir "$XAI_OUTPUT_DIR" \\\n  --strict\n\npython "$XAI_DIR/scripts/03_shap_forward_selection.py" \\\n  --output-dir "$XAI_OUTPUT_DIR" \\\n  --max-features 80')

# %% [markdown] cell 12
# ## Session 3. Final LightGBM, SHAP DRS, Report, and Audit

# %% cell 13
_NOTEBOOK_RUN_BASH('set -e\n\nPROJECT_DIR="/content/drive/MyDrive/TeamProject"\nRAW_DIR="$PROJECT_DIR/128.치매 고위험군 라이프로그"\nXAI_DIR="$PROJECT_DIR/training/XAI_Paper_Reproduction"\nXAI_OUTPUT_DIR="$XAI_DIR/outputs"\n\ncd "$PROJECT_DIR"\n\npip install -r "$XAI_DIR/requirements.txt"\n\npython "$XAI_DIR/scripts/00_validate_environment.py" \\\n  --raw-dir "$RAW_DIR" \\\n  --output-dir "$XAI_OUTPUT_DIR" \\\n  --strict\n\npython "$XAI_DIR/scripts/04_final_lgbm_shap_drs.py" \\\n  --output-dir "$XAI_OUTPUT_DIR" \\\n  --use-paper-params\n\npython "$XAI_DIR/scripts/05_make_reproduction_report.py" \\\n  --output-dir "$XAI_OUTPUT_DIR"\n\npython "$XAI_DIR/scripts/06_audit_reproduction_outputs.py" \\\n  --output-dir "$XAI_OUTPUT_DIR"')

# %% [markdown] cell 14
# ### Optional Session 3 Smoke Test
#
# This uses fewer trees and samples DRS rows. Do not use these smoke outputs for final paper comparison.

# %% cell 15
_NOTEBOOK_RUN_BASH('set -e\n\nPROJECT_DIR="/content/drive/MyDrive/TeamProject"\nXAI_DIR="$PROJECT_DIR/training/XAI_Paper_Reproduction"\nXAI_OUTPUT_DIR="$XAI_DIR/outputs_smoke"\n\ncd "$PROJECT_DIR"\n\npython "$XAI_DIR/scripts/04_final_lgbm_shap_drs.py" \\\n  --output-dir "$XAI_OUTPUT_DIR" \\\n  --use-paper-params \\\n  --n-splits 2 \\\n  --n-estimators-override 50 \\\n  --max-drs-rows 500\n\npython "$XAI_DIR/scripts/05_make_reproduction_report.py" \\\n  --output-dir "$XAI_OUTPUT_DIR"\n\npython "$XAI_DIR/scripts/06_audit_reproduction_outputs.py" \\\n  --output-dir "$XAI_OUTPUT_DIR" \\\n  --allow-smoke')

# %% [markdown] cell 16
# ## 4. Summarize Completed Results
#
# Run after Session 3. The final paper-comparable ROC-AUC is the 5-fold CV value from `final_cv_metrics.json`, not the in-sample ROC image.

# %% cell 17
from pathlib import Path
import json
import pandas as pd

output_dir = Path('/content/drive/MyDrive/TeamProject/training/XAI_Paper_Reproduction/outputs')

preprocess = json.loads((output_dir / 'data/preprocess_summary.json').read_text())
baseline = pd.read_csv(output_dir / 'baselines/model_comparison.csv').sort_values('roc_auc', ascending=False)
fs = pd.read_csv(output_dir / 'feature_selection/forward_selection_metrics.csv')
final = json.loads((output_dir / 'final/final_cv_metrics.json').read_text())
drs = json.loads((output_dir / 'final/dementia_risk_score_summary.json').read_text())

print('Preprocess:', preprocess)
print('\nTop baseline models:')
display(baseline[['model', 'accuracy', 'roc_auc', 'f1_macro']].head(7))

best_fs = fs.sort_values('roc_auc', ascending=False).iloc[0]
top40 = fs[fs['n_features'].eq(40)].iloc[0]
print('\nForward selection best:', best_fs.to_dict())
print('Forward selection top 40:', top40.to_dict())

print('\nFinal CV metrics:', {k: final['cv_metrics'][k] for k in ['accuracy', 'roc_auc', 'precision_macro', 'recall_macro', 'f1_macro']})
print('Final params:', final['params'])
print('\nDaily DRS t-test:', drs['daily_one_sided_t_test'])
print('\nAudit files:')
print(output_dir / 'reproduction_report.md')
print(output_dir / 'reproduction_audit.md')
