import json
import os

ipynb_files = [
    r'c:\ML4\Model\Binary\V26_Binary_SOTA_Optimization.ipynb',
    r'c:\ML4\Model\Binary\V29_Binary_Optuna_Stacking.ipynb',
    r'c:\ML4\Model\Binary\V35_V26_Super_Recall_Optimization.ipynb',
    r'c:\ML4\Model\Binary\V41_Binary_MMSE_Based.ipynb'
]

def make_silent(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    changed = False
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = "".join(cell.get('source', []))
            
            # CatBoost fit
            if "m_cat.fit(X_tr_res, y_tr_res, eval_set=(X_te, y_te), early_stopping_rounds=50)" in source:
                source = source.replace(
                    "m_cat.fit(X_tr_res, y_tr_res, eval_set=(X_te, y_te), early_stopping_rounds=50)",
                    "m_cat.fit(X_tr_res, y_tr_res, eval_set=(X_te, y_te), early_stopping_rounds=50, verbose=False)"
                )
            if "m_cat.fit(X_tr_res, y_tr_res, eval_set=[(X_te, y_te)], early_stopping_rounds=50)" in source:
                source = source.replace(
                    "m_cat.fit(X_tr_res, y_tr_res, eval_set=[(X_te, y_te)], early_stopping_rounds=50)",
                    "m_cat.fit(X_tr_res, y_tr_res, eval_set=[(X_te, y_te)], early_stopping_rounds=50, verbose=False)"
                )
            
            # LightGBM callbacks
            if "callbacks=[lgb.early_stopping(50, verbose=False)]" in source:
                source = source.replace(
                    "callbacks=[lgb.early_stopping(50, verbose=False)]",
                    "callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]"
                )
            if "callbacks=[lgb.early_stopping(30, verbose=False)]" in source:
                source = source.replace(
                    "callbacks=[lgb.early_stopping(30, verbose=False)]",
                    "callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)]"
                )
                
            # LGBM Classifier instances
            if "m_lgb = LGBMClassifier(\n" in source and "verbose=-1" not in source:
                source = source.replace("m_lgb = LGBMClassifier(\n", "m_lgb = LGBMClassifier(\n            verbose=-1,\n")
            if "eval_model = LGBMClassifier(\n" in source and "verbose=-1" not in source:
                source = source.replace("eval_model = LGBMClassifier(\n", "eval_model = LGBMClassifier(\n                verbose=-1,\n")

            # Check if any change occurred
            if "".join(cell.get('source', [])) != source:
                cell['source'] = [line + '\n' for line in source.split('\n')]
                if cell['source']:
                    cell['source'][-1] = cell['source'][-1].rstrip('\n')
                changed = True

    if changed:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(f"Made silent in {os.path.basename(file_path)}")

for f in ipynb_files:
    try:
        make_silent(f)
    except Exception as e:
        print(f"Error on {f}: {e}")
