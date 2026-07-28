import json
import os

ipynb_files = [
    'V26_Binary_SOTA_Optimization.ipynb',
    'V29_Binary_Optuna_Stacking.ipynb',
    'V35_V26_Super_Recall_Optimization.ipynb',
    'V41_Binary_MMSE_Based.ipynb'
]

old_path = '/content/drive/MyDrive/Google-Ajou-AICapstone'
new_path = '/content/drive/MyDrive/GoogleAI_contest/Taehyun'

def replace_colab_paths(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
        
    changed = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source_str = "".join(cell['source'])
            if old_path in source_str:
                source_str = source_str.replace(old_path, new_path)
                cell['source'] = [line + '\n' for line in source_str.split('\n')]
                cell['source'][-1] = cell['source'][-1].rstrip('\n')
                changed = True

    if changed:
        with open(file_path, 'w', encoding='utf-8') as out_f:
            json.dump(notebook, out_f, ensure_ascii=False, indent=1)
        print(f"Updated paths in {file_path}")

for f in ipynb_files:
    try:
        replace_colab_paths(f)
    except Exception as e:
        print(f"Error on {f}: {e}")
