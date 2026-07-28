import json
import os

ipynb_files = [
    'V26_Binary_SOTA_Optimization.ipynb',
    'V29_Binary_Optuna_Stacking.ipynb',
    'V35_V26_Super_Recall_Optimization.ipynb',
    'V41_Binary_MMSE_Based.ipynb'
]

old_xai_code = "try:\n    import google.colab\n    sys.path.insert(0, '/content/drive/MyDrive/GoogleAI_contest/Taehyun')\nexcept ImportError:"
new_xai_code = "try:\n    from google.colab import drive\n    drive.mount('/content/drive')\n    sys.path.insert(0, '/content/drive/MyDrive/GoogleAI_contest/Taehyun')\nexcept ImportError:"

for file_path in ipynb_files:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
            
        changed = False
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                source_str = "".join(cell['source'])
                if old_xai_code in source_str:
                    source_str = source_str.replace(old_xai_code, new_xai_code)
                    cell['source'] = [line + '\n' for line in source_str.split('\n')]
                    cell['source'][-1] = cell['source'][-1].rstrip('\n')
                    changed = True

        if changed:
            with open(file_path, 'w', encoding='utf-8') as out_f:
                json.dump(notebook, out_f, ensure_ascii=False, indent=1)
            print(f"Fixed mount order in {file_path}")
    except Exception as e:
        print(f"Error on {file_path}: {e}")
