import json
import os

def ipynb_to_py(ipynb_path, py_path):
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    lines = []
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = "".join(cell.get('source', []))
            if not source.startswith('!pip'):
                lines.append(source)
                lines.append('\n\n')
                
    with open(py_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"Converted {ipynb_path} to {py_path}")

ipynb_to_py(r'c:\ML4\Model\Binary\V29_Binary_Optuna_Stacking.ipynb', r'c:\ML4\Model\Binary\V29_Binary_Optuna_Stacking.py')
ipynb_to_py(r'c:\ML4\Model\Binary\V35_V26_Super_Recall_Optimization.ipynb', r'c:\ML4\Model\Binary\V35_V26_Super_Recall_Optimization.py')
ipynb_to_py(r'c:\ML4\Model\Binary\V41_Binary_MMSE_Based.ipynb', r'c:\ML4\Model\Binary\V41_Binary_MMSE_Based.py')
