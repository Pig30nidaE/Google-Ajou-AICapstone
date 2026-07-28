import json
import os

ipynb_files = [
    r'c:\ML4\Model\Binary\V26_Binary_SOTA_Optimization.ipynb',
    r'c:\ML4\Model\Binary\V29_Binary_Optuna_Stacking.ipynb',
    r'c:\ML4\Model\Binary\V35_V26_Super_Recall_Optimization.ipynb',
    r'c:\ML4\Model\Binary\V41_Binary_MMSE_Based.ipynb'
]

def fix_font_warning(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    changed = False
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = "".join(cell.get('source', []))
            
            if "plt.rcParams['font.family'] = 'Malgun Gothic'" in source:
                new_source = source.replace(
                    "plt.rcParams['font.family'] = 'Malgun Gothic'",
                    "if sys.platform == 'win32':\n    plt.rcParams['font.family'] = 'Malgun Gothic'"
                )
                if new_source != source:
                    cell['source'] = [line + '\n' for line in new_source.split('\n')]
                    if cell['source']:
                        cell['source'][-1] = cell['source'][-1].rstrip('\n')
                    changed = True

    if changed:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(f"Fixed font warning in {os.path.basename(file_path)}")

for f in ipynb_files:
    try:
        fix_font_warning(f)
    except Exception as e:
        print(f"Error on {f}: {e}")
