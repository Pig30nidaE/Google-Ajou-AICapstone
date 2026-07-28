import json
import os
import re

ipynb_files = [
    r'c:\ML4\Model\Binary\V29_Binary_Optuna_Stacking.ipynb',
    r'c:\ML4\Model\Binary\V35_V26_Super_Recall_Optimization.ipynb',
    r'c:\ML4\Model\Binary\V41_Binary_MMSE_Based.ipynb'
]

new_block = """# xai 패키지 경로 추가
import pathlib
current_dir = pathlib.Path(os.getcwd())
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from xai import ShapAnalyzer
    print("✅ ShapAnalyzer 모듈 임포트 성공!")
except Exception as e:
    print(f"🚨 xai 폴더는 찾았으나 임포트에 실패했습니다: {e}")
    try:
        from xai.analyzer import ShapAnalyzer
        print("✅ ShapAnalyzer (analyzer 우회) 임포트 성공!")
    except Exception as e2:
        print(f"🚨 우회 임포트도 실패했습니다: {e2}")
"""

def update_other(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    changed = False
    
    # We look for the block starting with try:\n    from google.colab import drive
    pattern = re.compile(r"try:\n\s*from google\.colab import drive.*?(?:from xai\.analyzer import ShapAnalyzer\n)", re.DOTALL)
    
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = "".join(cell.get('source', []))
            if "google.colab" in source and "xai" in source:
                new_source = pattern.sub(new_block + "\n", source)
                if new_source != source:
                    cell['source'] = [line + '\n' for line in new_source.split('\n')]
                    if cell['source']:
                        cell['source'][-1] = cell['source'][-1].rstrip('\n')
                    changed = True
                    
    if changed:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(f"Updated {os.path.basename(file_path)}")

for f in ipynb_files:
    try:
        update_other(f)
    except Exception as e:
        pass
