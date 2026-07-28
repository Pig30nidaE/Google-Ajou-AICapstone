import json
import os
import re

ipynb_files = [
    r'c:\ML4\Model\Binary\V29_Binary_Optuna_Stacking.ipynb',
    r'c:\ML4\Model\Binary\V35_V26_Super_Recall_Optimization.ipynb',
    r'c:\ML4\Model\Binary\V41_Binary_MMSE_Based.ipynb'
]

def split_code(code_text):
    lines = code_text.split('\n')
    
    cells = []
    current_block = []
    
    # States
    # 0: Imports
    # 1: Data / Features
    # 2: Modeling
    # 3: Plotting
    # 4: Main
    
    state = 0
    state_titles = [
        "# 1. 패키지 임포트 및 환경 설정",
        "# 2. 데이터 로드 및 전처리",
        "# 3. 핵심 모델링 및 앙상블 함수",
        "# 4. 시각화 및 결과 분석",
        "# 5. 메인 실행 블록"
    ]
    
    blocks = [[], [], [], [], []]
    
    for line in lines:
        if line.startswith("def load_") or line.startswith("def get_shap"):
            state = max(state, 1)
        elif line.startswith("def perform_") or line.startswith("def optimize_") or line.startswith("def get_v26_oof"):
            state = max(state, 2)
        elif line.startswith("def plot_"):
            state = max(state, 3)
        elif line.startswith("if __name__ =="):
            state = max(state, 4)
            
        blocks[state].append(line)
        
    for i in range(5):
        if blocks[i]:
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [state_titles[i] + "\n"]
            })
            code = "\n".join(blocks[i]).strip() + "\n"
            cells.append({
                "cell_type": "code",
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in code.split('\n')][:-1] # removing the very last empty newline from split
            })
            
    return cells

for fpath in ipynb_files:
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        new_cells = []
        for cell in nb['cells']:
            if cell['cell_type'] == 'code' and len(cell.get('source', [])) > 20:
                code_text = "".join(cell['source'])
                if "import os" in code_text and "if __name__ == " in code_text:
                    # It's a giant cell
                    split_c = split_code(code_text)
                    new_cells.extend(split_c)
                else:
                    new_cells.append(cell)
            else:
                new_cells.append(cell)
                
        nb['cells'] = new_cells
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(f"Splitted {os.path.basename(fpath)}")
    except Exception as e:
        print(f"Error on {fpath}: {e}")
