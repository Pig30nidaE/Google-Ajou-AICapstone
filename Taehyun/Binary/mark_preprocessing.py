import json
import os
import re

files = [
    'V26_Binary_SOTA_Optimization.py',
    'V29_Binary_Optuna_Stacking.py',
    'V35_V26_Super_Recall_Optimization.py',
    'V41_Binary_MMSE_Based.py'
]

def split_to_cells(content):
    cells = []
    cells.append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': ["!pip install catboost xgboost lightgbm optuna shap imbalanced-learn\n"]
    })
    
    separator = "# ========================================================="
    parts = content.split(separator)
    
    imports_part = parts[0].strip()
    if imports_part:
        cells.append({
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [line + '\n' for line in imports_part.split('\n')]
        })
    
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            header = parts[i].strip()
            code = parts[i+1].strip()
            
            # Explicitly mark preprocessing
            if "데이터 로드" in header or "데이터 로드 및 결합" in header:
                header = header.replace("데이터 로드", "데이터 전처리 (Data Preprocessing - Load, Impute, Engineering)")
                header = header.replace("데이터 로드 및 결합", "데이터 전처리 (Data Preprocessing - Load, Impute, Engineering)")
            elif "2." in header and "전처리" not in header:
                header = header.replace("2.", "2. 데이터 전처리 (Data Preprocessing)")
            
            # Add SMOTE emphasis in modeling sections
            if "SHAP 기반" in header or "학습" in header or "SOTA 앙상블" in header:
                header = header + "\n\n**📌 참고**: SMOTE를 활용한 핵심 전처리(오버샘플링)는 Data Leakage를 막기 위해 아래 교차검증(CV) 루프 내부에 구현되어 있습니다."
            
            if header:
                cells.append({
                    'cell_type': 'markdown',
                    'metadata': {},
                    'source': [line + '\n' for line in header.split('\n')]
                })
            
            if code:
                cells.append({
                    'cell_type': 'code',
                    'execution_count': None,
                    'metadata': {},
                    'outputs': [],
                    'source': [line + '\n' for line in code.split('\n')]
                })
                
    if len(parts) <= 1:
        cells.append({
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [line + '\n' for line in content.split('\n')]
        })
        
    return cells

for file in files:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        notebook = {
            'cells': split_to_cells(content),
            'metadata': {
                'kernelspec': {
                    'display_name': 'Python 3',
                    'language': 'python',
                    'name': 'python3'
                }
            },
            'nbformat': 4,
            'nbformat_minor': 4
        }
        
        ipynb_name = file.replace('.py', '.ipynb')
        with open(ipynb_name, 'w', encoding='utf-8') as out_f:
            json.dump(notebook, out_f, ensure_ascii=False, indent=1)
            
        print(f'Marked preprocessing in {ipynb_name}')
    except Exception as e:
        print(f'Error processing {file}: {e}')
