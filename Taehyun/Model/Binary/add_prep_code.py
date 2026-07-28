import json
import os

ipynb_files = [
    'V26_Binary_SOTA_Optimization.ipynb',
    'V29_Binary_Optuna_Stacking.ipynb',
    'V35_V26_Super_Recall_Optimization.ipynb',
    'V41_Binary_MMSE_Based.ipynb'
]

with open('c:\\ML4\\preprocessing_v2.py', 'r', encoding='utf-8') as f:
    prep_code = f.read()

prep_lines = [line + '\n' for line in prep_code.split('\n')]
prep_lines.insert(0, '# (참고) 원시 시계열 데이터를 환자 수준으로 사전 집계하는 기초 전처리 코드입니다.\n')
prep_lines.insert(1, '# 파일 경로 설정이 다를 수 있으므로 코드 구조만 참고하시기 바랍니다.\n\n')

for file in ipynb_files:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
            
        cells = notebook['cells']
        # Find where "데이터 전처리 (Data Preprocessing" is
        target_idx = -1
        for i, cell in enumerate(cells):
            if cell['cell_type'] == 'markdown':
                if any('데이터 전처리 (Data Preprocessing' in line for line in cell['source']):
                    target_idx = i
                    break
                    
        if target_idx != -1:
            # We want to insert a markdown cell and a code cell right after the header,
            # or before the header? User said "전처리 셀에 추가해줘"
            # I will insert it after the target_idx
            
            cells.insert(target_idx + 1, {
                'cell_type': 'markdown',
                'metadata': {},
                'source': ['### 📌 [참고] 원본 데이터 전처리 및 patient_level_all_v2.csv 생성 로직\n', '이 부분은 Raw 시계열 데이터에서 mean, std 등의 파생 변수를 추출하여 현재 노트북에서 사용하는 CSV 파일을 만들어낸 원본 스크립트입니다.']
            })
            cells.insert(target_idx + 2, {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': prep_lines
            })
            
        with open(file, 'w', encoding='utf-8') as out_f:
            json.dump(notebook, out_f, ensure_ascii=False, indent=1)
            
        print(f'Added preprocessing script to {file}')
    except Exception as e:
        print(f'Error processing {file}: {e}')
