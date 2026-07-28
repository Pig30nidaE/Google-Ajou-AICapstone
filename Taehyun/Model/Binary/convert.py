import json
import os

files = [
    'V26_Binary_SOTA_Optimization.py',
    'V29_Binary_Optuna_Stacking.py',
    'V35_V26_Super_Recall_Optimization.py',
    'V41_Binary_MMSE_Based.py'
]

for file in files:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        notebook = {
            'cells': [
                {
                    'cell_type': 'markdown',
                    'metadata': {},
                    'source': [f'# {file}\n', '이 노트북은 파이썬 스크립트에서 자동 변환되었습니다.']
                },
                {
                    'cell_type': 'code',
                    'execution_count': None,
                    'metadata': {},
                    'outputs': [],
                    'source': [line + '\n' for line in content.split('\n')]
                }
            ],
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
            
        print(f'Created {ipynb_name}')
    except Exception as e:
        print(f'Error processing {file}: {e}')
