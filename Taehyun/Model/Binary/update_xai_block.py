import json
import os
import pathlib

ipynb_files = [
    r'c:\ML4\Model\Binary\V26_Binary_SOTA_Optimization.ipynb',
    r'c:\ML4\Model\Binary\V29_Binary_Optuna_Stacking.ipynb',
    r'c:\ML4\Model\Binary\V35_V26_Super_Recall_Optimization.ipynb',
    r'c:\ML4\Model\Binary\V41_Binary_MMSE_Based.ipynb'
]

new_block = """# xai 패키지 경로 추가 (강력 진단 코드 적용)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    colab_path = '/content/drive/MyDrive/GoogleAI_contest/Taehyun'
    if colab_path not in sys.path:
        sys.path.insert(0, colab_path)
    if not os.path.exists(os.path.join(colab_path, 'xai')):
        print(f"🚨 에러: {colab_path} 경로에 xai 폴더가 없습니다!")
        if os.path.exists(colab_path):
            print("👉 현재 폴더 안에 있는 파일들:", os.listdir(colab_path))
    else:
        print("✅ 구글 드라이브에서 xai 폴더를 정상적으로 확인했습니다!")
except ImportError:
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

def update_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    changed = False
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if "# xai 패키지 경로 추가" in source:
                start_idx = source.find("# xai 패키지 경로 추가")
                end_idx = source.find("# 한글 폰트 설정")
                
                if end_idx != -1:
                    new_source = source[:start_idx] + new_block + "\n" + source[end_idx:]
                else:
                    new_source = source[:start_idx] + new_block
                
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
        update_notebook(f)
    except Exception as e:
        print(f"Failed {f}: {e}")
