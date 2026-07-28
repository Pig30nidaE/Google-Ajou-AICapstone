import glob
import os
import re

folder = r'c:\ML4\Model\Binary'
for py_file in glob.glob(os.path.join(folder, '*.py')):
    with open(py_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # We want to match the whole try-except block for google.colab
    pattern = r"# xai 패키지 경로 추가.*?except ImportError:\s*import pathlib\s*current_dir = pathlib\.Path\(os\.getcwd\(\)\)\s*if str\(current_dir\) not in sys\.path:\s*sys\.path\.insert\(0, str\(current_dir\)\)"
    
    if re.search(pattern, text, re.DOTALL):
        replacement = """# xai 패키지 경로 추가
import pathlib
current_dir = pathlib.Path(os.getcwd())
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))"""
        new_text = re.sub(pattern, replacement, text, flags=re.DOTALL)
        
        with open(py_file, 'w', encoding='utf-8') as f:
            f.write(new_text)
        print(f"Removed Colab block from {os.path.basename(py_file)}")
