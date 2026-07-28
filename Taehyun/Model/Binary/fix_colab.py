import json
import os

ipynb_files = [
    'V26_Binary_SOTA_Optimization.ipynb',
    'V29_Binary_Optuna_Stacking.ipynb',
    'V35_V26_Super_Recall_Optimization.ipynb',
    'V41_Binary_MMSE_Based.ipynb'
]

def fix_colab_paths(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
        
    cells = notebook['cells']
    
    for cell in cells:
        if cell['cell_type'] == 'code':
            source_str = "".join(cell['source'])
            
            # Fix BASE_DIR
            if 'BASE_DIR = pathlib.Path(r"c:\ML4")' in source_str or "BASE_DIR = pathlib.Path(r'c:\ML4')" in source_str:
                source_str = source_str.replace(
                    'BASE_DIR = pathlib.Path(r"c:\ML4")', 
                    "try:\n    from google.colab import drive\n    drive.mount('/content/drive')\n    BASE_DIR = pathlib.Path('/content/drive/MyDrive/Google-Ajou-AICapstone')\nexcept ImportError:\n    BASE_DIR = pathlib.Path(r\"c:\ML4\")"
                )
            
            # Fix xai path
            if 'sys.path.insert(0, str(current_dir))' in source_str:
                source_str = source_str.replace(
                    "current_dir = pathlib.Path(os.getcwd())\n    sys.path.insert(0, str(current_dir))",
                    "try:\n        import google.colab\n        sys.path.insert(0, '/content/drive/MyDrive/Google-Ajou-AICapstone')\n    except ImportError:\n        current_dir = pathlib.Path(os.getcwd())\n        sys.path.insert(0, str(current_dir))"
                )
                source_str = source_str.replace(
                    "current_dir = pathlib.Path(os.getcwd())\nsys.path.insert(0, str(current_dir))",
                    "try:\n    import google.colab\n    sys.path.insert(0, '/content/drive/MyDrive/Google-Ajou-AICapstone')\nexcept ImportError:\n    current_dir = pathlib.Path(os.getcwd())\n    sys.path.insert(0, str(current_dir))"
                )
                
            if source_str != "".join(cell['source']):
                # Remove empty strings from list comprehension
                cell['source'] = [line + '\n' for line in source_str.split('\n')]
                # Clean up trailing newlines correctly
                cell['source'][-1] = cell['source'][-1].rstrip('\n')

    with open(file_path, 'w', encoding='utf-8') as out_f:
        json.dump(notebook, out_f, ensure_ascii=False, indent=1)

for f in ipynb_files:
    try:
        fix_colab_paths(f)
        print(f"Fixed Colab paths in {f}")
    except Exception as e:
        print(f"Error on {f}: {e}")
