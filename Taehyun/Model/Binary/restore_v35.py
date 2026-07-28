import json
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

ipynb_to_py(r'c:\ML4\Model\Binary\V35_V26_Super_Recall_Optimization.ipynb', r'c:\ML4\Model\Binary\V35_V26_Super_Recall_Optimization.py')

# Now fix the PLOT_DIR definition
with open(r'c:\ML4\Model\Binary\V35_V26_Super_Recall_Optimization.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace PLOT_DIR / "confusion_matrix with BASE_DIR / 'report/plots' / "confusion_matrix
text = text.replace('PLOT_DIR / "confusion_matrix', "BASE_DIR / 'report' / 'plots' / 'confusion_matrix")
with open(r'c:\ML4\Model\Binary\V35_V26_Super_Recall_Optimization.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Restored V35 script and fixed PLOT_DIR.")
