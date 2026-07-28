import json

file_path = r'c:\ML4\Model\Binary\V35_V26_Super_Recall_Optimization.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = []
for cell in nb['cells']:
    if not new_cells:
        new_cells.append(cell)
    else:
        if cell['cell_type'] == 'code' and new_cells[-1]['cell_type'] == 'code':
            if "".join(cell.get('source', [])) == "".join(new_cells[-1].get('source', [])):
                print("Found duplicate cell, removing it.")
                continue
        new_cells.append(cell)

if len(new_cells) < len(nb['cells']):
    nb['cells'] = new_cells
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("Fixed V35_V26_Super_Recall_Optimization.ipynb")
