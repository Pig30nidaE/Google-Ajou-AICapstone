import json
def split_code(code_text):
    lines = code_text.split('\n')
    state = 0
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
    print([len(b) for b in blocks])

with open(r'c:\ML4\Model\Binary\V29_Binary_Optuna_Stacking.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
    # the giant cell is probably split already, wait, it's cell 3!
    code_text = "".join(nb['cells'][3]['source'])
    split_code(code_text)
