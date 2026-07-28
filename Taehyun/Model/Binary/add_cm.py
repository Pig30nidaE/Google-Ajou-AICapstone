import json
import os
import re

def add_cm_plot(file_path, version_name):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    cm_code = f"""
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    class_names = ['Normal(CN)', 'Abnormal']
    for ax, (m, r) in zip(axes, results.items()):
        sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={{"size": 14}}, ax=ax)
        ax.set_title(f"[{{m}}]\\nAUC: {{r['auc']:.4f}} | Acc: {{r['acc']:.4f}}", fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix_{version_name}.png", dpi=150)
    plt.show()
"""

    changed = False
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = "".join(cell.get('source', []))
            if "def plot_" in source and "roc_curve" in source and "Confusion Matrix" not in source:
                # Find the place to insert it (right before ROC curves)
                idx = source.find("    plt.figure(figsize=(9, 7))")
                if idx != -1:
                    new_source = source[:idx] + "    # Confusion Matrix\n" + cm_code + "\n" + source[idx:]
                    cell['source'] = [line + '\n' for line in new_source.split('\n')]
                    if cell['source']:
                        cell['source'][-1] = cell['source'][-1].rstrip('\n')
                    changed = True

    if changed:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(f"Added CM plot to {os.path.basename(file_path)}")

add_cm_plot(r'c:\ML4\Model\Binary\V29_Binary_Optuna_Stacking.ipynb', "v29_binary")
add_cm_plot(r'c:\ML4\Model\Binary\V41_Binary_MMSE_Based.ipynb', "v41_mmse")
