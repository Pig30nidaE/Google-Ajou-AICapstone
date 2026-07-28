import json
import os

file_path = r'c:\ML4\Model\Binary\V35_V26_Super_Recall_Optimization.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the last cell and append plot code
changed = False
for cell in reversed(nb.get('cells', [])):
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        if "best_overall['thresh']" in source and "seaborn" not in source:
            cm_code = """
    # Confusion Matrix Visualization
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    
    # Recreate the best predictions
    y_true, p_lgb, p_cat, p_xgb, p_rf = get_v26_oof_predictions(df, ranked_feats[:best_overall['k']])
    w1, w2, w3, w4 = best_overall['weights']
    ens_p = w1*p_lgb + w2*p_cat + w3*p_xgb + w4*p_rf
    preds = np.where(ens_p >= best_overall['thresh'], 1, 0)
    
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(6, 5))
    class_names = ['Normal(CN)', 'Abnormal']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 16})
    plt.title(f"[V35] Ultra-Recall Optimized\\nAUC: {best_overall['auc']:.4f} | Acc: {best_overall['acc']:.4f}", fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix_v35_binary.png", dpi=150)
    plt.show()
"""
            cell['source'] = [line + '\n' for line in (source + cm_code).split('\n')]
            changed = True
            break

if changed:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("Added CM plot to V35!")
else:
    print("Could not find the injection point in V35.")
