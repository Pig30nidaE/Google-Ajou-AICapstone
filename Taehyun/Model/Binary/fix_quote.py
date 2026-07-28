with open(r'c:\ML4\Model\Binary\V35_V26_Super_Recall_Optimization.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace("'confusion_matrix_v35_binary.png\"", '\"confusion_matrix_v35_binary.png\"')

with open(r'c:\ML4\Model\Binary\V35_V26_Super_Recall_Optimization.py', 'w', encoding='utf-8') as f:
    f.write(text)
