import glob
import os

folder = r'c:\ML4\Model\Binary'
for py_file in glob.glob(os.path.join(folder, '*.py')):
    with open(py_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    if 'plt.show()' in text:
        text = text.replace('plt.show()', 'plt.show()')
        with open(py_file, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Replaced plt.show() with plt.show() in {os.path.basename(py_file)}")
