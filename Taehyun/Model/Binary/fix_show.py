import glob
for py_file in glob.glob(r'c:\ML4\Model\Binary\*.py'):
    with open(py_file, 'r', encoding='utf-8') as f:
        text = f.read()
    if 'plt.show()' in text:
        text = text.replace('plt.show()', 'plt.show()')
        with open(py_file, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Replaced plt.show in {py_file}")
