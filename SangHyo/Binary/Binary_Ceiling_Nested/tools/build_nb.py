# Build an .ipynb from a cells file. Cell boundaries: lines starting with '# %%% CELL <n> [code|markdown]'.
# Prints the concatenated-code-cell sha256 (the pre-registration freeze hash; cell 13 recomputes it the same way).
import re, sys, hashlib, nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
src_path, out_path = sys.argv[1], sys.argv[2]
text = open(src_path, encoding="utf-8").read()
parts = re.split(r"^# %%% CELL (\d+) \[(code|markdown)\][^\n]*\n", text, flags=re.M)
assert parts[0].strip() == "", "text before first cell marker"
cells = []
for i in range(1, len(parts), 3):
    kind, body = parts[i+1], parts[i+2].strip("\n")
    cells.append(new_markdown_cell(body) if kind == "markdown" else new_code_cell(body))
nb = new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                          "language_info": {"name": "python"}, "colab": {"provenance": []}})
nbformat.validate(nb); nbformat.write(nb, out_path)
code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
print("cells:", len(cells), "| code cells:", sum(c.cell_type == 'code' for c in nb.cells), "| code sha256:", hashlib.sha256(code.encode("utf-8")).hexdigest())
