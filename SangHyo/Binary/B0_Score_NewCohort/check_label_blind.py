"""CT6 — static label-blindness check of the sealed scorer notebook: no code cell may mention label/MMSE paths or tokens."""
import json, re, sys
nb = json.load(open(sys.argv[1], encoding="utf-8"))
code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
# tokens that would indicate reading labels or cognitive tests; the guard/blocklist *definitions* are allowed (they are strings in a blocklist)
banned = [r"LabelingData", r"3\.CognitiveFunction", r"DIAG_NM", r"SAMPLE_EMAIL", r"mmse_", r"train_mmse", r"val_mmse", r"training_label", r"val_label"]
hits = {b: len(re.findall(b, code)) for b in banned}
allowed_context = {"LabelingData": 1, "3\\.CognitiveFunction": 1}   # each may appear once, inside BLOCKED_PATH_TOKENS
viol = {b: n for b, n in hits.items() if n > allowed_context.get(b, 0)}
print("token counts:", hits)
print("CT6 label-blind:", "PASS" if not viol else f"FAIL {viol}")
sys.exit(0 if not viol else 1)
