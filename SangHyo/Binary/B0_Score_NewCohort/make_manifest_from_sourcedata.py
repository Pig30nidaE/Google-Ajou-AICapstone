"""Build a label-free manifest (sid list) from wearable SourceData only: sids present in BOTH activity and sleep exports.
Never opens LabelingData or 3.CognitiveFunction.  Usage: python make_manifest_from_sourcedata.py <data_root> <out_manifest.csv>"""
import sys, pandas as pd
from pathlib import Path
root, out = Path(sys.argv[1]), Path(sys.argv[2])
def sids(pattern):
    s = set()
    for p in root.rglob("*.csv"):
        low = str(p).lower()
        if pattern not in p.name.lower() or "labelingdata" in low or "3.cognitivefunction" in low:
            continue
        s |= set(pd.read_csv(p, usecols=["EMAIL"])["EMAIL"].astype(str).str.strip())
    return s
common = sorted(sids("activity") & sids("sleep"))
pd.DataFrame({"sid": common}).to_csv(out, index=False)
print(f"manifest: {len(common)} sids -> {out}")
