# add_decision_to_events.py
import pandas as pd
import sys

in_path  = "features/sub001_events_combined.csv"   # adjust if your file lives elsewhere
out_path = "features/sub001_events_with_labels.csv"

if len(sys.argv) > 1: in_path  = sys.argv[1]
if len(sys.argv) > 2: out_path = sys.argv[2]

df = pd.read_csv(in_path)
df["decision"] = df["eventType"].astype(str).str.lower().str.startswith("sz").astype(int)

df.to_csv(out_path, index=False)
print(f"[INFO] wrote {out_path}")
print(df["decision"].value_counts(dropna=False).to_dict())