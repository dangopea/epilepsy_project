import pandas as pd

IN = "features/unified_downsampled_labeled_sub001.csv"

def contiguous_blocks(df):
    # df must be filtered to a single run_key
    df = df.sort_values("time_sec").reset_index(drop=True)
    blocks = []
    in_block = False
    start = None
    for i, row in df.iterrows():
        if row["label"] == 1 and not in_block:
            in_block = True
            start = row["time_sec"]
        elif row["label"] == 0 and in_block:
            in_block = False
            end = df.loc[i-1, "time_sec"]
            blocks.append((start, end))
    if in_block and len(df) > 0:
        blocks.append((start, df.iloc[-1]["time_sec"]))
    return blocks

def main():
    u = pd.read_csv(IN)
    u["time_sec"] = pd.to_numeric(u["time_sec"], errors="coerce")
    u = u.dropna(subset=["time_sec"])
    pos = u[u["label"] == 1]
    if pos.empty:
        print("[INFO] No positive labels found.")
        return

    print("[INFO] Positive label counts per run:")
    print(pos.groupby("run_key")["time_sec"].agg(["count","min","max"]))

    print("\n[INFO] Contiguous seizure blocks per run (approx ranges):")
    for run, g in u.groupby("run_key"):
        g = g[["time_sec", "label"]]
        blocks = contiguous_blocks(g)
        if blocks:
            for s, e in blocks:
                print(f"{run}: [{s:.3f}, {e:.3f}] (≈{int((e-s)*10)/10}s)")
    print("\nDone.")

if __name__ == "__main__":
    main()