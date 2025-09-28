import os
import pandas as pd

def merge_by_modality(in_dir="features", out_dir="features"):
    os.makedirs(out_dir, exist_ok=True)

    modalities = ["eeg", "ecg", "emg", "mov"]
    for mod in modalities:
        files = [f for f in os.listdir(in_dir) if f.endswith(".csv") and f"_{mod}_" in f]
        if not files:
            print(f"[WARNING] No files found for {mod}")
            continue

        dfs = []
        for f in sorted(files):  # sort keeps windows in order
            df = pd.read_csv(os.path.join(in_dir, f))
            df.insert(0, "source_file", f)   # track origin
            dfs.append(df)

        merged = pd.concat(dfs, ignore_index=True)
        out_file = os.path.join(out_dir, f"all_{mod}.csv")
        merged.to_csv(out_file, index=False)
        print(f"[INFO] Merged {len(files)} {mod.upper()} files → {out_file}")

# Example usage
if __name__ == "__main__":
    merge_by_modality("features")
