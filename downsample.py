import pandas as pd
import os

# Input/output folder
FEATURE_DIR = "features"
OUTPUT_DIR = "features/downsampled"
os.makedirs(OUTPUT_DIR, exist_ok=True)

modalities = ["ecg", "eeg", "emg", "mov"]

for mod in modalities:
    in_file = os.path.join(FEATURE_DIR, f"all_{mod}.csv")
    out_file = os.path.join(OUTPUT_DIR, f"all_{mod}_downsampled.csv")
    
    if not os.path.exists(in_file):
        print(f"[WARNING] {in_file} not found, skipping.")
        continue
    
    print(f"[INFO] Processing {in_file} → {out_file}")
    
    # Load file
    df = pd.read_csv(in_file)
    
    # Take every 25th row
    df_downsampled = df.iloc[::25, :].reset_index(drop=True)
    
    # Save
    df_downsampled.to_csv(out_file, index=False)
    print(f"[INFO] Saved {len(df_downsampled)} rows to {out_file}")