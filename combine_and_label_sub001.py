import os
import re
import pandas as pd

# -----------------------
# Paths
# -----------------------
FEATURE_DIR = "features"
DOWNSAMPLED_DIR = os.path.join(FEATURE_DIR, "downsampled")

EEG_CSV = os.path.join(DOWNSAMPLED_DIR, "all_eeg_downsampled.csv")
ECG_CSV = os.path.join(DOWNSAMPLED_DIR, "all_ecg_downsampled.csv")
EMG_CSV = os.path.join(DOWNSAMPLED_DIR, "all_emg_downsampled.csv")
MOV_CSV = os.path.join(DOWNSAMPLED_DIR, "all_mov_downsampled.csv")

EVENTS_CSV = os.path.join(FEATURE_DIR, "sub001_events_combined.csv")
OUT_CSV = os.path.join(FEATURE_DIR, "unified_downsampled_labeled_sub001.csv")

# -----------------------
# Helpers
# -----------------------
def parse_run_key(source_file: str):
    """
    Extract 'sub-001_ses-01_run-01' from a window filename stored in source_file.
    Example:
      sub-001_ses-01_ecg_sub-001_ses-01_task-szMonitoring_run-01_ecg_window0001.csv
    """
    base = str(source_file)
    base = re.sub(r"\.csv$", "", base)
    base = re.sub(r"_window\d{4}$", "", base)
    idx = base.rfind("_sub-")
    if idx == -1:
        return None
    run_stub = base[idx + 1:]  # sub-001_ses-01_task-..._run-XX_<mod>
    m_run = re.search(r"(sub-\d{3})_(ses-\d{2}).*?(run-\d{2})", run_stub)
    if not m_run:
        return None
    return f"{m_run.group(1)}_{m_run.group(2)}_{m_run.group(3)}"

def load_modality(path: str, modality: str):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)

    # keep only sub-001 rows
    if "source_file" not in df.columns or "time_sec" not in df.columns:
        raise ValueError(f"{path} must contain columns: source_file, time_sec")

    df = df[df["source_file"].astype(str).str.startswith("sub-001_")].copy()
    if df.empty:
        return None

    # derive run_key
    df["run_key"] = df["source_file"].apply(parse_run_key)
    df = df.dropna(subset=["run_key"]).copy()

    # ensure numeric time
    df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
    df = df.dropna(subset=["time_sec"])

    # rename source_file → <mod>_source_file to avoid merge suffix clashes
    df = df.rename(columns={"source_file": f"{modality}_source_file"})

    # prefix signal columns (exclude metadata)
    meta = {f"{modality}_source_file", "time_sec", "run_key"}
    signal_cols = [c for c in df.columns if c not in meta]
    # but some meta from original file may still be around; limit to non-shared columns
    signal_cols = [c for c in signal_cols if c not in {"source_file", "remote_events_path"}]

    # add modality prefix to signal cols (so channels don't collide across modalities)
    rename_map = {c: f"{modality}_{c}" for c in signal_cols}
    df = df.rename(columns=rename_map)

    keep = ["run_key", "time_sec", f"{modality}_source_file"] + list(rename_map.values())
    return df[keep]

def label_with_events(df_merged: pd.DataFrame, events_csv: str) -> pd.DataFrame:
    ev = pd.read_csv(events_csv)
    ev["is_seizure"] = ev["eventType"].astype(str).str.lower().str.startswith("sz")

    seiz = ev[ev["is_seizure"]].copy()
    if seiz.empty:
        df_merged["label"] = 0
        return df_merged

    seiz["start"] = pd.to_numeric(seiz["onset"], errors="coerce")
    seiz["end"] = pd.to_numeric(seiz["onset"], errors="coerce") + pd.to_numeric(seiz["duration"], errors="coerce")
    intervals = seiz[["start", "end"]].dropna().to_numpy()

    def label_time(t):
        for s, e in intervals:
            if s <= t < e:
                return 1
        return 0

    df_merged["label"] = df_merged["time_sec"].astype(float).apply(label_time)
    return df_merged

# -----------------------
# Main
# -----------------------
def main():
    dfs = []
    for path, mod in [(EEG_CSV, "eeg"), (ECG_CSV, "ecg"), (EMG_CSV, "emg"), (MOV_CSV, "mov")]:
        dfm = load_modality(path, mod)
        if dfm is not None:
            dfs.append(dfm)

    if not dfs:
        raise SystemExit("No sub-001 rows found in downsampled files. Check paths/content.")

    # outer merge on (run_key, time_sec)
    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, on=["run_key", "time_sec"], how="outer")

    merged = merged.sort_values(["run_key", "time_sec"]).reset_index(drop=True)

    if not os.path.exists(EVENTS_CSV):
        raise SystemExit(f"Events file not found: {EVENTS_CSV}")

    labeled = label_with_events(merged, EVENTS_CSV)

    # nice column order
    front = ["run_key", "time_sec", "label"]
    # Add modality-specific source_files if present
    for mod in ["eeg", "ecg", "emg", "mov"]:
        col = f"{mod}_source_file"
        if col in labeled.columns:
            front.append(col)
    others = [c for c in labeled.columns if c not in front]
    labeled = labeled[front + others]

    os.makedirs(FEATURE_DIR, exist_ok=True)
    labeled.to_csv(OUT_CSV, index=False)

    counts = labeled["label"].value_counts(dropna=False).to_dict()
    print(f"[INFO] wrote {OUT_CSV}")
    print(f"[INFO] label balance: {counts}")
    print(labeled.head(8))

if __name__ == "__main__":
    main()