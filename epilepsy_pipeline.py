import os
import requests
import tempfile
import mne
import pandas as pd
import shutil

# --------------------------
# CONFIG
# --------------------------
DATASET = "ds005873"
VERSION = "1.1.0"
BASE_URL = f"https://openneuro.org/crn/datasets/{DATASET}/snapshots/{VERSION}/files/"
OUT_DIR = "features"
WINDOW_SIZE = 30  # seconds
os.makedirs(OUT_DIR, exist_ok=True)

# Example EDFs across modalities (EEG, ECG, EMG, MOV)
edf_files = [
    "sub-001:ses-01:eeg:sub-001_ses-01_task-szMonitoring_run-01_eeg.edf",
    "sub-001:ses-01:ecg:sub-001_ses-01_task-szMonitoring_run-01_ecg.edf",
    "sub-001:ses-01:emg:sub-001_ses-01_task-szMonitoring_run-01_emg.edf",
    "sub-001:ses-01:mov:sub-001_ses-01_task-szMonitoring_run-01_mov.edf"
]

# --------------------------
# STEP 0: Clean old CSVs
# --------------------------
def clean_old_csvs(out_dir=OUT_DIR):
    if os.path.exists(out_dir):
        for f in os.listdir(out_dir):
            if f.endswith(".csv"):
                os.remove(os.path.join(out_dir, f))
        print(f"[INFO] Cleaned old CSVs in {out_dir}")
    else:
        os.makedirs(out_dir)

# --------------------------
# STEP 1: Download EDF temporarily
# --------------------------
def download_temp_file(remote_path):
    url = BASE_URL + remote_path
    local = tempfile.NamedTemporaryFile(delete=False, suffix=".edf")
    print(f"[INFO] Downloading {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local.name, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
    return local.name

# --------------------------
# STEP 2: Split into windows + Save CSVs
# --------------------------
def save_windows_csv(edf_path, remote_file, out_dir=OUT_DIR, win_size=WINDOW_SIZE):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data, times = raw[:, :]  # shape = (n_channels, n_samples)
    sfreq = raw.info['sfreq']  # sampling frequency

    samples_per_win = int(sfreq * win_size)
    total_samples = data.shape[1]
    n_windows = total_samples // samples_per_win

    safe_base = os.path.basename(remote_file).replace(":", "_").replace(".edf", "")
    out_files = []

    for w in range(n_windows):
        start = w * samples_per_win
        end = start + samples_per_win
        segment = data[:, start:end].T
        t_segment = times[start:end]

        df = pd.DataFrame(segment, columns=raw.ch_names)
        df.insert(0, "time_sec", t_segment)

        out_file = os.path.join(out_dir, f"{safe_base}_window{w+1:04d}.csv")
        df.to_csv(out_file, index=False)
        out_files.append(out_file)

    print(f"[INFO] Saved {len(out_files)} windows for {remote_file}")
    return out_files

# --------------------------
# MAIN
# --------------------------
def main():
    clean_old_csvs(OUT_DIR)

    for remote_file in edf_files:
        local_edf = download_temp_file(remote_file)
        try:
            save_windows_csv(local_edf, remote_file, OUT_DIR, WINDOW_SIZE)
        except Exception as e:
            print(f"[ERROR] Failed on {remote_file}: {e}")
        finally:
            os.remove(local_edf)
            print(f"[INFO] Deleted temp file {local_edf}")

if __name__ == "__main__":
    main()