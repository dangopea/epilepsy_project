import os
import requests
import tempfile
import mne
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# --------------------------
# CONFIG
# --------------------------
BASE_URL = "https://openneuro.org/crn/datasets/ds005873/snapshots/1.1.0/files/"
FEATURE_DIR = "features"
os.makedirs(FEATURE_DIR, exist_ok=True)

# Example list of EEG files (replace/extend with more as needed)
edf_files = [
    "sub-001:ses-01:eeg:sub-001_ses-01_task-szMonitoring_run-01_eeg.edf",
    "sub-001:ses-01:eeg:sub-001_ses-01_task-szMonitoring_run-02_eeg.edf"
]

# --------------------------
# STEP 1: Download one EDF
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
# STEP 2: Extract Features
# --------------------------
def extract_features(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data, _ = raw[:, :]
    
    features = {}
    features["mean"] = np.mean(data, axis=1)
    features["var"] = np.var(data, axis=1)

    # Frequency bands
    psd, freqs = mne.time_frequency.psd_array_welch(
        data, sfreq=raw.info['sfreq'], n_fft=512
    )
    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12),
             "beta": (12, 30), "gamma": (30, 45)}
    for band, (low, high) in bands.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        features[band] = np.mean(psd[:, idx], axis=1)

    feat_vector = np.concatenate([features[k] for k in features.keys()])
    return feat_vector

# --------------------------
# MAIN LOOP
# --------------------------
def main():
    X, y = [], []
    for remote_file in edf_files:
        local_edf = download_temp_file(remote_file)
        try:
            feats = extract_features(local_edf)
            X.append(feats)
            y.append(0)  # TODO: use events.tsv to get true seizure labels
            out_file = os.path.join(FEATURE_DIR, os.path.basename(remote_file).replace(".edf", "_features.npy"))
            np.save(out_file, feats)
            print(f"[INFO] Saved features → {out_file}")
        finally:
            os.remove(local_edf)
            print(f"[INFO] Deleted temp file {local_edf}")

    print(f"[INFO] Processed {len(X)} files")

if __name__ == "__main__":
    main()