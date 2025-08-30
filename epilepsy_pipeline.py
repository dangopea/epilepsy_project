import os
import requests
from tqdm import tqdm
import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------
# CONFIG
# --------------------------
BASE_URL = "https://biomedepi.github.io/seizure_detection_challenge/dataset/"
DOWNLOAD_DIR = "edf_files"
FEATURE_DIR = "features"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)

# Example dataset (replace with full list from dataset site)
edf_files = [
    "subject1.edf",
    "subject2.edf",
    "subject3.edf"
]

# --------------------------
# STEP 1: Download EDF files
# --------------------------
def download_file(url, folder):
    local_path = os.path.join(folder, os.path.basename(url))
    if os.path.exists(local_path):
        print(f"[SKIP] Already downloaded {local_path}")
        return local_path

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f, tqdm(
            total=int(r.headers.get("content-length", 0)),
            unit="B", unit_scale=True, desc=os.path.basename(url)
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    return local_path

# --------------------------
# STEP 2: Extract Features
# --------------------------
def extract_features(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data, _ = raw[:, :]
    
    features = {}
    features["mean"] = np.mean(data, axis=1)
    features["var"] = np.var(data, axis=1)

    # Frequency bands (delta, theta, alpha, beta, gamma)
    psd, freqs = mne.time_frequency.psd_array_welch(
        data, sfreq=raw.info['sfreq'], n_fft=512
    )
    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12),
             "beta": (12, 30), "gamma": (30, 45)}
    for band, (low, high) in bands.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        features[band] = np.mean(psd[:, idx], axis=1)

    # Flatten into 1D vector
    feat_vector = np.concatenate([features[k] for k in features.keys()])
    return feat_vector

def process_and_save_features():
    X, y = [], []
    for file in edf_files:
        url = BASE_URL + file
        local_path = download_file(url, DOWNLOAD_DIR)

        try:
            feats = extract_features(local_path)
            label = 1 if "seizure" in file.lower() else 0  # simple heuristic
            X.append(feats)
            y.append(label)
            np.save(os.path.join(FEATURE_DIR, file.replace(".edf", "_features.npy")), feats)
            print(f"[INFO] Extracted features from {file}")
        except Exception as e:
            print(f"[ERROR] Feature extraction failed for {file}: {e}")
    return np.array(X), np.array(y)

# --------------------------
# STEP 3: Train + Evaluate Model
# --------------------------
def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

# --------------------------
# MAIN
# --------------------------
def main():
    X, y = process_and_save_features()
    print(f"[INFO] Feature matrix shape: {X.shape}, Labels: {y.shape}")
    if len(np.unique(y)) > 1:  # ensure at least 2 classes
        train_and_evaluate(X, y)
    else:
        print("[WARNING] Only one class present in dataset subset!")

if __name__ == "__main__":
    main()