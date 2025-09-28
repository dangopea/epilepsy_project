# Epilepsy Project – SeizeIT2 (OpenNeuro ds005873 v1.1.0)

This repository contains scripts for processing the **SeizeIT2 dataset** (OpenNeuro ID: ds005873, version 1.1.0), a large-scale wearable EEG study for seizure detection and prediction. The project aims to move beyond post-onset detection toward **prediction and proactive epilepsy management**.

---

## 📦 Dataset

- **Name**: SeizeIT2  
- **Source**: [OpenNeuro ds005873 v1.1.0](https://openneuro.org/datasets/ds005873/versions/1.1.0)  
- **Format**: [BIDS](https://bids.neuroimaging.io/)  
- **Subjects**: 125 patients with focal epilepsy, across 5 European Epilepsy Monitoring Units  
- **Duration**: ~11,600 hours of data (~44 GB EDF files)  
- **Modalities**:
  - **EEG** (behind-the-ear, bte-EEG)  
  - **ECG**  
  - **EMG**  
  - **MOV** (motion sensors)  

### Events
- Each EEG run is paired with an `*_events.tsv` file.  
- Contains annotated seizure events with `onset`, `duration`, and `trial_type`.  
- These are the **ground-truth labels** used for seizure detection/forecasting.  

---

## ⚙️ Pipeline

### 1. Raw data → CSV
- **`epilepsy_pipeline.py`**  
  Downloads EDF files from OpenNeuro → splits into **30s windows** → saves each window as CSV.  
  Handles all four modalities: EEG, ECG, EMG, MOV.

### 2. Merge CSVs
- **`combo.py`**  
  Combines window CSVs into one big file per modality:  
  - `features/all_eeg.csv`  
  - `features/all_ecg.csv`  
  - `features/all_emg.csv`  
  - `features/all_mov.csv`  

Adds a `source_file` column to track origin.

### 3. Downsample
- **`downsample.py`**  
  Reduces size by keeping every 25th row.  
  Output stored in `features/downsampled/`.

### 4. Visualization
- **`plot_ecg.py`**  
  Plots raw ECG waveform (optionally FFT) from merged or downsampled files.

### 5. Feature Extraction
- **`hr_hrv_extraction.py`**  
  Extracts **Heart Rate (HR)** and **Heart Rate Variability (HRV)** from ECG using [NeuroKit2](https://neuropsychology.github.io/NeuroKit/).  
  ⚠️ Requires Python ≥ 3.10.

---

## 🧪 Next Steps

1. **Event Mapping**  
   - Use EEG `*_events.tsv` to label each 30s window as:  
     - `1` → seizure overlaps window  
     - `0` → no seizure  
   - Also compute `time_to_next_seizure_sec` for forecasting tasks.  
   - Output: `features/window_labels.csv`.

2. **Modeling**  
   - **Detection**: classify seizure vs. non-seizure windows.  
   - **Forecasting**: predict seizure risk horizon (e.g. <1h, 1–4h, >4h).  
   - Approaches: classical ML (ensemble models on features) and DL (CNN/LSTM on raw).  

---

## 🛠️ Installation

### Using pip
```bash
# Recommended: Python 3.11+
pip install -r requirements.txt

### Using conda
conda create -n epilepsy python=3.11
conda activate epilepsy
pip install -r requirements.txt


