import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt

# -----------------------
# Path to ECG CSV (one window or merged)
file_path = "features/all_ecg.csv"
# -----------------------

# Load ECG data
df = pd.read_csv(file_path)

# Pick first ECG channel (skip metadata columns if present)
ecg_channel = [col for col in df.columns if col not in ["source_file", "time_sec"]][0]
ecg_signal = df[ecg_channel].values[:5000]   # use first 5000 samples as an example

# Assume sampling rate (replace with actual from EDF metadata, e.g., 256 or 512 Hz)
sampling_rate = 256  

# -----------------------
# 1. ECG processing
# -----------------------
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)

# -----------------------
# 2. Extract HR & HRV features
# -----------------------
hr = signals["ECG_Rate"]          # heart rate (bpm)
hrv = nk.hrv_time(info, sampling_rate=sampling_rate)  # HRV time-domain metrics

# -----------------------
# 3. Plot ECG with detected R-peaks
# -----------------------
nk.ecg_plot(signals, info)
plt.show()

print("Heart Rate (first 10 values):")
print(hr.head())

print("\nHRV metrics:")
print(hrv)