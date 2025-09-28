import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Path to your merged ECG file
file_path = "features/all_ecg.csv"
# -----------------------

# Load CSV
df = pd.read_csv(file_path)

# Skip "source_file" column if present, time is usually second column
# Find the first ECG channel automatically
ecg_channel = [col for col in df.columns if col not in ["source_file", "time_sec"]][0]

# Take first 100 samples
ecg_data = df[ecg_channel].iloc[:100]

# Plot
plt.figure(figsize=(10,4))
plt.plot(range(100), ecg_data, marker="o")
plt.title(f"ECG Signal - First 100 Samples ({ecg_channel})")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude (V or µV)")
plt.grid(True)
plt.show()
