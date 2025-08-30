# Epilepsy Prediction Pipeline

This project is an end-to-end pipeline for detecting and predicting epileptic seizures using EEG data.  

## Steps
1. Download EDF files from dataset
2. Extract EEG features using `mne`
3. Train ML model (RandomForest baseline, LSTM planned)
4. Evaluate seizure prediction accuracy

## Install Dependencies
```bash
pip install -r requirements.txt