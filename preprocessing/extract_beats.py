import os
import wfdb
import numpy as np
from tqdm import tqdm

# Dataset path
DATASET_PATH = "dataset/mitdb"

# Output path
OUTPUT_PATH = "dataset/processed"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Beat classes
BEAT_CLASSES = {
    "N": 0,
    "L": 1,
    "R": 2,
    "A": 3,
    "V": 4,
    "F": 5,
    "/": 6
}

WINDOW = 180

X = []
y = []

records = sorted([
    file.replace(".dat", "")
    for file in os.listdir(DATASET_PATH)
    if file.endswith(".dat")
])

print(f"\nFound {len(records)} ECG Records\n")

for record_name in tqdm(records):

    record = wfdb.rdrecord(os.path.join(DATASET_PATH, record_name))
    annotation = wfdb.rdann(os.path.join(DATASET_PATH, record_name), "atr")

    signal = record.p_signal[:, 0]

    for sample, symbol in zip(annotation.sample, annotation.symbol):

        if symbol not in BEAT_CLASSES:
            continue

        if sample - WINDOW < 0:
            continue

        if sample + WINDOW >= len(signal):
            continue

        beat = signal[sample-WINDOW:sample+WINDOW]

        # Normalize
        beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)

        X.append(beat)
        y.append(BEAT_CLASSES[symbol])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print("\nDataset Created")
print("------------------------")
print("Total Beats :", len(X))
print("Beat Shape  :", X.shape)
print("Labels      :", y.shape)

np.save(os.path.join(OUTPUT_PATH, "X.npy"), X)
np.save(os.path.join(OUTPUT_PATH, "y.npy"), y)

print("\nSaved Successfully")
print("dataset/processed/X.npy")
print("dataset/processed/y.npy")