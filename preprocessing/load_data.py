import os
import wfdb
import matplotlib.pyplot as plt

# Dataset path
DATASET_PATH = os.path.join("dataset", "mitdb")

# ECG record to load
record_name = "100"

record_path = os.path.join(DATASET_PATH, record_name)

# Read ECG signal
record = wfdb.rdrecord(record_path)

# Read annotations
annotation = wfdb.rdann(record_path, "atr")

print("=" * 50)
print("Record Loaded Successfully")
print("=" * 50)

print(f"Record Name      : {record_name}")
print(f"Sampling Rate    : {record.fs} Hz")
print(f"Signal Shape     : {record.p_signal.shape}")
print(f"Signal Names     : {record.sig_name}")
print(f"Total Annotations: {len(annotation.sample)}")

# Plot first 2000 samples
plt.figure(figsize=(15,5))
plt.plot(record.p_signal[:2000,0], color="blue")
plt.title("ECG Signal - Record 100")
plt.xlabel("Samples")
plt.ylabel("Amplitude (mV)")
plt.grid(True)
plt.tight_layout()
plt.savefig("images/ecg_sample.png")
print("ECG graph saved successfully.")
# plt.show()

# Display first 20 annotations
print("\nFirst 20 Beat Labels")
print("-" * 30)

for sample, symbol in zip(annotation.sample[:20], annotation.symbol[:20]):
    print(f"Sample: {sample:<8} Label: {symbol}")