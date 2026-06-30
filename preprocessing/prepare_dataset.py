import os
import numpy as np
from sklearn.model_selection import train_test_split

# Paths
DATASET_PATH = "dataset/processed"

# Load data
X = np.load(os.path.join(DATASET_PATH, "X.npy"))
y = np.load(os.path.join(DATASET_PATH, "y.npy"))

print("Dataset Loaded")
print("X Shape:", X.shape)
print("y Shape:", y.shape)

# Show class distribution
print("\nClass Distribution")
unique, counts = np.unique(y, return_counts=True)

class_names = {
    0: "Normal",
    1: "LBBB",
    2: "RBBB",
    3: "APB",
    4: "PVC",
    5: "Fusion",
    6: "Paced"
}

for u, c in zip(unique, counts):
    print(f"{class_names[u]:10s}: {c}")

# Reshape for CNN
X = X.reshape((-1, 360, 1))

# Train 80%
# Temp 20%
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Validation 10%
# Test 10%
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDataset Split")
print("-----------------------")
print("Train :", X_train.shape)
print("Validation :", X_val.shape)
print("Test :", X_test.shape)

# Save
np.save(os.path.join(DATASET_PATH, "X_train.npy"), X_train)
np.save(os.path.join(DATASET_PATH, "X_val.npy"), X_val)
np.save(os.path.join(DATASET_PATH, "X_test.npy"), X_test)

np.save(os.path.join(DATASET_PATH, "y_train.npy"), y_train)
np.save(os.path.join(DATASET_PATH, "y_val.npy"), y_val)
np.save(os.path.join(DATASET_PATH, "y_test.npy"), y_test)

print("\nDataset Saved Successfully")