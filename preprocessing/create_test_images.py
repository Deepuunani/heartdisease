import os
import numpy as np
import matplotlib.pyplot as plt

# Create folder
os.makedirs("test_images", exist_ok=True)

# Load test data
X = np.load("dataset/processed/X_test.npy")
y = np.load("dataset/processed/y_test.npy")

labels = {
    0: "Normal",
    1: "LBBB",
    2: "RBBB",
    3: "APB",
    4: "PVC",
    5: "Fusion",
    6: "Paced"
}

count = {k: 0 for k in labels.keys()}

# Save 20 images per class
for i in range(len(X)):

    cls = int(y[i])

    if count[cls] >= 20:
        continue

    signal = X[i].reshape(-1)

    plt.figure(figsize=(8,2))
    plt.plot(signal, color="red", linewidth=2)
    plt.axis("off")

    filename = f"test_images/{labels[cls]}_{count[cls]+1}.png"

    plt.savefig(
        filename,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

    count[cls] += 1

    if all(v >= 20 for v in count.values()):
        break

print("Images Created Successfully")