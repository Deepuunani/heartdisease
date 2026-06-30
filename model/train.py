import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from cnn_cbam import build_model

# ==========================
# Paths
# ==========================

DATASET_PATH = "dataset/processed"
MODEL_PATH = "saved_model"

os.makedirs(MODEL_PATH, exist_ok=True)

# ==========================
# Load Dataset
# ==========================

print("Loading Dataset...")

X_train = np.load(os.path.join(DATASET_PATH, "X_train.npy"))
X_val = np.load(os.path.join(DATASET_PATH, "X_val.npy"))
X_test = np.load(os.path.join(DATASET_PATH, "X_test.npy"))

y_train = np.load(os.path.join(DATASET_PATH, "y_train.npy"))
y_val = np.load(os.path.join(DATASET_PATH, "y_val.npy"))
y_test = np.load(os.path.join(DATASET_PATH, "y_test.npy"))

print("Dataset Loaded Successfully")

print("Train :", X_train.shape)
print("Validation :", X_val.shape)
print("Test :", X_test.shape)
# ==========================
# Compute Class Weights
# ==========================

print("\nComputing Class Weights...")

classes = np.unique(y_train)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(enumerate(class_weights))

print("Class Weights:")
for k, v in class_weights.items():
    print(f"Class {k} : {v:.4f}")

# ==========================
# Build Model
# ==========================

print("\nBuilding CNN + CBAM Model...")

model = build_model(
    input_shape=(360, 1),
    num_classes=7
)

model.summary()

# ==========================
# Callbacks
# ==========================

checkpoint = ModelCheckpoint(
    filepath=os.path.join(
        MODEL_PATH,
        "ecg_model.keras"
    ),
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

callbacks = [
    checkpoint,
    early_stop,
    reduce_lr
]

# ==========================
# Training Configuration
# ==========================

EPOCHS = 100
BATCH_SIZE = 128

print("\nTraining Started...")
# ==========================
# Train Model
# ==========================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

print("\nTraining Completed Successfully!")

# ==========================
# Evaluate Model
# ==========================

print("\nEvaluating Model...")

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"\nTest Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy*100:.2f}%")

# ==========================
# Predictions
# ==========================

y_pred_prob = model.predict(X_test)

y_pred = np.argmax(y_pred_prob, axis=1)

# ==========================
# Classification Report
# ==========================

class_names = [
    "Normal",
    "LBBB",
    "RBBB",
    "APB",
    "PVC",
    "Fusion",
    "Paced"
]

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=class_names
    )
)

# ==========================
# Confusion Matrix
# ==========================

cm = confusion_matrix(
    y_test,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

plt.figure(figsize=(8,8))

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig("images/confusion_matrix.png")

plt.close()

print("Confusion Matrix Saved")

# ==========================
# Accuracy Graph
# ==========================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["accuracy"],
    label="Train Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Training Accuracy")

plt.legend()

plt.grid(True)

plt.savefig("images/accuracy.png")

plt.close()

# ==========================
# Loss Graph
# ==========================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["loss"],
    label="Train Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training Loss")

plt.legend()

plt.grid(True)

plt.savefig("images/loss.png")

plt.close()

print("Graphs Saved")

print("\nBest Model Saved At:")

print("saved_model/ecg_model.keras")