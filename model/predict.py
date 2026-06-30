import os
import numpy as np
import tensorflow as tf

# ----------------------------
# Load Model
# ----------------------------

MODEL_PATH = "saved_model/ecg_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

print("Model Loaded Successfully!")

# ----------------------------
# Load Test Dataset
# ----------------------------

DATASET_PATH = "dataset/processed"

X_test = np.load(os.path.join(DATASET_PATH, "X_test.npy"))
y_test = np.load(os.path.join(DATASET_PATH, "y_test.npy"))

print("Test Dataset Loaded")

# ----------------------------
# Select Sample
# ----------------------------

sample_index = 100

sample = X_test[sample_index]
true_label = y_test[sample_index]

sample = np.expand_dims(sample, axis=0)

# ----------------------------
# Prediction
# ----------------------------

prediction = model.predict(sample)

predicted_class = np.argmax(prediction)

confidence = np.max(prediction) * 100

# ----------------------------
# Labels
# ----------------------------

labels = {
    0: "Normal Beat",
    1: "Left Bundle Branch Block",
    2: "Right Bundle Branch Block",
    3: "Atrial Premature Beat",
    4: "Premature Ventricular Contraction",
    5: "Fusion Beat",
    6: "Paced Beat"
}

print("\n===============================")
print("ECG Prediction Result")
print("===============================")

print("Actual Label    :", labels[true_label])
print("Predicted Label :", labels[predicted_class])
print("Confidence      : {:.2f}%".format(confidence))

if predicted_class == true_label:
    print("\nPrediction Correct")
else:
    print("\nPrediction Incorrect")