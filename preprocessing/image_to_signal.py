import cv2
import numpy as np


# ---------------------------------------------------------
# Convert ECG image into 1D signal
# ---------------------------------------------------------

def image_to_signal(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise Exception("Image not found.")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur to remove noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Threshold
    _, mask = cv2.threshold(
        gray,
        180,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Remove tiny noise
    kernel = np.ones((2, 2), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    h, w = mask.shape

    signal = []

    # Extract waveform column by column
    for x in range(w):

        ys = np.where(mask[:, x] > 0)[0]

        if len(ys) == 0:
            signal.append(h // 2)

        else:
            signal.append(np.mean(ys))

    signal = np.array(signal, dtype=np.float32)

    # Invert image coordinates
    signal = h - signal

    # Remove baseline
    signal = signal - np.mean(signal)

    std = np.std(signal)

    if std == 0:
        std = 1

    signal = signal / std

    return signal


# ---------------------------------------------------------
# Resize signal to 360 samples
# ---------------------------------------------------------

def preprocess_image(image_path):

    signal = image_to_signal(image_path)

    x_old = np.linspace(
        0,
        1,
        len(signal)
    )

    x_new = np.linspace(
        0,
        1,
        360
    )

    signal = np.interp(
        x_new,
        x_old,
        signal
    )

    signal = signal.astype(np.float32)

    signal = signal.reshape(
        1,
        360,
        1
    )

    return signal


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    sample = preprocess_image(
        "test_images/APB_1.png"
    )

    print(sample.shape)