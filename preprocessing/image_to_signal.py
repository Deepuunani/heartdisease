import cv2
import numpy as np


def image_to_signal(image_path):

    # Read image
    img = cv2.imread(image_path)

    if img is None:
        raise Exception("Image not found.")

    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Extract red waveform
    lower_red = np.array([150, 0, 0])
    upper_red = np.array([255, 120, 120])

    mask = cv2.inRange(img, lower_red, upper_red)

    h, w = mask.shape

    signal = []

    # Scan every column
    for x in range(w):

        ys = np.where(mask[:, x] > 0)[0]

        if len(ys) == 0:
            signal.append(h // 2)

        else:
            signal.append(np.mean(ys))

    signal = np.array(signal, dtype=np.float32)

    # Invert because image y-axis is downward
    signal = h - signal

    # Normalize
    signal = signal - np.mean(signal)
    signal = signal / np.std(signal)

    return signal
# -----------------------------------------
# Convert waveform to 360 samples
# -----------------------------------------

def preprocess_image(image_path):

    signal = image_to_signal(image_path)

    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, 360)

    signal = np.interp(
        x_new,
        x_old,
        signal
    )

    signal = signal.astype(np.float32)

    signal = signal.reshape(1, 360, 1)

    return signal


# -----------------------------------------
# Testing
# -----------------------------------------

if __name__ == "__main__":

    sample = preprocess_image("test_images/APB_1.png")

    print(sample.shape)
    