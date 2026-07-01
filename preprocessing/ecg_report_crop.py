import cv2
import os

def crop_ecg_report(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return image_path

    h, w = image.shape[:2]

    # Remove top patient details (about 18%)
    top = int(h * 0.18)

    # Remove bottom interpretation (about 15%)
    bottom = int(h * 0.85)

    # Keep almost full width
    left = int(w * 0.02)
    right = int(w * 0.98)

    cropped = image[top:bottom, left:right]

    output_path = os.path.join(
        os.path.dirname(image_path),
        "cropped_ecg.png"
    )

    cv2.imwrite(output_path, cropped)

    return output_path