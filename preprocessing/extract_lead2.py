import cv2


def extract_lead2(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise Exception("Image not found.")

    h, w = img.shape[:2]

    # Bottom rhythm strip (Lead II)
    y1 = int(h * 0.74)
    y2 = int(h * 0.97)

    x1 = int(w * 0.05)
    x2 = int(w * 0.98)

    lead2 = img[y1:y2, x1:x2]

    output_path = "lead2.png"

    cv2.imwrite(output_path, lead2)

    return output_path