import cv2
import numpy as np

def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    factor = max(0.1, min(2.0, factor))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)