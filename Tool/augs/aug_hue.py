import cv2
import numpy as np

def adjust_hue(image: np.ndarray, delta: int) -> np.ndarray:
    delta = max(-90, min(90, delta))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + delta) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
