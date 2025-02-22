import cv2
import numpy as np

def adjust_exposure(image: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(0.1, min(3.0, gamma))
    lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, lookup_table)
