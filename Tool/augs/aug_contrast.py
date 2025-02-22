import cv2
import numpy as np

def adjust_contrast(image: np.ndarray, alpha: float) -> np.ndarray:
    alpha = max(0.1, min(3.0, alpha))
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)
