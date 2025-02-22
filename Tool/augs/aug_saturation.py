import cv2
import numpy as np

def adjust_saturation(image, scale=1.5):
    def user_defined_saturation(image, scale):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return user_defined_saturation(image, scale)
