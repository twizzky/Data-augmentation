import cv2
import numpy as np

def add_noise(image, mean=0, var=30):
    def user_defined_noise(image, mean, var):
        row, col, ch = image.shape
        sigma = var ** 0.5
        noise = np.random.normal(mean, sigma, (row, col, ch))
        return np.clip(image + noise, 0, 255).astype(np.uint8)
    
    return user_defined_noise(image, mean, var)