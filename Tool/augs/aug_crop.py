import cv2
import numpy as np

def crop_image(image: np.ndarray, bboxes: list, size: float) -> tuple:
    h, w, _ = image.shape
    new_h, new_w = int(h * size), int(w * size)
    
    # If the crop dimensions are invalid, return the original image and bounding boxes.
    if new_h >= h or new_w >= w:
        return image, bboxes
    
    x_start = np.random.randint(0, w - new_w)
    y_start = np.random.randint(0, h - new_h)
    
    cropped_img = image[y_start:y_start + new_h, x_start:x_start + new_w]
    
    new_bboxes = []
    for bbox in bboxes:
        class_id, x_center, y_center, bw, bh = bbox
        abs_x = x_center * w
        abs_y = y_center * h
        
        # Keep bounding box only if its center lies within the cropped region
        if x_start <= abs_x <= x_start + new_w and y_start <= abs_y <= y_start + new_h:
            new_x = (abs_x - x_start) / new_w
            new_y = (abs_y - y_start) / new_h
            new_bboxes.append([class_id, new_x, new_y, bw, bh])
    
    return cropped_img, new_bboxes
