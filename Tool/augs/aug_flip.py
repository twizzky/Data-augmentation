import cv2
import os
from YOLOv5_UTILS import load_yolo_labels, save_yolo_labels

def flip(image, bboxes):
    flipped_img = cv2.flip(image, 1)
    h, w, _ = image.shape
    
    flipped_bboxes = []
    for bbox in bboxes:
        class_id, x_center, y_center, width, height = bbox
        new_x_center = 1 - x_center
        flipped_bboxes.append([class_id, new_x_center, y_center, width, height])
    
    return flipped_img, flipped_bboxes