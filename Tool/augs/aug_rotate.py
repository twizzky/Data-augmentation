import cv2 
import math
import os
import numpy as np
from YOLOv5_UTILS import load_yolo_labels, save_yolo_labels

def rotate_image(image, bboxes, angle):
    h, w, _ = image.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(image, M, (w, h))
    
    rotated_bboxes = []
    for bbox in bboxes:
        class_id, x_center, y_center, box_w, box_h = bbox
        
        cx = x_center * w
        cy = y_center * h
        bw = box_w * w
        bh = box_h * h
        
        tl = [cx - bw / 2, cy - bh / 2]  
        tr = [cx + bw / 2, cy - bh / 2] 
        br = [cx + bw / 2, cy + bh / 2]  
        bl = [cx - bw / 2, cy + bh / 2]  
        corners = np.array([tl, tr, br, bl])  
        
        ones = np.ones((corners.shape[0], 1))
        corners_hom = np.hstack([corners, ones])  
        
        rotated_corners = np.dot(corners_hom, M.T)  
        
        x_min = np.min(rotated_corners[:, 0])
        y_min = np.min(rotated_corners[:, 1])
        x_max = np.max(rotated_corners[:, 0])
        y_max = np.max(rotated_corners[:, 1])
        
        new_cx = (x_min + x_max) / 2.0
        new_cy = (y_min + y_max) / 2.0
        new_bw = x_max - x_min
        new_bh = y_max - y_min
        
        new_bbox = [class_id, new_cx / w, new_cy / h, new_bw / w, new_bh / h]
        rotated_bboxes.append(new_bbox)
        
    return rotated_img, rotated_bboxes
