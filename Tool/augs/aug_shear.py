import cv2
import numpy as np

def shear_image(image, bboxes, shear_range=0.2):
    rows, cols, ch = image.shape
    shear_factor = np.random.uniform(-shear_range, shear_range)
    M = np.array([[1, shear_factor, 0],[0, 1, 0]], dtype=np.float32)
    sheared_img = cv2.warpAffine(image, M, (cols, rows))
    
    sheared_bboxes = []
    for bbox in bboxes:
        sheared_bbox = shear_bbox(bbox, M, cols, rows)
        sheared_bboxes.append(sheared_bbox)
    
    return sheared_img, sheared_bboxes

def shear_bbox(bbox, M, img_width, img_height):
    class_id, x_center, y_center, bw, bh = bbox
    cx = x_center * img_width
    cy = y_center * img_height
    box_w = bw * img_width
    box_h = bh * img_height
    
    tl = [cx - box_w / 2, cy - box_h / 2]
    tr = [cx + box_w / 2, cy - box_h / 2] 
    br = [cx + box_w / 2, cy + box_h / 2]  
    bl = [cx - box_w / 2, cy + box_h / 2] 
    corners = np.array([tl, tr, br, bl])   
    
    ones = np.ones((corners.shape[0], 1))
    corners_hom = np.hstack([corners, ones]) 
    
    transformed_corners = np.dot(corners_hom, M.T)  
    
    x_min = np.min(transformed_corners[:, 0])
    y_min = np.min(transformed_corners[:, 1])
    x_max = np.max(transformed_corners[:, 0])
    y_max = np.max(transformed_corners[:, 1])
    

    new_cx = (x_min + x_max) / 2.0 / img_width
    new_cy = (y_min + y_max) / 2.0 / img_height
    new_w = (x_max - x_min) / img_width
    new_h = (y_max - y_min) / img_height
    
    return [class_id, new_cx, new_cy, new_w, new_h]
