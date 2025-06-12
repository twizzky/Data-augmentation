import cv2
import numpy as np

def apply_mosaic(images, bboxes_list, img_size=640):
    if len(images) < 4:
        raise ValueError("Mosaic requires at least 4 images")
    
    if len(images) != 4:
        raise ValueError("Mosaic requires exactly 4 images")
    
    mosaic_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    cut_x = img_size // 2
    cut_y = img_size // 2
    new_bboxes = []
    
    for i, (img, bboxes) in enumerate(zip(images, bboxes_list)):
        resized_img = cv2.resize(img, (cut_x, cut_y))
        x_offset = (i % 2) * cut_x
        y_offset = (i // 2) * cut_y
        mosaic_img[y_offset:y_offset + cut_y, x_offset:x_offset + cut_x] = resized_img
        
        for box in bboxes:
            class_id, x_center, y_center, width, height = box
            x_resized = x_center * cut_x
            y_resized = y_center * cut_y
            abs_x = x_resized + x_offset
            abs_y = y_resized + y_offset
            new_x = abs_x / img_size
            new_y = abs_y / img_size
            new_width = width * (cut_x / img_size)
            new_height = height * (cut_y / img_size)
            new_bboxes.append([class_id, new_x, new_y, new_width, new_height])
    
    return mosaic_img, new_bboxes
