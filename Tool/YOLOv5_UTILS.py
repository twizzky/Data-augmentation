import os
import logging
from typing import List

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

def load_yolo_labels(label_path: str) -> List[List[float]]:
    boxes = []
    if not os.path.exists(label_path):
        logging.warning(f"Label file not found - {label_path}")
        return boxes
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                parts = line.strip().split()
                if len(parts) != 5:
                    logging.error(f"Error in {label_path}, line {line_num}: Expected 5 values, got {len(parts)}")
                    continue
                try:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    boxes.append([class_id, x_center, y_center, width, height])
                except ValueError:
                    logging.error(f"Error in {label_path}, line {line_num}: Non-numeric value found")
    except OSError as e:
        logging.error(f"File error with {label_path}: {e}")
    return boxes

def save_yolo_labels(label_path: str, boxes: List[List[float]]) -> None:
    if not boxes:
        logging.warning(f"No valid boxes to save for {label_path}")
        return
    try:
        with open(label_path, "w", encoding="utf-8") as f:
            for box in boxes:
                if len(box) != 5:
                    logging.warning(f"Skipping invalid box {box}")
                    continue
                f.write(" ".join(map(str, box)) + "\n")
    except OSError as e:
        logging.error(f"File error with {label_path}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error writing {label_path}: {e}")