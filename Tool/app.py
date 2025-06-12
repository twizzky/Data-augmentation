import streamlit as st
import os
import cv2
import numpy as np
from glob import glob
import random
from YOLOv5_UTILS import load_yolo_labels, save_yolo_labels

# Config paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
IMG_SOURCE_DIR = os.path.join(BASE_DIR, "Upload/images")
LABEL_SOURCE_DIR = os.path.join(BASE_DIR, "Upload/labels")
IMG_DEST_DIR = os.path.join(BASE_DIR, "Destination/images")
LABEL_DEST_DIR = os.path.join(BASE_DIR, "Destination/labels")

os.makedirs(IMG_DEST_DIR, exist_ok=True)
os.makedirs(LABEL_DEST_DIR, exist_ok=True)
st.sidebar.header("Augmentation Parameters")
# Initialize session state variables if not set.
if 'exporting' not in st.session_state:
    st.session_state.exporting = False
if 'rotate_angle' not in st.session_state:
    st.session_state.rotate_angle = 0
if 'brightness' not in st.session_state:
    st.session_state.brightness = 1.0
if 'contrast' not in st.session_state:
    st.session_state.contrast = 1.0
if 'crop_val' not in st.session_state:
    st.session_state.crop_val = 1.0
if 'exposure' not in st.session_state:
    st.session_state.exposure = 1.0
if 'flip_val' not in st.session_state:
    st.session_state.flip_val = False
if 'grayscale' not in st.session_state:
    st.session_state.grayscale = False
if 'hue' not in st.session_state:
    st.session_state.hue = 0
if 'mosaic' not in st.session_state:
    st.session_state.mosaic = False
if 'noise' not in st.session_state:
    st.session_state.noise = 0
if 'saturation' not in st.session_state:
    st.session_state.saturation = 1.0
if 'shear' not in st.session_state:
    st.session_state.shear = 0.0

# Define disabled flag based on exporting state.
disabled_flag = st.session_state.exporting

# Sidebar controls (all disabled if exporting)

rotate_angle = st.sidebar.slider("Rotation Angle", -45, 45, st.session_state.rotate_angle, key="rotate_angle", disabled=disabled_flag)
brightness = st.sidebar.slider("Brightness Factor", 0.5, 2.0, st.session_state.brightness, key="brightness", disabled=disabled_flag)
contrast = st.sidebar.slider("Contrast Factor", 0.5, 3.0, st.session_state.contrast, key="contrast", disabled=disabled_flag)
crop_val = st.sidebar.slider("Crop Factor", 0.5, 1.0, st.session_state.crop_val, key="crop_val", disabled=disabled_flag)
crop = None if crop_val == 1.0 else crop_val
exposure = st.sidebar.slider("Exposure Factor", 0.5, 2.5, st.session_state.exposure, key="exposure", disabled=disabled_flag)
hue = st.sidebar.slider("Hue Delta", -50, 50, st.session_state.hue, key="hue", disabled=disabled_flag)
noise = st.sidebar.slider("Noise Variance", 0, 50, st.session_state.noise, key="noise", disabled=disabled_flag)
saturation = st.sidebar.slider("Saturation Factor", 0.5, 2.0, st.session_state.saturation, key="saturation", disabled=disabled_flag)
shear = st.sidebar.slider("Shear Factor", -0.3, 0.3, st.session_state.shear, key="shear", disabled=disabled_flag)

# Augmentation selection (radio button)
selected_aug = st.sidebar.radio(
    "Select Augmentation", 
    options=["None", "Rotation", "Brightness", "Contrast", "Crop", "Exposure", "Flip", "Grayscale", "Hue", "Mosaic", "Noise", "Saturation", "Shear"],
    index=0, disabled=disabled_flag
)

# For simple on/off augmentations:
mosaic_checkbox = st.sidebar.checkbox("Apply Mosaic", key="mosaic", disabled=disabled_flag)
flip_checkbox = st.sidebar.checkbox("Apply Horizontal Flip", key="flip_val", disabled=disabled_flag)
grayscale_checkbox = st.sidebar.checkbox("Apply Grayscale", key="grayscale", disabled=disabled_flag)

# Reset button in sidebar; clears state and reruns.
if st.sidebar.button("Reset to Default"):
    st.session_state.clear()
    st.experimental_rerun()

st.title("YOLOv5 Dataset Augmentation Tool")


# Helper function to draw bounding boxes.
def draw_boxes(image, bboxes, color=(255, 0, 0), thickness=2):
    img = image.copy()
    h, w, _ = img.shape
    for bbox in bboxes:
        class_id, x_center, y_center, bw, bh = bbox
        x_center_abs = int(x_center * w)
        y_center_abs = int(y_center * h)
        box_w = int(bw * w)
        box_h = int(bh * h)
        x1 = x_center_abs - box_w // 2
        y1 = y_center_abs - box_h // 2
        x2 = x1 + box_w
        y2 = y1 + box_h
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(img, str(int(class_id)), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    return img

#individual augmentation functions
def process_rotation(img, bboxes):
    from augs.aug_rotate import rotate_image
    return rotate_image(img, bboxes, rotate_angle)

def process_brightness(img, bboxes):
    from augs.aug_brightness import adjust_brightness
    return adjust_brightness(img, brightness), bboxes

def process_contrast(img, bboxes):
    from augs.aug_contrast import adjust_contrast
    return adjust_contrast(img, contrast), bboxes

def process_crop(img, bboxes):
    if crop is None:
        return img, bboxes
    from augs.aug_crop import crop_image
    return crop_image(img, bboxes, crop)

def process_exposure(img, bboxes):
    from augs.aug_exposure import adjust_exposure
    return adjust_exposure(img, exposure), bboxes

def process_flip(img, bboxes):
    from augs.aug_flip import flip
    return flip(img, bboxes)

def process_grayscale(img, bboxes):
    from augs.aug_grayscale import convert_to_grayscale
    return convert_to_grayscale(img), bboxes

def process_hue(img, bboxes):
    from augs.aug_hue import adjust_hue
    return adjust_hue(img, hue), bboxes

def process_noise(img, bboxes):
    from augs.aug_noise import add_noise
    return add_noise(img, noise), bboxes

def process_saturation(img, bboxes):
    from augs.aug_saturation import adjust_saturation
    return adjust_saturation(img, saturation), bboxes

def process_shear(img, bboxes):
    from augs.aug_shear import shear_image
    return shear_image(img, bboxes, shear)

def process_mosaic(img, bboxes, selected_img, image_files):
    from augs.aug_mosaic import apply_mosaic
    index = image_files.index(selected_img)
    if index <= len(image_files) - 4:
        candidates = image_files[index+1:index+4]
    else:
        candidates = image_files[max(0, index-3):index]
    if len(candidates) != 3:
        return img, bboxes
    mosaic_images = [img]
    mosaic_bboxes_list = [bboxes]
    for path in candidates:
        im = cv2.imread(path)
        if im is None:
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        lbl = path.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt")
        bb = load_yolo_labels(lbl)
        mosaic_images.append(im)
        mosaic_bboxes_list.append(bb)
    return apply_mosaic(mosaic_images, mosaic_bboxes_list)

# Map augmentation names to functions
augmentation_funcs = {
    "Rotation": process_rotation,
    "Brightness": process_brightness,
    "Contrast": process_contrast,
    "Crop": process_crop,
    "Exposure": process_exposure,
    "Flip": process_flip,
    "Grayscale": process_grayscale,
    "Hue": process_hue,
    "Noise": process_noise,
    "Saturation": process_saturation,
    "Shear": process_shear,
    "Mosaic": lambda img, bboxes: process_mosaic(img, bboxes, preview_img_path, image_files)
}

# Preview Section
image_files = glob(os.path.join(IMG_SOURCE_DIR, "*.jpg")) + glob(os.path.join(IMG_SOURCE_DIR, "*.png"))
if image_files:
    # Use preview_img_path as the selected image for preview
    preview_img_path = st.selectbox("Select an image to preview", image_files)
    orig_img = cv2.imread(preview_img_path)
    if orig_img is None:
        st.error("Failed to load image!")
    else:
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        lbl_path = preview_img_path.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt")
        bboxes = load_yolo_labels(lbl_path)
        
        if selected_aug == "None":
            preview_img, preview_bboxes = orig_img, bboxes
        else:
            process_func = augmentation_funcs.get(selected_aug)
            if process_func is not None:
                # For mosaic, our function expects the selected image and list of image files.
                if selected_aug == "Mosaic":
                    preview_img, preview_bboxes = process_func(orig_img, bboxes)
                else:
                    preview_img, preview_bboxes = process_func(orig_img, bboxes)
            else:
                preview_img, preview_bboxes = orig_img, bboxes
        
        preview_img_with_boxes = draw_boxes(preview_img, preview_bboxes)
        col1, col2 = st.columns(2)
        col1.image(orig_img, caption="Original Image", use_column_width=True)
        col2.image(preview_img_with_boxes, caption=f"Preview: {selected_aug}", use_column_width=True)
        
        #Dataset Processing Section
        process_button = st.button("Process Full Dataset", disabled=st.session_state.exporting)
        if process_button:
            st.session_state.exporting = True
            aug_name = selected_aug if selected_aug != "None" else "original"
            dest_img_folder = os.path.join(IMG_DEST_DIR, aug_name)
            dest_lbl_folder = os.path.join(LABEL_DEST_DIR, aug_name)
            os.makedirs(dest_img_folder, exist_ok=True)
            os.makedirs(dest_lbl_folder, exist_ok=True)
            
            total = len(image_files)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, img_path in enumerate(image_files):
                progress_bar.progress((i + 1) / total)
                status_text.text(f"Processing image {i + 1} of {total} for {aug_name}")
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                lbl_path = img_path.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt")
                bboxes = load_yolo_labels(lbl_path)
                if selected_aug == "None":
                    proc_img, proc_bboxes = img, bboxes
                else:
                    proc_func = augmentation_funcs.get(selected_aug)
                    if proc_func is not None:
                        
                        if selected_aug == "Mosaic":
                            proc_img, proc_bboxes = proc_func(img, bboxes)
                        else:
                            proc_img, proc_bboxes = proc_func(img, bboxes)
                    else:
                        proc_img, proc_bboxes = img, bboxes
                proc_img_with_boxes = draw_boxes(proc_img, proc_bboxes)
                filename = os.path.basename(img_path)
                cv2.imwrite(os.path.join(dest_img_folder, filename), cv2.cvtColor(proc_img_with_boxes, cv2.COLOR_RGB2BGR))
                save_yolo_labels(os.path.join(dest_lbl_folder, os.path.splitext(filename)[0] + ".txt"), proc_bboxes)
            
            progress_bar.empty()
            status_text.empty()
            st.success(f"Dataset augmentation for {aug_name} complete!")
            st.session_state.exporting = False
            st.rerun()
else:
    st.warning("No images found in the input directory.")
