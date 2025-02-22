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
mosaic = st.sidebar.checkbox("Apply Mosaic", key="mosaic", disabled=disabled_flag)
flip_val = st.sidebar.checkbox("Apply Horizontal Flip", key="flip_val", disabled=disabled_flag)
grayscale = st.sidebar.checkbox("Apply Grayscale", key="grayscale", disabled=disabled_flag)

# Reset button in sidebar; clears state and reruns.
if st.sidebar.button("Reset to Default"):
    st.session_state.clear()
    st.rerun()

st.title("YOLOv5 Dataset Augmentation Tool")
st.sidebar.header("Augmentation Settings")

# Helper function to draw bounding boxes on an image.
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

# Helper: Apply base augmentations (all except mosaic)
def apply_base_augmentations(img, bboxes):
    if rotate_angle != 0:
        from augs.aug_rotate import rotate_image
        img, bboxes = rotate_image(img, bboxes, rotate_angle)
    if brightness != 1.0:
        from augs.aug_brightness import adjust_brightness
        img = adjust_brightness(img, brightness)
    if contrast != 1.0:
        from augs.aug_contrast import adjust_contrast
        img = adjust_contrast(img, contrast)
    if crop is not None:
        from augs.aug_crop import crop_image
        img, bboxes = crop_image(img, bboxes, crop)
    if exposure != 1.0:
        from augs.aug_exposure import adjust_exposure
        img = adjust_exposure(img, exposure)
    if flip_val:
        from augs.aug_flip import flip
        img, bboxes = flip(img, bboxes)
    if grayscale:
        from augs.aug_grayscale import convert_to_grayscale
        img = convert_to_grayscale(img)
    if hue != 0.0:
        from augs.aug_hue import adjust_hue
        img = adjust_hue(img, hue)
    if noise > 0:
        from augs.aug_noise import add_noise
        img = add_noise(img, noise)
    if saturation != 1.0:
        from augs.aug_saturation import adjust_saturation
        img = adjust_saturation(img, saturation)
    if shear != 0.0:
        from augs.aug_shear import shear_image
        img, bboxes = shear_image(img, bboxes, shear)
    return img, bboxes

# Main augmentation function incorporating mosaic logic.
def apply_augmentations(img, bboxes, selected_image, image_files):
    if mosaic:
        from augs.aug_mosaic import apply_mosaic
        base_img, base_bboxes = apply_base_augmentations(img, bboxes)
        index = image_files.index(selected_image)
        if index <= len(image_files) - 4:
            mosaic_candidates = image_files[index+1:index+4]
        else:
            mosaic_candidates = image_files[max(0, index-3):index]
        if len(mosaic_candidates) != 3:
            st.warning("Not enough images available for mosaic augmentation.")
            return base_img, base_bboxes
        mosaic_images = [base_img]
        mosaic_bboxes_list = [base_bboxes]
        for path in mosaic_candidates:
            im = cv2.imread(path)
            if im is None:
                continue
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            lbl = path.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt")
            bb = load_yolo_labels(lbl)
            base_im, base_bb = apply_base_augmentations(im, bb)
            mosaic_images.append(base_im)
            mosaic_bboxes_list.append(base_bb)
        img, bboxes = apply_mosaic(mosaic_images, mosaic_bboxes_list)
        return img, bboxes
    else:
        return apply_base_augmentations(img, bboxes)

# Select image for preview.
image_files = glob(os.path.join(IMG_SOURCE_DIR, "*.jpg")) + glob(os.path.join(IMG_SOURCE_DIR, "*.png"))
if image_files:
    selected_image = st.selectbox("Select an image to preview", image_files)
    image = cv2.imread(selected_image)
    if image is None:
        st.error("Failed to load image!")
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label_path = selected_image.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt")
        bboxes = load_yolo_labels(label_path)
        aug_image, aug_bboxes = apply_augmentations(image, bboxes, selected_image, image_files)
        aug_image_with_boxes = draw_boxes(aug_image, aug_bboxes)
        col1, col2 = st.columns(2)
        col1.image(image, caption="Original Image", use_column_width=True)
        col2.image(aug_image_with_boxes, caption="Augmented Preview with Boxes", use_column_width=True)
        
        process_button = st.button("Process Full Dataset", disabled=st.session_state.exporting)
        if process_button:
            st.session_state.exporting = True
            total = len(image_files)
            progress_bar = st.progress(0)
            status_text = st.empty()
            with st.spinner("Processing images..."):
                for i, img_path in enumerate(image_files):
                    progress = (i + 1) / total
                    progress_bar.progress(progress)
                    status_text.text(f"Processing image {i + 1} of {total}")
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    lbl_path = img_path.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt")
                    bboxes = load_yolo_labels(lbl_path)
                    proc_img, proc_bboxes = apply_augmentations(img, bboxes, img_path, image_files)
                    proc_img_with_boxes = draw_boxes(proc_img, proc_bboxes)
                    filename = os.path.basename(img_path)
                    cv2.imwrite(os.path.join(IMG_DEST_DIR, filename), cv2.cvtColor(proc_img_with_boxes, cv2.COLOR_RGB2BGR))
                    save_yolo_labels(os.path.join(LABEL_DEST_DIR, os.path.splitext(filename)[0] + ".txt"), proc_bboxes)
            st.success("Dataset augmentation complete!")
            st.session_state.exporting = False
            progress_bar.empty()
            status_text.empty()
            st.experimental_rerun()
else:
    st.warning("No images found in the input directory.")
