#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 11:54:06 2025

@author: holidayj
"""

import json
import os
from tqdm import tqdm

# Paths
coco_annotation_file = "/media/holidayj/Documents/data/datasets_for_object_detection/coco/annotations/instances_train2017.json"
image_dir = "/media/holidayj/Documents/data/datasets_for_object_detection/coco/images/train2017"
yolo_output_dir = "/media/holidayj/Documents/data/datasets_for_object_detection/coco/images/train2017_yolo"

# Create output directory if not exists
os.makedirs(yolo_output_dir, exist_ok=True)

# Load COCO annotations
with open(coco_annotation_file, 'r') as f:
    coco = json.load(f)

# Create mapping from image IDs to file names and sizes
image_id_to_info = {image['id']: image for image in coco['images']}

# Create mapping from category IDs to class indices (YOLO classes start from 0)
categories = coco['categories']
category_id_to_class_index = {cat['id']: idx for idx, cat in enumerate(categories)}

# Collect annotations per image
annotations_per_image = {}
for ann in coco['annotations']:
    image_id = ann['image_id']
    if image_id not in annotations_per_image:
        annotations_per_image[image_id] = []
    annotations_per_image[image_id].append(ann)

# Convert COCO annotations to YOLO format
def convert_bbox(size, bbox):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x, y, w, h = bbox
    x_center = x + w / 2.0
    y_center = y + h / 2.0
    return [x_center * dw, y_center * dh, w * dw, h * dh]

# Process each image
for image_id, image_info in tqdm(image_id_to_info.items(), desc="Converting to YOLO format"):
    file_name = image_info['file_name']
    width = image_info['width']
    height = image_info['height']

    yolo_lines = []
    annotations = annotations_per_image.get(image_id, [])

    for ann in annotations:
        category_id = ann['category_id']
        class_index = category_id_to_class_index[category_id]
        bbox = convert_bbox((width, height), ann['bbox'])
        yolo_line = f"{class_index} {' '.join(f'{x:.6f}' for x in bbox)}"
        yolo_lines.append(yolo_line)

    # Write to .txt file if there are annotations
    if yolo_lines:
        txt_file = os.path.join(yolo_output_dir, f"{os.path.splitext(file_name)[0]}.txt")
        with open(txt_file, 'w') as f:
            f.write("\n".join(yolo_lines))

print("Conversion completed successfully.")
