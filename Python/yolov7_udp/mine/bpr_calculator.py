#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 08:33:38 2025

@author: holidayj
"""

import os
from glob import glob
import numpy as np
import torch
import random


# Config
label_path = "/home/holidayj/Desktop/yolov7_udp/data/train"
image_size = 320  # assumed image size, if known
thr        = 1.33  # threshold used in the original BPR calculation
scale_min, scale_max = 0.95, 1.05  # scale variance range

# Given anchor boxes (in pixels)
anchor_1  = [66,157]
anchor_9  = [49,90,   84,70,   40,164,  
             59,119,  60,160,  68,177, 
             80,186,  122,134, 101,170]


anchors   = np.array(anchor_1, dtype=np.float32).reshape(-1, 2)  # shape (9, 2)
print("anchors_actual     =", anchors)
anchors  /= image_size  # normalize if using normalized bbox sizes
print("anchors_normalized =", anchors)
# Step 1: Collect all normalized width and height values
wh_list = []

label_files = glob(os.path.join(label_path, "*.txt"))

for label_file in label_files:
    # print(label_file)
    with open(label_file, 'r') as f:
        lines = f.readlines()
        # print(lines)
        for line in lines:
            parts = line.strip().split()
            # print("parts =", parts)
            if len(parts) != 5:
                continue  # skip malformed lines
            _, _, _, w, h = map(float, parts)
            
            # # Apply scale variance here
            # scale = random.uniform(scale_min, scale_max)
            # w *= scale
            # h *= scale

            wh_list.append([w, h])
    # break

# print("wh_list =", wh_list) # wh_list of all objects
# Ensure data exists
if not wh_list:
    raise ValueError("No bounding box data found in the label files.")

# Convert to torch tensors
wh = torch.tensor(wh_list)  # shape: (num_objects, 2)
k  = torch.tensor(anchors)   # shape: (9, 2)
# print("k =", k)

# Step 2: Calculate the BPR metric
r = wh[:, None] / k[None]  # shape: (N_objects, 9, 2)
# print(r.shape)
# print('r =', r)
# print("torch.min(r, 1. / r).shape = ", torch.min(r, 1. / r).shape)
# print("torch.min(r, 1. / r).min(1) =", torch.min(r, 1. / r).min(1))

x = torch.min(r, 1. / r).min(2)[0]  # ratio metric: (N_objects, 9) -> min per box
# print('x =', x)
best = x.max(1)[0]  # best matching anchor per box
aat  = (x > 1. / thr).float().sum(1).mean()  # anchors above threshold
bpr  = (best > 1. / thr).float().mean()  # Best Possible Recall

# Step 3: Print the results
print(f"Best Possible Recall (BPR): {bpr:.4f}")
print(f"Average Anchors Above Threshold (AAT): {aat:.4f}")
