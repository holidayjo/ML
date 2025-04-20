#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 21:30:59 2024

@author: holidayj
"""

import torch
r = torch.tensor([
    [[2.0, 0.5], [1.5, 0.8]],
    [[1.0, 0.6], [2.5, 0.4]],
    [[0.9, 1.2], [1.1, 1.8]],
])  # Shape: [3, 2, 2]
print(r)

reciprocal_r = 1. / r

print(reciprocal_r)


max_r = torch.max(r, reciprocal_r)
print(max_r)


