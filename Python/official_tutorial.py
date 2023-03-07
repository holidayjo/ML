'''
This tutorial is from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
'''

import os
import numpy as np
import torch
from PIL import Image
# import transforms as T

im = Image.open(r'D:\data\data\PennFudanPed\PNGImages\FudanPed00001.png')
im.show()
#%%
# class PennFudanDataset(torch.utils.data.Dataset):
#     def __init__(self, root, transforms):
#         self.root       = root
#         self.transforms = transforms
#         # Loading all image files, sorting them to ensure that they are aligned.
#         self.imgs  = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
#         self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
    
#     def __getitem__(self, idx):
#         # Loading imgaes and masks
#         img_path  = os.path.join(self.root, "PNGImages", self.imgs[idx])
#         mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        
# dataset = PennFudanDataset(r'D:\data\data\PennFudanPed', get_transform(train=False))
