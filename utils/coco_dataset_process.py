import os
import shutil
import textwrap
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob
from fiftyone import ViewField as F

def define():
    pass


def coco_to_yolo(bbox):
    # print('bbox =', bbox)
    x = bbox[0] + bbox[2]/2
    y = bbox[1] + bbox[3]/2
    w = bbox[2]
    h = bbox[3]
    cxywh = str(0) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n'
    return cxywh
    

# Below is for saving images from COCO dataset selected only 'person' class. and saving labels in yolo format.
'''
#%% Setting config

fo.config.dataset_zoo_dir = r"D:\my_doc\data\fiftyone" # for windows

#%% Loading dataset

dataset = foz.load_zoo_dataset("coco-2017", split="train")

view = dataset.view()


#%% Extracting from coco dataset.
# print(view)
print('number of files =', len(view))
# print(view.media_type)

out_path = r'D:/my_doc/data/coco_persons/train/'

# loop for each file.
for i, sample in enumerate(view):
    if sample.ground_truth == None:
        continue
    # if i == 1:
    #     break
    
    img_w     = sample.metadata['width']
    img_h     = sample.metadata['height']
    
    img_path    = sample.filepath
    print(img_path)
    
    img_file      = os.path.basename(img_path)
    img_name, ext = os.path.splitext(img_file)
    
    img_out = out_path+img_file
    txt_out = out_path+img_name+'.txt'
    
    
    gt       = sample.ground_truth['detections']
    
    num_gt   = len(gt)
    
    person = False
    
    # loop for each detection.
    cxywh = ''
    for k in range(num_gt):
        # print('k =', k)
        if gt[k]['label'] == 'person':
            person = True
            # print("gt[{}]['bounding_box'] =".format(k), gt[k]['bounding_box'])
            cxywh += coco_to_yolo(gt[k]['bounding_box'])
            # print(cxywh)
            
            # print(cxywh)
    if person == True:
        with open(txt_out, 'w') as txt:
            # print('writing!!!')
            txt.write(cxywh)
        shutil.copy(img_path, img_out)
'''


#%% After filtering, we need to move original images and labels to the new folder so that we can check the annotations.
