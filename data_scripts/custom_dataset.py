import os
from torch.utils.data import Dataset
import numpy as np
import torch
from data_scripts.data_utils import *

X_SIZE = 1280
Y_SIZE = 1280

# x[0]: im_blob
# Contains the tensor form of image (2,3,1280,1280)
# X[1]: text_blob
# Contains the tensor form of the text map created (2,128,160,160)
# X[2]: boxes_blob
# Contains tensor form of all candidate boxes and their positions and batch index (200,5)
# (100 candidates for classification in each image)
# (1st element is index of batch images (0,1), other 4 are positions)
# X[3]: labels_blob
# Contains tensor form of labels 
# return image, labels



class CustomDataset(Dataset):
    def __init__(self, img_path, txt_map_path, boxes_path, train=False, img_transform=None, txt_transform=None):
        self.train = train
        self.img_transform = img_transform
        self.txt_transform = txt_transform

        self.img_path = img_path
        self.txt_map_path = txt_map_path
        self.boxes_path = boxes_path
        
        self.images = os.listdir(img_path)
        self.text_maps = os.listdir(txt_map_path)
        self.boxes = os.listdir(boxes_path)
        self.synchronize_lists()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        top = []
        labels_blob = np.zeros((0), dtype=np.float32)

        img_path = os.path.join(self.img_path, self.images[idx])
        txt_map_path = os.path.join(self.txt_map_path, self.text_maps[idx])
        boxes_path = os.path.join(self.boxes_path, self.boxes[idx])
        image = load_image(filename=img_path, train=self.train) 
        text_map = load_text_map(filename=txt_map_path)
        boxes, labels = load_boxes(self.train, boxes_filename=boxes_path, size=100)
        labels_blob = np.hstack((labels_blob, labels))
        
        im_blob = tensor_list_to_blob(image)
        top.append(torch.tensor(im_blob))
        txt_blob = tensor_list_to_blob(text_map)
        top.append(torch.tensor(txt_blob))
        top.append(torch.tensor(boxes))
        return top, torch.tensor(labels_blob)


    def synchronize_lists(self):
        stripped_img  = [element.split('.')[0] for element in self.images]
        stripped_txt_maps = [element.split('.')[0] for element in self.text_maps]
        stripped_boxes = [element.split('.')[0] for element in self.boxes]
        common_elements = set(stripped_img) & set(stripped_txt_maps) & set(stripped_boxes)    
        self.images = [img + '.jpeg' for img in common_elements]
        self.text_maps = [img +'.pkl' for img in common_elements]
        self.boxes = [img +'.pkl' for img in common_elements]
       