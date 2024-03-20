import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import cv2
import pickle
import numpy as np
from custom_layers.web_data_utils import *



IMAGES_PATH='/Users/anthonyroumi/Desktop/WebExtraction-NN/data_news/images'
TEXT_MAPS_PATH='/Users/anthonyroumi/Desktop/WebExtraction-NN/data_news/text_maps'
BOXES_PATH='/Users/anthonyroumi/Desktop/WebExtraction-NN/data_news/input_boxes'

x_size = 1280
y_size = 1280
text_map_scale = 0.125


def load_data_set(path):
    print('Loading data from:'+ str(path))
    with(open(path,'r')) as f:
        data = [line.strip() for line in f.readlines()]
        return data


def tensor_list_to_blob(ims):
        max_shape = np.array([im.shape for im in ims]).max(axis=0)
        n_channels = ims.shape[2]

        blob = np.zeros((max_shape[0], max_shape[0], n_channels),
                    dtype=np.float32)
       
        blob[0:ims.shape[0], 0:ims.shape[1], :] = ims
 
        blob = blob.transpose((2, 0, 1)) #(3,1280,1280)
        return blob



def load_boxes(train, boxes_filename, size):
        n_others = size - 3

        # load data
        with open(boxes_filename,'rb') as f:
            boxes = pickle.load(f)

        gt_boxes = boxes['gt_boxes']
        other_boxes = boxes['other_boxes']

        # remove boxes outside the considered area
        keep_indices = np.logical_and.reduce(((other_boxes[:,0]>=0), (other_boxes[:,1]>=0),(other_boxes[:,2]<=x_size), (other_boxes[:,3]<=y_size)))
        other_boxes = other_boxes[keep_indices,:]
        
        # create boxes
        if train == True:
            rows_to_include=np.random.randint(other_boxes.shape[0],size=n_others)
            other_boxes = other_boxes[rows_to_include,:]

        boxes = np.vstack((gt_boxes,other_boxes))
        
        # labels
        gt_labels = np.asarray(range(1,4))
        other_labels = np.zeros((other_boxes.shape[0]), dtype = np.float32)
        labels = np.hstack((gt_labels,other_labels))

        return boxes, labels

def load_text_map(filename):
        with open(filename,'rb') as f:
            obj = pickle.load(f)

        text_shape = obj['shape']
        text_nodes = obj['text_nodes']        

        spatial_shape = [y_size,x_size]
        n_features = text_shape[2]

        
        text_map = get_text_maps(text_nodes, n_features, spatial_shape, text_map_scale)
        return text_map   


def load_image(filename, train):
    im = cv2.imread(filename)

    size_x = min(im.shape[1],x_size)
    size_y = min(im.shape[0],y_size)

    # Crop I wonder if I just keep the size. This aspect ration is 1280x800
    im_croped = np.zeros((y_size,x_size,3),dtype=np.uint8)
    im_croped[:size_y,:size_x,:] = im[:size_y,:size_x,:] 

    if train == True:
        #Pre-processing data
        # Change HUE randomly
        hue_ratio = 0.4
        if np.random.uniform() < hue_ratio:
            hsv = cv2.cvtColor(im_croped, cv2.COLOR_BGR2HSV) 
            add_value = np.random.randint(low=0,high=180,size=1)[0]
            add_matrix = np.ones(hsv.shape[0:2],dtype=np.uint8)*add_value
            hsv2 = hsv
            hsv2[:,:,0] += add_matrix
            im_croped = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

            # With probability 15 percent invert image
        invert_ratio = 0.15
        if np.random.uniform() < invert_ratio:
            im_croped = (255-im_croped)


    im_scaled = cv2.resize(im_croped, (0,0), fx=1, fy=1)

    return im_scaled

if __name__ == "__main__":
    pass

