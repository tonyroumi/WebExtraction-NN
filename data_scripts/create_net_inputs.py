import os
import cv2
import pickle
import argparse
import numpy as np
import sys
import pyperclip
import json
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
import custom_layers.web_data_utils as data_utils
from custom_layers.dom_tree import DOMTree
from tools.utils import *

IMAGES_PATH = 'data_news/images/'
LABELED_DOM_PATH = 'data_news/labeled_dom_trees/'
PAGE_SETS_PATH = 'data_news/page_sets/'
BOXES_PATH = 'data_news/input_boxes/'
TEXT_MAPS_PATH = 'data_news/text_maps/'
POS_PATH = 'data_news/position_maps'

### CONSTANTS
N_FEATURES = 128
Y_SIZE = 1280
X_SIZE = 1280
SPATIAL_SHAPE = (Y_SIZE, X_SIZE)
TEXT_MAP_SCALE = 0.125


### labeled boxes
label_to_ind = {
    'author' : 0,
    'date' : 1,
    'content' : 2        
}

#----- MAIN PART
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str, help='PREFIX')
    args = parser.parse_args()

    # create boxes directory if it does not exist
    if not os.path.exists(BOXES_PATH):
        os.makedirs(BOXES_PATH)

    # create text maps directory if it does not exist
    if not os.path.exists(TEXT_MAPS_PATH):
        os.makedirs(TEXT_MAPS_PATH)

    # load pages from final page set
    page_set_path = os.path.join(PAGE_SETS_PATH, args.prefix+'.txt')
    with open(page_set_path,'r') as f:
        pages = [line.strip() for line in f.readlines()]

    # for each page
    for page in pages:
        print(str(page))

        # init boxes and text nodes
        gt_boxes = np.zeros((len(label_to_ind),4),dtype = np.float32)
        other_boxes = []

        # load image
        image_path = os.path.join(IMAGES_PATH,page+'.jpeg')
        im = cv2.imread(image_path)
        shape = (im.shape[0],im.shape[1],N_FEATURES)

        # load labeled dom tree
        dom_path = os.path.join(LABELED_DOM_PATH,page+'.json')
        dom = DOMTree(dom_path)
        
        # get leaf nodes
        leafNodes = dom.getPositionedLeafNodes()

        # for each leaf node
        for node in leafNodes:
            #-- process input boxes
            if 'label' in node:
                label = node['label']
                ind = label_to_ind[label]
                position = node['position']
                position = [x * 2 for x in position]
                gt_boxes[ind,:] = position
            else:
                position = node['position']
                position = [x * 2 for x in position]
                node['position'] = position
                other_boxes.append(node['position'])


        text_nodes = data_utils.get_text_nodes(leafNodes,N_FEATURES)

       

        other_boxes =  np.array(other_boxes,dtype = np.float32)

        #-- SAVE BOXES
        box_obj = {}
        box_obj['other_boxes'] = other_boxes #other leaf nodes positions 
        box_obj['gt_boxes'] = gt_boxes #ground truth positions for classification 0: author, 1: date, 2: content
        box_path = os.path.join(BOXES_PATH,page+'.pkl')
        with open(box_path,'wb+') as f:    
            pickle.dump(box_obj, f)

        

        #-- SAVE TEXT NODES
        txt_obj = {}
        txt_obj['shape'] = shape
        txt_obj['text_nodes'] = text_nodes #text nodes contains leaf nodes (text nodes) that have a value:
                                           #they are tuples (0): position , (1): information vectorized with 128 features. (2) area of bounding box
        txt_path = os.path.join(TEXT_MAPS_PATH,page+'.pkl')
        with open(txt_path,'wb+') as f:    
            pickle.dump(txt_obj, f)

       #--SAVE POSITION MAP
        # pos_map = create_position_maps(page)
        # pos_path = os.path.join(POS_PATH,page+'.pkl')
        # with open(pos_path,'wb+') as f:    
        #     pickle.dump(pos_map, f)





        

        
