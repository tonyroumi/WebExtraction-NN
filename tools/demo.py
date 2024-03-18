import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import utils
import argparse
import tempfile
import subprocess
import matplotlib.pyplot as plt
from utils import load_position_map
# from test import  get_probabilities_with_position
from custom_layers.dom_tree import DOMTree
from models.model import SegNet
import torch
import torch.optim as optim
from tools.utils import *


if __name__ == "__main__":
    #--- Get params
    top = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, help='URL to classify', required=True)
    args = parser.parse_args()
    
    #-- Load params
    url = args.url

    # DOWNLOAD PAGE
    # try:
    #     download_dir = download_page(url)
    # except subprocess.CalledProcessError:
    #     print("Download was not succesfull")
    #     sys.exit(1)

    # screenshot_path = os.path.join(download_dir,"screenshot.jpeg")
    screenshot_path = 'data_news/images/aisnakeoil-000002.jpeg'

    
    # dom_path = os.path.join(download_dir,"dom.json")
    dom_path = 'data_news/dom_trees/aisnakeoil-000002.json'

    # LOAD POSITION LIKELIHOODS
    # position_maps = load_position_maps(position_map_path)

    # LOAD IMAGE BLOB
    im_blob = load_image_blob(screenshot_path)
    
    top.append(torch.tensor(im_blob))

    ## Make sure net inputs are correct

    # LOAD TEXT BLOB AND BOXES BLOB
    dom = DOMTree(dom_path)
    leaf_nodes = dom.getPositionedLeafNodes()
    text_blob = load_text_blob(leaf_nodes)
    top.append(torch.tensor(text_blob))
    boxes_blob = load_boxes_blob(leaf_nodes,im_blob.shape[3],im_blob.shape[2])
    top.append(torch.tensor(boxes_blob))
    model = SegNet()
    optimizer = optim.SGD(model.parameters())
    utils.load_checkpoint(torch.load('models/checkpoint/model_checkpoint.tar'), model=model, optimizer=optimizer)
    model.eval()
    #Net forward
    with torch.no_grad():
        scores = model(top)
        prob = F.softmax(scores)

    show(screenshot_path, boxes_blob, prob)
