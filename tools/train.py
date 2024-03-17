import os
import sys
import pickle
import utils
import test
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from data_scripts.custom_dataset import CustomDataset
import torchvision.models as models

from models.model import SegNet
from tools.utils import *

dtype=torch.float32
#HyperParams:
LEARNING_RATE = 5e-4
BATCH_SIZE = 2
NUM_EPOCHS = 5
NUM_WORKERS = 2
IMAGE_HEIGHT = 1280
IMAGE_WIDTH = 1280
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_TRAIN = 40
WEIGHTS = torch.tensor([0.1, 10, 10, 10],dtype=dtype)

print_every = 200

def train_fn(train_loader, val_loader, model, optimizer, epochs=1):
    model = model.to(device=DEVICE)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(train_loader):
            model.train()  # put model to training mode
            x[0] = x[0].to(device=DEVICE, dtype=dtype)
            x[1] = x[1].to(device=DEVICE, dtype=dtype)
            x[2] = x[2].to(device=DEVICE, dtype=dtype)
            y = y.view(-1)
            y = y.to(device=DEVICE, dtype=torch.long)
            scores = model(x)
            
            loss = F.cross_entropy(scores, y, weight=WEIGHTS, ignore_index=4)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            torch.autograd.set_detect_anomaly(True)
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            print('Epoch %d, Iteration %d, loss = %.4f' % (e, t + 1, loss.item()))

            # if (t + 1) % 1 == 0:
            #     print('Epoch %d, Iteration %d, loss = %.4f' % (e, t + 1, loss.item()))
            #     # check_accuracy(val_loader, model)
            #     # print()
    return check_accuracy(val_loader, model)



if __name__ == "__main__":
    #(FILE) input boxes: contain pickled files of dictionary: box_obj['gt_boxes'] contain ground truth positions for classification
                                                                          # 0: author, 1: date, 2: content
                                                      #box_obj['other_boxes'] contain other leaf nodes positions
    
    #(FILE) text_maps: contain dictionary:  txt_obj['shape'] = shape (shape of image (1600,2560,128))
                                           # txt_obj['text_nodes'] = all leaf nodes that are text nodes with value:
                                            #they are tuples (0): position , (1): information MM3 hash vectorized with 128 features. (2) area of bounding box
                                      
    #(FILE) page_sets: contains txt files of successfully downloaded pages with unique id
    #(FILE) images: contain all images (unique id)

    ##param_str = {{'phase':'TRAIN','batch_size': 2, 'data': '','im_scale':1,'txt_scale':0.125}}

    # img_transform  = T.Compose([
    #             # T.ColorJitter(hue=(0,0.5)),
    #             # T.RandomInvert(p=0.15),
    #             T.Normalize(mean=0.5, std=0.5),
    #         ])
    # txt_transform = T.Compose(T.Normalize(mean=0.5, std=0.5))

    train_ds = CustomDataset(img_path='data_news/images', txt_map_path='data_news/text_maps', boxes_path='data_news/input_boxes', train=True)
    train_loader = DataLoader(train_ds, batch_size=2, num_workers=2, sampler=sampler.SubsetRandomSampler(range(24)))

    val_ds = CustomDataset(img_path='data_news/images', txt_map_path='data_news/text_maps', boxes_path='data_news/input_boxes', train=True)
    val_loader = DataLoader(val_ds, batch_size=2,num_workers=2,sampler=sampler.SubsetRandomSampler(range(24,28)))
    
    test_ds = CustomDataset(img_path='data_news/images', txt_map_path='data_news/text_maps', boxes_path='data_news/input_boxes')
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=2)

 

    # alexnet = models.alexnet(pretrained=True)

    # conv1_weights = alexnet.features[0].weight
    # conv2_weights = alexnet.features[10].weight

    model = SegNet()

    # model.conv1.weight.data.copy_(conv1_weights)
    # model.conv2.weight.data.copy_(conv2_weights)
   
    model.to(device=DEVICE)

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, nesterov=True, weight_decay=0.0005)
    train_fn(train_loader, val_loader, model, optimizer, epochs=10)
