import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from data_scripts.custom_dataset import CustomDataset
from tools.trainer import Trainer
import torchvision.models as models

from models.model import SegNet
from tools.utils import *


dtype=torch.float32
#HyperParams:
LEARNING_RATE = 5e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    train_ds = CustomDataset(img_path='../data_news/images', txt_map_path='../data_news/text_maps', boxes_path='../data_news/input_boxes', train=True)
    train_loader = DataLoader(train_ds, batch_size=2,  num_workers=2, sampler=sampler.SubsetRandomSampler(range(60)))

    val_ds = CustomDataset(img_path='../data_news/images', txt_map_path='../data_news/text_maps', boxes_path='../data_news/input_boxes', train=True)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2,sampler=sampler.SubsetRandomSampler(range(60,66)))
    
    test_ds = CustomDataset(img_path='../data_news/images', txt_map_path='../data_news/text_maps', boxes_path='../data_news/input_boxes')
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    model = SegNet()
    model.to(device=DEVICE)
    
    #Gaussian variance
    # position_maps = load_position_maps(80)
    

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=True, weight_decay=WEIGHT_DECAY)
    best_model = Trainer(model,train_loader,val_loader,test_loader, optimizer=optimizer, num_epochs=50)
    
    #Save model
    checkpoint = {'state_dict' : best_model.state_dict(), 'optimizer': optimizer.state_dict()}
    save_checkpoint(checkpoint)

    check_accuracy(test_loader, best_model)

#load_checkpoint(torch.load('../models/checkpoint'/..tar))

#(FILE) input boxes: contain pickled files of dictionary: box_obj['gt_boxes'] contain ground truth positions for classification
                                                                          # 0: title, 1: date, 2: content
                                                      #box_obj['other_boxes'] contain other leaf nodes positions
    
#(FILE) text_maps: contain dictionary:  txt_obj['shape'] = shape (shape of image (1600,2560,128))
                                           # txt_obj['text_nodes'] = all leaf nodes that are text nodes with value:
                                            #they are tuples (0): position , (1): information MM3 hash vectorized with 128 features. (2) area of bounding box
                                      
#(FILE) page_sets: contains txt files of successfully downloaded pages with unique id
#(FILE) images: contain all images (unique id)