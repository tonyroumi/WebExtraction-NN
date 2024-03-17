import os
import pickle
import random
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch

# PATHS
split_directory = '../data_news/page_sets/splits/'
boxes_directory = 'data_news/input_boxes/'
priors_directory = '../data_news/position_maps/'

# CONSTANTS
max_x = 1280
max_y = 1280
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

def calculate_iou(boxA, boxB):
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = torch.max(boxA[:, 0], boxB[:, 0])
    yA = torch.max(boxA[:, 1], boxB[:, 1])
    xB = torch.min(boxA[:, 2], boxB[:, 2])
    yB = torch.min(boxA[:, 3], boxB[:, 3])

    # Compute the area of intersection rectangle
    interArea = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[:, 2] - boxA[:, 0]) * (boxA[:, 3] - boxA[:, 1])
    boxBArea = (boxB[:, 2] - boxB[:, 0]) * (boxB[:, 3] - boxB[:, 1])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / (boxAArea + boxBArea - interArea)

    return iou



def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')

    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            # x = x.to(device=DEVICE, dtype=dtype)  # move to device, e.g. GPU
            y = y.view(-1)
            y = y.to(device=DEVICE, dtype=torch.long)
            scores = model(x)

            # Convert scores to bounding box coordinates
            # Assuming scores shape is [2, 3, 4] (batch_size, num_classes, 4)
            # Reshape scores to [2*3, 4] for easier computation
            

            # Assuming y shape is [2, 3, 4] (batch_size, num_classes, 4)
            # Reshape y to [2*3, 4] for easier computation
            

            # Calculate Intersection over Union (IoU) for each bounding box
            iou = calculate_iou(scores, y)

            # If IoU is greater than a certain threshold, consider it correct
            correct_predictions = (iou > 0.5).sum().item()

            num_correct += correct_predictions
            num_samples += x.size(0)

    accuracy = num_correct / num_samples
    print(f'Accuracy: {accuracy * 100:.2f}%')

# def check_accuracy(loader, model):
#     if loader.dataset.train:
#         print('Checking accuracy on validation set')
#     else:
#         print('Checking accuracy on test set')   
#     num_correct = 0
#     num_samples = 0
#     model.eval()  # set model to evaluation mode
#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device=DEVICE, dtype=dtype)  # move to device, e.g. GPU
#             y = y.to(device=DEVICE, dtype=torch.long)
#             scores = model(x)
#             _, preds = scores.max(1)
#             num_correct += (preds == y).sum()
#             num_samples += preds.size(0)
#         acc = float(num_correct) / num_samples
#         print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
#     return acc
    
    

#####--- GET DATA PATHS

def get_train_data_path(split_name):
    return os.path.join(split_directory,'split_'+split_name+'_train.txt')
 
def get_val_data_path(split_name):
    return os.path.join(split_directory,'split_'+split_name+'_val.txt')

def get_result_path(experiment, split_name):
    results_dir = os.path.join(test_results_directory, experiment)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)  

    return os.path.join(results_dir, split_name+'.txt')

def get_snapshot_name(experiment, split_name, iter):
    snapshots_dir = os.path.join(snapshots_directory, experiment)
    if not os.path.exists(snapshots_dir):
        os.makedirs(snapshots_dir)

    return os.path.join(snapshots_dir, 'snapshot_split_'+split_name+'_'+str(iter)+'.caffemodel')


####---  POSITION MAPS

def create_position_maps(train_data, split_name):
    print('Creating and saving position maps')

    n_samples = 10
    train_pages = train_data
    #--- GET TRAINING DATA
    # with open(train_data) as f:
    #     train_pages = [line.strip() for line in f.readlines()]

    #--- GET RANDOM SUBSET
    sample_pages = random.sample(train_pages, n_samples)

    #--- FROM EACH RANDOM PAGE, USE POSITION OF GT ELEMENTS 
    final_maps = [np.ones((max_y,max_x),dtype=np.float32)]*4
    for page in sample_pages:
        # load boxes
        boxes_path = os.path.join(boxes_directory, page+'.pkl')
        with open(boxes_path,'rb') as f:
            boxes = pickle.load(f)
        gt_boxes = boxes['gt_boxes']

        # add to finals
        background_map = np.ones((max_y,max_x),dtype=np.float32) # we add one (we do not want zero probability)
        for i in range(1,4):
            bb = (gt_boxes[i-1])
            bb = [int(x) for x in bb]


            background_map[bb[1]:bb[3],bb[0]:bb[2]] = 0
            type_map = np.zeros((max_y,max_x),dtype=np.float32)
            type_map[bb[1]:bb[3],bb[0]:bb[2]] = 1
            final_maps[i] = final_maps[i] + type_map

        final_maps[0] = final_maps[0] + background_map


    #--- normalize
    for i in range(0,4):
        final_maps[i] = final_maps[i]/(1+n_samples)

    #--- save
    for i in range(0,4):
        path = os.path.join(priors_directory,'split_'+str(split_name)+'_'+str(i)+'.pkl')
        #need to create file directory
        pickle.dump(final_maps[i], open(path,"wb"))


def load_position_map(file_name, sigma):
    map = pickle.load(open(file_name,'rb'))
    ### Gaussian convolution
    filtered_maps = []
    filtered = F.gaussian_filter(map, sigma=sigma)
    return filtered

def load_position_maps(split_name, sigma):
    print('Loading position maps smoothed with Gausian filter, sigma=' + str(sigma))

    maps = []
    for i in range(4):
        ### Load map
        #../data_news/position_maps/split_1_i.pkl
        path = os.path.join(priors_directory,'split_'+split_name+'_'+str(i)+'.pkl')
        filtered = load_position_map(path, sigma)
        maps.append(filtered)
    return maps


