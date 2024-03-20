import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import pickle
import random
import numpy as np
import torch.nn.functional as F
import cv2
import tempfile
import subprocess
import numpy as np
import matplotlib.pyplot as plt
# from utils import load_position_map
# from test import  get_probabilities_with_position
import custom_layers.web_data_utils as data_utils
import torch
from torchvision import transforms as T


# PATHS
split_directory = '../data_news/page_sets/splits/'
boxes_directory = '../data_news/input_boxes/'
priors_directory = '../data_news/position_maps/'
CHECKPOINT_DIR = '../models/checkpoint'

# CONSTANTS
max_x = X_SIZE = 1280
max_y = Y_SIZE = 1280
N_FEATURES = 128
SPATIAL_SHAPE = (Y_SIZE, X_SIZE)
TEXT_MAP_SCALE = 0.125
GAUSS_VAR = 80
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32


def get_acc(scores):
    title_acc = 0
    date_acc = 0
    content_acc = 0
    with torch.no_grad():
        predicted = F.softmax(scores)
               
    # find boxes with highest probability
    max_boxes = np.argmax(predicted,axis=0)     
                
    # GT RESULTS     
    for idx, cls in enumerate(range(1, 4), start=0):
        if max_boxes[cls] == idx:
            if idx == 0:
                title_acc += 1
            elif idx == 1:
                date_acc += 1
            elif idx == 2:
                content_acc += 1
    
    return title_acc, date_acc, content_acc

def get_preds(prob):
    num_correct = 0
    predicted = prob            
    # find boxes with highest probability
    max_boxes = np.argmax(predicted,axis=0)     
                
    # GT RESULTS     
    for cls in range(1,4):
        ind = max_boxes[cls]
        winner_prob = predicted[ind,cls]

        if max_boxes[cls]==cls-1:
            num_correct += 1
            result = 'Right'
        else: 
            result = 'Wrong'
        
        print('CLASS ID:'+ str(cls), '('+str(result)+')')
        print('PROBABILITIES')
        print('GT: ' + str(predicted[cls-1,cls]) +', winner:' + str(winner_prob))
    
    return num_correct


def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')

    title_results = []
    date_results = []
    content_results = []

    num_correct = 0
    num_samples = 0
    num_classes = 3
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x[0] = x[0].to(device=DEVICE, dtype=dtype)
            x[1] = x[1].to(device=DEVICE, dtype=dtype)
            x[2] = x[2].to(device=DEVICE, dtype=dtype)
            y = y.view(-1)
            y = y.to(device=DEVICE, dtype=torch.long)
            scores = model(x)
            prob = F.softmax(scores)
            correct = get_acc(scores)
            title_results.append(correct[0])
            date_results.append(correct[1])
            content_results.append(correct[2])
            correct = get_preds(prob)
            num_correct += correct
            num_samples += x[0].size(0)
    
    print('NET TESTING title accuracy' + str(np.mean(title_results)))
    print('NET TESTING date accuracy' + str(np.mean(date_results)))
    print('NET TESTING content accuracy' + str(np.mean(content_results)))

    accuracy = num_correct / (num_samples*num_classes)
    print('(TEST) Got %d / %d correct (%.2f)' % (num_correct, (num_samples*num_classes), 100 * accuracy))

    
def load_checkpoint(checkpoint, model, optimizer):
    print("Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def save_checkpoint(state, filename="model_checkpoint.tar"):
    print("Saving checkpoint")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # Create the directory if it doesn't exist
    checkpoint_path = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(state, checkpoint_path)

def download_page(url):
    print("Downloading: " + str(url))
    temp_dir = tempfile.mkdtemp()
    result = subprocess.run(['python', 'download_page.py',url,temp_dir], check=True)
    return temp_dir

def load_position_maps(position_map_path):
     #--- LOAD SMOTHED POSITION MAPS
    position_maps = []
    for i in range(4):
        path = os.path.join(position_map_path,str(i)+'.pkl')
        position_maps.append(load_position_map(path,sigma=80))
    return position_maps

def load_image_blob(image_path):
    # load image
    im = cv2.imread(image_path)
    size_x = min(im.shape[1],X_SIZE)
    size_y = min(im.shape[0],Y_SIZE)

    # Crop
    im_croped = np.zeros((Y_SIZE,X_SIZE,3),dtype=np.uint8)
    im_croped[:size_y,:size_x,:] = im[:size_y,:size_x,:] 

    n_channels = im.shape[2]
    im_blob = np.zeros((1, Y_SIZE, X_SIZE, n_channels), dtype=np.float32)
    im_blob[0, 0:im_croped.shape[0], 0:im_croped.shape[1], :] = im_croped
    im_blob = im_blob.transpose((0, 3, 1, 2))
    return im_blob



def load_text_blob(leaf_nodes):
    # get text nodes
    text_nodes = data_utils.get_text_nodes(leaf_nodes,N_FEATURES)

    # get text maps
    text_maps = data_utils.get_text_maps(text_nodes, N_FEATURES, SPATIAL_SHAPE, TEXT_MAP_SCALE)

    n_channels = text_maps.shape[2]
    text_blob = np.zeros((1, text_maps.shape[0], text_maps.shape[1], n_channels), dtype=np.float32)
    text_blob[0, 0:text_maps.shape[0], 0:text_maps.shape[1], :] = text_maps
    text_blob = text_blob.transpose((0, 3, 1, 2))
    return text_blob

def load_boxes_blob(leaf_nodes, max_x, max_y):
    # get input boxes
    boxes = np.array([leaf['position'] for leaf in leaf_nodes],dtype = np.float32)
    # remove boxes outside the considered area
    keep_indices = np.logical_and.reduce(((boxes[:,0]>=0), (boxes[:,1]>=0),(boxes[:,2]<=max_x), (boxes[:,3]<=max_y)))
    boxes = boxes[keep_indices,:]
    boxes_this_image = np.hstack((np.zeros((boxes.shape[0], 1)), boxes), dtype=np.float32)

    return boxes_this_image



def show(im_path, boxes_blob, net):
    im = cv2.imread(im_path)
    # im = im[:crop_top,:,:]    
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    
    # colors for particular classes
    colors = ['r','g','b']

    # get predictions with boxes
    predicted = net
    boxes = boxes_blob

    # get probabilities with position likelihoods
    # probs = get_probabilities_with_position(boxes, predicted, position_maps)

    # compute maximum
    # box_class = np.argmax(probs,axis=1)
    max_boxes = np.argmax(predicted,axis=0)


    for cls in range(1,4):
        ind = max_boxes[cls]
        print(predicted[ind])
    
        pred_box = boxes[ind,:]
        _, x1, y1, x2, y2 = pred_box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=True, alpha=0.5, facecolor=colors[cls-1],edgecolor=colors[cls-1], linewidth=3)
        plt.gca().add_patch(rect)

   


    plt.show()


def get_position_probabilities(position_maps, boxes):
    box_i = 0
    probs = np.zeros((boxes.shape[0],len(position_maps)),dtype=np.float32)

    for box_i in range(boxes.shape[0]):
        for cls in range(len(position_maps)):
            map = position_maps[cls]
            box = boxes[box_i]
            box_map = map[box[1]:box[3],box[0]:box[2]]
            box_cls_prob = np.mean(box_map)
            probs[box_i,cls] = box_cls_prob
    return probs

def get_probabilities_with_position(boxes, local_probs, position_maps):
    #-- get position probability for each box
    position_probs = get_position_probabilities(position_maps, boxes)

    #-- multiply with local prob
    probs = (local_probs*position_probs)
    return probs

def get_results_with_position(boxes, local_probs, position_maps):
    #-- get probabalities
    probs = get_probabilities_with_position(boxes, local_probs, position_maps)[:,1:4]

    #-- are the first 3 those with maximum probability
    max_inds = np.argmax(probs,axis=0)
    results = [0]*3
    for cls in range(0,3):
        if max_inds[cls] == cls:
            results[cls]=1

    return results


####---  POSITION MAPS (later)

def create_position_maps(train_data):
    print('Creating and saving position maps')

    n_samples = 40
   
    #--- GET TRAINING DATA
    with open(train_data) as f:
        train_pages = [line.strip() for line in f.readlines()]

    #--- GET RANDOM SUBSET
    sample_pages = random.sample(train_pages, len(train_pages))

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
        path = os.path.join(priors_directory,'_'+str(i)+'.pkl')
        #need to create file directory
        pickle.dump(final_maps[i], open(path,"wb"))


def load_position_map(file_name, sigma):
    map = pickle.load(open(file_name,'rb'))
    ### Gaussian convolution

    #Need to implement
    filtered = T.GaussianBlur(3, sigma)(map)
    return filtered

def load_position_maps(sigma):
    print('Loading position maps smoothed with Gausian filter, sigma=' + str(sigma))
    maps = []
    for i in range(4):
        ### Load map
        #../data_news/position_maps/split_1_i.pkl
        path = os.path.join(priors_directory,'_'+str(i)+'.pkl')
        filtered = load_position_map(path, sigma)
        maps.append(filtered)
    return maps


