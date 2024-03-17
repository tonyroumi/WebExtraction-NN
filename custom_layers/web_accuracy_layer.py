import numpy as np
from random import randint
import torch.nn as nn

class ReshapeWebAccuracy(nn.Module):
    def reshape(self, top):
        for i in range(0,3):
            top[i] = top[i].view(1)

class WebAccuracyLayer(nn.Module):
    
    def __init__(self, top):
        super().__init__()
        for i in range(0,3):
            top[i] = top[i].view(1)

    #bottom is the probability
    #top is 'web_price_accuracy'
    # top: 'web_image_accuracy'
    # top: 'web_name_accuracy'

    def forward(self, bottom, top):
        
        ### get results and boxes
        predicted = bottom[0].data[:,1:4]
        max_inds = np.argmax(predicted,axis=0)

        for cls in range(0,3):
           if max_inds[cls] == cls:
               top[cls].data[0] = 1
           else:
               top[cls].data[0] = 0

    def backward(self, top, propagate_down, bottom):
        pass
