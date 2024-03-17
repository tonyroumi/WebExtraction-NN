import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_layers.web_accuracy_layer import WebAccuracyLayer
import torch.nn.init as init
from torchvision import ops
import numpy as np





class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
    
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=5)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2, groups=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        init.normal_(self.conv3.weight, mean=0.0, std=0.01)
        init.constant_(self.conv3.bias, val=0.1)
        self.txt_conv = nn.Conv2d(in_channels=128, out_channels=48, kernel_size=1, padding=0, stride=1)
        init.normal_(self.txt_conv.weight, mean=0.0, std=0.01)
        init.constant_(self.txt_conv.bias, val=0.1)
        self.concat = ConcatLayer()
        self.both_conv = nn.Conv2d(in_channels=432, out_channels=96, kernel_size=5, padding=2, groups=2)
        init.normal_(self.both_conv.weight, mean=0.0, std=0.01)
        init.constant_(self.both_conv.bias, val=0.1)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #input is boxes
        # need to figure out # of input channels for conv classifier
        self.conv_classifier = nn.Conv2d(out_channels=4, in_channels=96, kernel_size=1)
        init.kaiming_normal_(self.conv_classifier.weight)
        init.constant_(self.conv_classifier.bias, val=0)

        #Missing custom accuracy layer, custom cross entropy loss
        self.reshape = Reshape()

        self.relu = nn.ReLU()

        self.lrn_norm = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)
        #  top: 'web_price_accuracy'
            # top: 'web_image_accuracy'
            # top: 'web_name_accuracy'
        # self.accuracy_layer = WebAccuracyLayer(self.top)
        # self.roi_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        txt = x[1]
        boxes = x[2]
        boxes = self.reshape(boxes)
       
        

        # Convolutional layers
        x = self.lrn_norm(self.pool(self.relu(self.conv1(x[0]))))
        x = self.lrn_norm(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        y = self.relu(self.txt_conv(txt))
        x = self.concat(x,y)
        x = ops.roi_pool(input=self.relu(self.both_conv(x)), boxes=boxes, output_size=(1,1), spatial_scale=0.125)
        x = self.conv_classifier(x)

        #potentially return this ^, then use 
       
       #here they calculate softmax loss w custom loss param input is conv_class, labels

       #custom accuracy layer, input is conv_classifier, labels, output is accuracy, per_class_accuracy

        # x = self.soft(x)
        # x = self.accuracy_layer(x)

        
        return x.view(200,4)


class ConcatLayer(nn.Module):
    def forward(self, input1, input2):
        return torch.cat((input1, input2), dim=1)
    
class Reshape(nn.Module):
    def forward(self, boxes):
        batch_ind = torch.arange(boxes.shape[0]).unsqueeze(1).expand(-1, boxes.shape[1])
        boxes_with_batch = torch.cat((batch_ind.unsqueeze(-1), boxes), dim=-1)
        return boxes_with_batch.view(-1, 5)
