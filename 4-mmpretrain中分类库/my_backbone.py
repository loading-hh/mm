import torch
import torch.nn as nn
import torch.nn.functional as F 

from mmpretrain.registry import MODELS
from mmpretrain.models.backbones.base_backbone import BaseBackbone

class Restidual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv = False, strides = 1, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = nn.Conv2d(in_channels = input_channels, out_channels = num_channels, 
                               kernel_size = 3, padding = 1, stride = strides)
        self.conv2 = nn.Conv2d(in_channels = num_channels, out_channels = num_channels, 
                               kernel_size = 3, padding = 1, stride = 1)
        if use_1x1conv == True:
            self.conv3 = nn.Conv2d(in_channels = input_channels, out_channels = num_channels, 
                                   kernel_size = 1, stride = strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
    
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 != None:
            X = self.conv3(X)
        Y += X
        
        return F.relu(Y)
    
def resnet_block(input_channels, num_channels, num_restiduals, first_block = True):
    blk = []
    for i in range(num_restiduals):
        if (i == 0) and (first_block == True):
            blk.append(Restidual(input_channels, num_channels, use_1x1conv = True, strides = 2))
        else:
            blk.append(Restidual(num_channels, num_channels))
    return blk

@MODELS.register_module()
class MyResNet18_BackBone(BaseBackbone):
    def __init__(self, input_channels):
        super().__init__()
        self.input_channels = input_channels
        self.b1 = nn.Sequential(nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block = False))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*resnet_block(256, 512, 2))
        
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        return x