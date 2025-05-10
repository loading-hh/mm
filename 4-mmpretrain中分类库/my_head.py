import torch
import torch.nn as nn
import torch.nn.functional as F 

from mmengine.model import BaseModule
from mmpretrain.registry import MODELS

@MODELS.register_module()
class MyResNet18_Head(BaseModule):
    def __init__(self, cls_num, input_channels, loss):
        super().__init__()
        self.cls_num = cls_num
        self.input_channels = input_channels
        self.loss_model = loss
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.input_channels, self.cls_num)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
    def loss(self, feats, data_samples):
        x = self(feats)
        loss = self.loss_model(x, data_samples)
        return dict(loss = loss)
    
    def predict(self, feats, data_samples):
        x = self(feats)
        return  torch.argmax(x, dim = 1), data_samples
        