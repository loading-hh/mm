import torch
import torch.nn as nn
import torch.nn.functional as F 


from mmengine.model import BaseModule
from mmpretrain.models import ClsHead
from mmpretrain.registry import MODELS

# 因为使用mmpretrain的Accuracy的接口比较繁琐，所以继承mmpretrain的clshead来简化。
@MODELS.register_module()
class MyResNet18_Head(ClsHead):
    def __init__(self, cls_num, input_channels, loss):
        super().__init__(loss=loss)
        self.cls_num = cls_num
        self.input_channels = input_channels
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.input_channels, self.cls_num)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x
        