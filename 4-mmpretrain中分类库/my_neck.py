import torch
import torch.nn as nn
import torch.nn.functional as F 

from mmengine.model import BaseModule
from mmpretrain.registry import MODELS

@MODELS.register_module()
class MyResNet18_Neck(BaseModule):
    def __init__(self, dim = 2):
        super().__init__()
        assert dim in [1, 2, 3], 'GlobalAveragePooling dim only support ' \
            f'{1, 2, 3}, get {dim} instead.'
        if dim == 1:
            self.gap = nn.AdaptiveAvgPool1d(1)
        elif dim == 2:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        
    def forward(self, x):
        return self.gap(x)
        