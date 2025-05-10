'''
Author: loading-hh
Date: 2025-04-17 20:57:49
LastEditTime: 2025-04-23 11:25:31
LastEditors: loading-hh
Description: 
FilePath: \mm\第一章-多层感知机\1.1多层感知机\非配置文件\noconfig.py
可以输入预定的版权声明、个性签名、空行等
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

from mmengine.model import BaseModel
from mmengine.evaluator import BaseMetric
from mmengine.registry import MODELS, DATASETS, METRICS

@MODELS.register_module()
class Mlp(BaseModel):
    def __init__(self, input, hidden, output):
        super().__init__()
        self.linear1 = nn.Linear(input, hidden)
        self.linear2 = nn.Linear(hidden, output)
    
    def forward(self, image, labels, mode):
        x = nn.Flatten()(image)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        
        if mode == "tensor":
            return x
        if mode == "predict":
            return torch.argmax(x, dim = 1), labels
        if mode == "loss":
            return {"loss":F.cross_entropy(x, labels)}

@DATASETS.register_module()
def MyTrain():
    return torchvision.datasets.FashionMNIST(root="E:/BaiduNetdiskDownload/mm/data", train=True, 
                                             transform = transforms.ToTensor(), download=True)

@DATASETS.register_module()
def MyTest():
    return torchvision.datasets.FashionMNIST(root="E:/BaiduNetdiskDownload/mm/data", train=False, 
                                             transform = transforms.ToTensor(), download=True)

# trans = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
# @DATASETS.register_module()
# class MyTrain(Dataset):
#     def __init__(self):
#         super().__init__()
#         self.datasets = ImageFolder(root=r'E:\BaiduNetdiskDownload\mm\data\a\train', transform=trans)
    
#     def __getitem__(self, index):
#         data = self.datasets[index][0]
#         label = self.datasets[index][1]
#         return data, label
    
#     def __len__(self):
#         return len(self.datasets)
    
# @DATASETS.register_module()
# class MyTest(Dataset):
#     def __init__(self):
#         super().__init__()
#         self.datasets = ImageFolder(root=r'E:\BaiduNetdiskDownload\mm\data\a\test', transform=trans)
        
#     def __getitem__(self, index):
#         data = self.datasets[index][0]
#         label = self.datasets[index][1]
#         return data, label
    
#     def __len__(self):
#         return len(self.datasets)

@METRICS.register_module()
class Accuracy(BaseMetric):
    def __init__(self):
        super().__init__()

    def process(self, data_batch, data_samples):
        score, gt = data_samples
        self.results.append({
            'batch_size': len(gt),
            'correct': (score == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(r['correct'] for r in results)
        total_size = sum(r['batch_size'] for r in results)
        return dict(accuracy=100*total_correct/total_size)