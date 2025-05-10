import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

from mmengine.model import BaseModel
from mmengine.dataset import BaseDataset, force_full_init
from mmengine.evaluator import BaseMetric
from mmengine.registry import MODELS, DATASETS, METRICS
from mmengine.config import Config
from mmengine.runner import Runner
from mmpretrain import datasets

@MODELS.register_module()
class Mlp(BaseModel):
    def __init__(self, input, hidden, output):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input, hidden)
        self.linear2 = nn.Linear(hidden, output)
    
    def forward(self, image, labels, mode):
        x = self.flatten(image)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        
        if mode == 'tensor':
            return x
        if mode == 'predict':
            return torch.argmax(x, dim = 1), labels
        if mode == 'loss':
            return dict(loss = F.cross_entropy(x, labels))  #pytorch的损失默认都是批量损失的平均值。

@DATASETS.register_module()
class MyDataset(BaseDataset):
    def __init__(self, data_root, ann_file):
        super().__init__(data_root = data_root, ann_file = ann_file)
        
    def load_data_list(self):
        data_list = []
        df = np.loadtxt(os.path.join(self.data_root, self.ann_file), delimiter="\t", dtype = str)
        for i in range(len(df)):
            data_info = {
                        "image_path":os.path.join(self.data_root, "images", df[i][0]), 
                        "label":int(df[i][1])
            }
            data_list.append(data_info)
        return data_list


@DATASETS.register_module()
class MyClass:
    def __init__(self, dataset, lazy_init = False):
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, BaseDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                'elements in datasets sequence should be config or '
                f'`BaseDataset` instance, but got {type(dataset)}')
        # 记录原数据集的元信息
        self._metainfo = self.dataset.metainfo
        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    def full_init(self):
        if self._fully_initialized:
            return
        self.dataset.full_init()
        self._fully_initialized = True

    def __getitem__(self, idx):
        if not self._fully_initialized:
            self.full_init()

        img = cv.imdecode(np.fromfile(self.dataset[idx]["image_path"], dtype=np.uint8), cv.IMREAD_COLOR)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        return (transforms.ToTensor()(img), self.dataset[idx]["label"])

    # 提供与 `self.dataset` 一样的对外接口。
    @force_full_init
    def __len__(self):
        len_wrapper = len(self.dataset)
        return len_wrapper


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
    
if __name__ == "__main__":
    # config = Config.fromfile(r'E:\BaiduNetdiskDownload\mm\2-数据集与数据集加载\config.py')
    # runner = Runner.from_cfg(config)
    # runner.train()

    a = MyDataset(r"E:\BaiduNetdiskDownload\数据集\FashionMNIST\train", r"annotation\annotations.txt")
    b = MyClass(a)
    print(b[0])