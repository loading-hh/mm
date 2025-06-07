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

#作为数据预处理器的部分
class MyDataPre(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, data, training = True):
        data = list(_data.cuda() for _data in data)
        if training is not True:
            imgs, labels = data
            imgs = imgs * 1.0
            return tuple([imgs, labels])  #模型要的是浮点型，不能是整形。
        
        imgs, labels = data
        imgs = imgs / 255  #在这个进行除以255，是因为在没有在MyClass中的getitem中进行transform.Totensor(),这是因为归一化这类计算量较低的操作,其耗时会远低于数据搬运，
                            #transform.Totensor()后是float32类型，而原来是int8类型。如果我能够在数据仍处于 uint8 时、归一化之前将其搬运到指定设备上
                            #(归一化后的 float 型数据大小是 unit8 的 4 倍)，就能降低带宽，大大 提升数据搬运的效率。除以255的操作与transform.Totensor()
                            #中归一化操作是一样的。你可以进行对比，会发现速度确实提升了。
        return tuple([imgs, labels])
    
    
@MODELS.register_module()
class Mlp(BaseModel):
    def __init__(self, input, hidden, output):
        super().__init__(data_preprocessor=MyDataPre())
        # self.datapreprocessor = MyDataPre()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear1 = nn.Linear(input, hidden)
        self.linear2 = nn.Linear(hidden, output)
    
    def forward(self, image, labels, mode):
        x = self.flatten(image)
        x = F.relu(self.linear1(x))
        x = self.linear2(x).squeeze()
        if mode == 'tensor':
            return x
        if mode == 'predict':
            return torch.argmax(x, dim = 1), labels
        if mode == 'loss':
            return {"loss":F.cross_entropy(x, labels)}
    
    def train_step(self, data, optim_wrapper):
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data)
            # data = self.datapreprocessor(data)
            if isinstance(data, (list, tuple)):
                losses = self(*data, mode = "loss")
            elif isinstance(data, dict):
                losses = self(**data, mode = "loss")
        parsed_losses, log_vars = self.parse_losses(losses)
        optim_wrapper.update_params(parsed_losses)
        return log_vars
    
    def val_step(self, data):
        data = self.data_preprocessor(data, False)
        # data = self.datapreprocessor(data, False)
        if isinstance(data, (list, tuple)):
            output = self(*data, mode = "predict")
        elif isinstance(data, dict):
            output = self(**data, mode = "predict")
        return output
    
    def test_step(self, data):
        data = self.data_preprocessor(data, False)
        # data = self.datapreprocessor(data, False)
        if isinstance(data, (list, tuple)):
            output = self(*data, mode = "predict")
        elif isinstance(data, dict):
            output = self(**data, mode = "predict")
        return output
    
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
        img = np.expand_dims(img, axis=0)
        
        return (img, self.dataset[idx]["label"])

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
    config = Config.fromfile(r'E:\BaiduNetdiskDownload\mm\3-模型model\config.py')
    runner = Runner.from_cfg(config)
    runner.train()