import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 

from mmengine.dataset import BaseDataset
from mmpretrain.registry import MODELS, DATASETS


@DATASETS.register_module()
class MyDataset(BaseDataset):
    def __init__(self, data_root, ann_file, pipeline):
        super().__init__(data_root = data_root, ann_file = ann_file, pipeline = pipeline)
        
    def load_data_list(self):
        data_list = []
        df = np.loadtxt(os.path.join(self.data_root, self.ann_file), delimiter="\t", dtype = str)
        for i in range(len(df)):
            data_info = {
                        "img_path":os.path.join(self.data_root, "images", df[i][0]), 
                        "gt_label":int(df[i][1])
            }
            data_list.append(data_info)
        return data_list
    
@MODELS.register_module()
class MyDataPre(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, datas, training = True):
        
        datas["inputs"] = datas["inputs"].cuda()
        if training is True:
            for label in datas["data_samples"]:
                label.gt_label = label.gt_label.cuda()
        if training is not True:
            datas["inputs"] = datas["inputs"] * 1.0
            return datas  #模型要的是浮点型，不能是整形。
        
        datas["inputs"] = datas["inputs"] * 1.0  #在这个进行除以255，是因为在没有在MyClass中的getitem中进行transform.Totensor(),这是因为归一化这类计算量较低的操作,其耗时会远低于数据搬运，
                            #transform.Totensor()后是float32类型，而原来是int8类型。如果我能够在数据仍处于 uint8 时、归一化之前将其搬运到指定设备上
                            #(归一化后的 float 型数据大小是 unit8 的 4 倍)，就能降低带宽，大大 提升数据搬运的效率。除以255的操作与transform.Totensor()
                            #中归一化操作是一样的。你可以进行对比，会发现速度确实提升了。
        return datas