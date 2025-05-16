import os
from mmengine.config import Config
from mmengine.runner import Runner
from dataset import MyDataset, MyDataPre
from my_backbone import MyResNet18_BackBone, MyClassifier
from my_neck import MyResNet18_Neck
from my_head import MyResNet18_Head
from mmpretrain.utils import register_all_modules 

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
if __name__ == "__main__":
    register_all_modules()
    config = Config.fromfile(r'/home/data1/wjh/4-mmpretrain中分类库/config.py')
    runner = Runner.from_cfg(config)
    runner.train()

