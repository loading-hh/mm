import torch
from mmpretrain import get_model, inference_model
from mmpretrain import ImageClassificationInferencer
from dataset import MyDataset, MyDataPre
from my_backbone import MyResNet18_BackBone, MyClassifier
from my_neck import MyResNet18_Neck
from my_head import MyResNet18_Head

image = r"/home/data1/wjh/FashionMNIST/test/images/0.png"
config = r"/home/data1/wjh/mm/4-mmpretrain中分类库/config.py"
checkpoint = r"/home/data1/wjh/mm/4-mmpretrain中分类库/exp/my_awesome_model/epoch_20.pth"
inferencer = ImageClassificationInferencer(model=config, pretrained=checkpoint, device='cuda')
result = inferencer(image)
print(result)