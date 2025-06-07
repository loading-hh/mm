import torch
from torch import nn

from mmdet.registry import MODELS
from mmengine.model import BaseModule, BaseTTAModel
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead

class Convolutional(nn.Module):
    '''
    description: 搭建Convolutional模块
    param {*} self
    param {*} in_channels是输入特征映射图的通道数
    param {*} out_channels是输出特征映射图的通道数
    param {*} kernel_size是卷积核的大小
    param {*} padding是否有填充,有的话为多少
    return {*}
    '''
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride = 1, padding = padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class ConvolutionalBlock(nn.Module):
    '''
    description: 搭建Convolutional块,因为是多个Convolutional块在一块的.
    param {*} self
    param {*} block_num有几个Convolutional块
    param {*} in_channels是每个Convolutional的输入通道数
    param {*} out_channels是每个Convolutional的输出通道数
    param {*} kernel_sizes是每个Convolutional的卷积核大小
    param {*} paddings是每个Convolutional的填充大小
    return {*}
    '''
    def __init__(self, block_num, in_channels, out_channels, kernel_sizes, paddings):
        super().__init__()
        self.block_num = block_num
        for i in range(self.block_num):
            setattr(self, f"blk_{i}", Convolutional(in_channels[i], out_channels[i], kernel_sizes[i], paddings[i]))

    def forward(self, x):
        for i in range(self.block_num):
            x = getattr(self, f"blk_{i}")(x)

        return x

@MODELS.register_module()
class Yolo(nn.Module):
    '''
    description: 搭建my yolo完整模型。
    param {*} self
    param {*} block_list是模型每块有几个ConvolutionalBlock块
    param {*} in_channels是模型每个卷积层的输入通道数列表
    param {*} out_channels是模型每个卷积层的输出通道数列表
    param {*} kernel_sizes是模型每个卷积层的卷积核大小列表
    param {*} paddings是每个卷积层的填充数列表
    param {*} N为最后一个卷积的输出通道数
    return {*}
    '''    
    def __init__(self, block_list, in_channels, out_channels, kernel_sizes, paddings, N):
        super().__init__()
        self.block_num = len(block_list)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv = nn.Conv2d(out_channels[-1][-1], N, 1)

        for i in range(self.block_num):
            if i < (self.block_num - 1):
                setattr(self, f"blk_{i}", nn.Sequential(ConvolutionalBlock(block_list[i], in_channels[i], out_channels[i], kernel_sizes[i], paddings[i]), self.pool))
            else:
                setattr(self, f"blk_{i}", ConvolutionalBlock(block_list[i], in_channels[i], out_channels[i], kernel_sizes[i], paddings[i]))


    def forward(self, x):
        for i in range(self.block_num):
            x = getattr(self, f"blk_{i}")(x)
        x = self.conv(x)

        return x

@MODELS.register_module()
class MyNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.ident = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ident(x)

if __name__ == "__main__":
    block_list = [1, 1, 3, 3, 5, 5]
    in_channels = [[3], [32], [64, 128, 64], [128, 256, 128], [256, 512, 256, 512, 256], [512, 1024, 512, 1024, 512]]
    out_channels = [[32], [64], [128, 64, 128], [256, 128, 256], [512, 256, 512, 256, 512], [1024, 512, 1024, 512, 1024]]
    kernel_sizes = [[3], [3], [3, 1, 3], [3, 1, 3], [3, 1, 3, 1, 3], [3, 1, 3, 1, 3]]
    paddings = [[1], [1], [1, 0, 1], [1, 0, 1], [1, 0, 1, 0, 1], [1, 0, 1, 0, 1]]
    net = Yolo(block_list, in_channels, out_channels, kernel_sizes, paddings, 14)
    x = torch.rand((1, 3, 416, 416))
    print(net(x).shape)
