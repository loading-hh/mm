import torch
from torch import nn

from mmdet.registry import MODELS
from mmengine.model import BaseModule, BaseTTAModel
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead

@MODELS.register_module()
class MyHead(BaseDenseHead):
    def __init__(self, init_cfg = None):
        super().__init__(init_cfg)