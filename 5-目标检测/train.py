import os
from mmengine.config import Config
from mmengine.runner import Runner

if __name__ == "__main__":
    config = Config.fromfile(r'E:\BaiduNetdiskDownload\mm\5-目标检测\yolov3_mobilenetv2_8xb24-ms-416-300e_coco.py')
    runner = Runner.from_cfg(config)
    runner.train()

