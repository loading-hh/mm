from mmengine.config import Config
from mmengine.runner import Runner
from mmpretrain.utils import register_all_modules 

if __name__ == "__main__":
    register_all_modules()
    config = Config.fromfile(r'E:\BaiduNetdiskDownload\mm\4-mmpretrain中分类库\config.py')
    runner = Runner.from_cfg(config)
    runner.train()