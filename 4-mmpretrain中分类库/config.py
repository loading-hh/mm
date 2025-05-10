_base_ = "../mmpretrain/configs/_base_/default_runtime.py"

model = dict(
            type = "ImageClassifier", 
            backbone = dict(
                            type = "MyResNet18_BackBone", 
                            input_channels = 3), 
            neck = dict(
                        type = "MyResNet18_Neck"), 
            head = dict(
                        type = "MyResNet18_Head", 
                        cls_num = 10, 
                        input_channels = 512, 
                        loss = dict(type='CrossEntropyLoss', loss_weight=1.0)), 
            data_preprocessor = dict(type = "MyDataPre"))

work_dir = 'exp/my_awesome_model'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=256, keep_ratio=True),
    dict(type='RandomFlip', prob=[0.5, 0.5], direction=['horizontal', 'vertical']),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=256, keep_ratio=True),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(dataset = dict(
                                        type = "MyDataset", 
                                        data_root = r"E:\BaiduNetdiskDownload\数据集\FashionMNIST\train", 
                                        ann_file = r"annotation\annotations.txt", 
                                        pipeline = train_pipeline), 
                        sampler = dict(type = "DefaultSampler", shuffle = True),
                        collate_fn=dict(type = "default_collate"),
                        batch_size=128,
                        num_workers=0)


val_dataloader = dict(dataset = dict(type = "MyDataset", 
                                    data_root = r"E:\BaiduNetdiskDownload\数据集\FashionMNIST\test", 
                                    ann_file = r"annotation\annotations.txt"), 
                        sampler = dict(type = "DefaultSampler", shuffle = True),
                        collate_fn=dict(type = "default_collate"),
                        batch_size=128,
                        num_workers=0)

val_evaluator = dict(type='Accuracy', topk=(1, 5))
test_dataloader = val_dataloader
test_evaluator = val_evaluator

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))

param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[10, 15], gamma=0.1)

train_cfg = dict(by_epoch=True, max_epochs=20, val_begin = 5, val_interval = 2)
val_cfg = dict()
test_cfg = dict()

default_scope = 'mmpretrain'
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2))