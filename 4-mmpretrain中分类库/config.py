_base_ = "./default_runtime.py"
custom_imports = dict(imports=['mmpretrain'],
                      allow_failed_imports=False)

all_epochs = 10
model = dict(
            type = "MyClassifier",
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
''', data_preprocessor = dict(type = "MyDataPre")'''

work_dir = 'exp/my_awesome_model'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale =256, keep_ratio=True),
    dict(type='RandomFlip', prob=[0.5, 0.5], direction=['horizontal', 'vertical']),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale =256, keep_ratio=True),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(dataset = dict(
                                        type = "MyDataset", 
                                        data_root = r"/home/data1/wjh/FashionMNIST/train",
                                        ann_file = r"annotation/annotations.txt",
                                        pipeline = train_pipeline), 
                        sampler = dict(type = "DefaultSampler", shuffle = True),
                        collate_fn=dict(type = "default_collate"),
                        batch_size=256,
                        num_workers=0)


val_dataloader = dict(dataset = dict(type = "MyDataset", 
                                    data_root = r"/home/data1/wjh/FashionMNIST/test",
                                    ann_file = r"annotation/annotations.txt",
                                    pipeline = test_pipeline), 
                        sampler = dict(type = "DefaultSampler", shuffle = True),
                        collate_fn=dict(type = "default_collate"),
                        batch_size=256,
                        num_workers=0)

val_evaluator = dict(type='Accuracy', topk=(1, 5))
test_dataloader = val_dataloader
test_evaluator = val_evaluator

optim_wrapper = dict(optimizer=dict(type='SGD', lr=0.02))

param_scheduler = dict(type='CosineAnnealingLR', by_epoch=True, T_max = all_epochs)

train_cfg = dict(by_epoch=True, max_epochs=all_epochs, val_begin = 2, val_interval = 1)
val_cfg = dict()
test_cfg = dict()

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2))