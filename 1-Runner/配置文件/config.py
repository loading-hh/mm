model = dict(type = "Mlp", 
             input = 784, 
             hidden=256, 
             output=10)

work_dir='exp/my_awesome_model'

train_dataloader = dict(dataset = dict(type = "MyTrain"), 
                        sampler = dict(type = "DefaultSampler", shuffle = True),
                        collate_fn=dict(type = "default_collate"),
                        batch_size=128,
                        pin_memory=True,
                        num_workers=0)
train_cfg = dict(
    by_epoch=True,
    max_epochs=10,
    val_begin=1,
    val_interval=1)     

optim_wrapper = dict(
    optimizer=dict(
        type='Adam',
        lr=0.001))

val_dataloader = dict(
    dataset=dict(type='MyTest'),
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),
    collate_fn=dict(type='default_collate'),
    batch_size=128,
    pin_memory=True,
    num_workers=0)

val_cfg = dict()

val_evaluator = dict(type='Accuracy')
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2))
launcher = 'none'
env_cfg = dict(
    cudnn_benchmark=False,
    backend='nccl',
    mp_cfg=dict(mp_start_method='fork'))
log_level = 'INFO'
load_from = None
resume = False

