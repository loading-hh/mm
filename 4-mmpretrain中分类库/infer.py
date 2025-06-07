from mmdet.apis import DetInferencer

# 初始化模型
inferencer = DetInferencer(model=r"/home/data1/wjh/mm/5-目标检测/yolov3_mobilenetv2_8xb24-ms-416-300e_coco.py",
                           weights=r"/home/data1/wjh/mm/5-目标检测/exp/my_awesome_model/epoch_30.pth")

# 推理示例图片
inferencer("/home/data1/wjh/0.png", out_dir = "outputs", no_save_pred=False)