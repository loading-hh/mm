import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Sequence

def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def creat(read_path: str, classes :Tuple[str], save_path: str) -> None:
    # metainfo的信息
    metainfo = dict(classes = classes)
    # 读取原csv
    data = pd.read_csv(os.path.join(read_path, "label.csv"))
    # data_list的信息
    images = list()
    annotations = list()
    categories = list()
    data = data.rename(columns={"xmax":"weight", "ymax":"height"})
    data["weight"] = data["weight"] - data["xmin"]
    data["height"] = data["height"] - data["ymin"]
    # data.to_csv("./a.csv", index=False)
    for i in range(len(data)):
        # images中的内容
        file_name = os.path.join(read_path, "images", data.iloc[i, 0])
        height = 256
        width = 256
        id = i
        instances = []
        a = dict(file_name = file_name, height = height, width = width, id = id)
        images.append(a)
        # annotations中的内容
        area = data["weight"][i] * data["height"][i]
        iscrowd = 0
        image_id = i
        bbox = data.iloc[i, 2:].tolist()
        category_id = data.iloc[i, 1]
        b = dict(area = area, iscrowd = iscrowd, image_id = image_id, bbox = bbox, category_id = category_id, id = id)
        annotations.append(b)
    # categories中的内容
    for i in range(len(classes)):
        c = dict(id = i, name = classes[i])
        categories.append(c)
    # 总字典
    all_data = dict(images = images, annotations = annotations, categories = categories)
    all_data = json.dumps(all_data, indent=4, ensure_ascii=False, default=default_dump)
    # json中文不为乱码
    with open(save_path, 'w', encoding="utf8", newline='\n') as f:
 	    f.write(all_data)
    
if __name__ == "__main__":
    classes = ['banana']
    creat(r"E:\BaiduNetdiskDownload\dataset\banana-detection\banana-detection\bananas_train", classes, 
          r"E:\BaiduNetdiskDownload\dataset\banana-detection\banana-detection\bananas_train\train.json")