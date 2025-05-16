# 导入所需库
import os  # 操作系统相关的功能，如路径处理和目录创建
import struct  # 用于解析二进制数据结构
import numpy as np  # 数值计算库，用于高效数组操作
import cv2  # OpenCV库，用于图像处理
import gzip  # 压缩文件处理库，用于读取.gz压缩文件
 
 
# 解码idx3-ubyte.gz格式的MNIST图像文件
def decode_idx3_ubyte(idx3_ubyte_gz_file):
    # 使用gzip打开并读取文件内容
    with gzip.open(idx3_ubyte_gz_file, 'rb') as f:
        print('解析文件：', idx3_ubyte_gz_file)
        fb_data = f.read()
 
    # 定义文件头格式，读取魔数和图像数据的基本信息
    offset = 0
    fmt_header = '>IIII'  # 大端序，四个无符号整型
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, fb_data, offset)
    print('魔数：{}，图片数：{}'.format(magic_number, num_images))
    offset += struct.calcsize(fmt_header)
 
    # 确保图片数量为正数
    if num_images < 0:
        raise ValueError("图片数量解析为负数，文件可能损坏或格式不正确。")
 
    # 定义图像数据格式并解析所有图像
    fmt_image = '>' + str(num_rows * num_cols) + 'B'  # 图像数据为无符号字节序列
    images = np.empty((num_images, num_rows, num_cols), dtype=np.uint8)
    for i in range(num_images):
        im = struct.unpack_from(fmt_image, fb_data, offset)
        images[i] = np.array(im).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images
 
 
# 解码idx1-ubyte.gz格式的MNIST标签文件
def decode_idx1_ubyte(idx1_ubyte_gz_file):
    with gzip.open(idx1_ubyte_gz_file, 'rb') as f:
        print('解析文件：', idx1_ubyte_gz_file)
        fb_data = f.read()
 
    offset = 0
    fmt_header = '>II'  # 两个无符号整型，分别对应魔数和标签数量
    magic_number, label_num = struct.unpack_from(fmt_header, fb_data, offset)
    print('魔数：{}，标签数：{}'.format(magic_number, label_num))
 
    # 验证魔数是否为MNIST标签文件的正确值
    if magic_number != 2049:
        raise ValueError("标签文件的魔数不正确，可能不是MNIST数据集的标签文件。")
 
    offset += struct.calcsize(fmt_header)
 
    # 确保标签数量非负
    if label_num < 0:
        raise ValueError("解析到的标签数为负数，文件可能损坏。")
 
    fmt_label = '>' + str(label_num) + 'B'  # 标签数据为字节序列
    labels = struct.unpack_from(fmt_label, fb_data, offset)
    return np.array(labels)
 
 
# 检查并创建目录
def check_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f'创建目录：{folder}')
 
 
# 将解析出的图像和标签导出为图片文件
def export_img(exp_dir, img_ubyte, lbl_ubyte):
    check_folder(exp_dir)
    images = decode_idx3_ubyte(img_ubyte)
    labels = decode_idx1_ubyte(lbl_ubyte)
 
    for i, label in enumerate(labels):
        img_dir = os.path.join(exp_dir, f'{label}')  # 根据标签创建子目录
        check_folder(img_dir)
        img_file = os.path.join(img_dir, f'{i}.png')  # 图片文件名
        # 归一化图像数据并保存为PNG
        img_normalized = (images[i] - images[i].min()) * (255 / (images[i].max() - images[i].min()))
        cv2.imwrite(img_file, img_normalized.astype(np.uint8))  # 写入图像文件
 
 
# 处理整个MNIST数据集，包括训练集和测试集
def parser_mnist_data(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    train_img_ubyte = os.path.join(train_dir, 'train-images-idx3-ubyte.gz')
    train_label_ubyte = os.path.join(train_dir, 'train-labels-idx1-ubyte.gz')
    print(train_img_ubyte, train_label_ubyte)
    export_img(train_dir, train_img_ubyte, train_label_ubyte)
 
    test_dir = os.path.join(data_dir, 'test')
    test_img_ubyte = os.path.join(test_dir, 't10k-images-idx3-ubyte.gz')
    test_label_ubyte = os.path.join(test_dir, 't10k-labels-idx1-ubyte.gz')
    export_img(test_dir, test_img_ubyte, test_label_ubyte)
 
 
# 主程序入口
if __name__ == '__main__':
    data_dir = r'E:\BaiduNetdiskDownload\mm\data\a'    # MNIST数据集的根目录
    parser_mnist_data(data_dir)     # 开始处理数据集