# 按照Dataset的使用规范，构建肺部数据集
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as transforms
import os
import torch
from PIL import Image as PILImage
class GetDataset(Dataset):
    # 肺部扫描数据集
    """
    步骤一：继承paddle.io.Dataset类
    """
    def __init__(self, x, y, transform=None):
        """
        步骤二：实现构造函数，定义数据集大小
        Args:
            x: 图像
            y: 图片存储的文件夹路径
            transform (callable, optional): 应用于图像上的数据处理方法
        """
        self.x = x
        self.y = y
        self.transform = transform # 获取 transform 方法

    def __getitem__(self, idx):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据/测试数据，对应的标签）
        """
        img = self.x[idx]
        label = self.y[idx]
        # 如果定义了transform方法，使用transform方法
        if self.transform:
            img,label = self.transform([img,label])

        # 因为上面我们已经把数据集处理好了生成了numpy形式，没必要处理了
        return img, label

    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return len(self.y) # 返回数据集大小，
