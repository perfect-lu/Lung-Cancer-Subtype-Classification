#CT扫描也通过在训练期间进行随机角度旋转来增强数据。由于数据存储在Rank - 3
#的形状HWD（高度，宽度，深度）中，因此我们先把图像格式变为DHW，再在轴0处添加大小1的尺寸，以便能够对数据执行3D卷积。因此，新形状（1，深度，高度，宽度）。还有不同类型的预处理和增强技术，这个例子通过自定义transform进行了一些简单的操作。
#
# class TransformAPI(object):
#     """
#     步骤一：继承 object 类
#     """
#     def __call__(self, data):
#
#         """
#         步骤二：在 __call__ 中定义数据处理方法
#         """
#
#         processed_data = data
#         return  processed_data
# ```
#

import numpy as np
import torch
import random
from scipy import ndimage
import torchvision.transforms.functional as F


# # 将图像旋转me
# class Rotate(object):

#     def __call__(self, data):
#         image = data[0]
#         key_pts = data[1]
#         # 定义一些旋转角度
#         angles = [-20, -10, -5, 5, 10, 20]
#         # 随机挑选角度
#         angle = random.choice(angles)
#         # 旋转体积
#         image = ndimage.rotate(image, angle, reshape=False)
#         image[image < 0] = 0
#         image[image > 1] = 1
#         return image, key_pts


# # 将图像的格式由HWD改为CDHW
# class ToCDHW(object):

#     def __call__(self, data):
#         image = data[0]
#         key_pts = data[1]
#         # image = torch.transpose(torch.to_tensor(image), perm=[2, 0, 1])
#         image = torch.tensor(image).permute(2, 0, 1)
#         image = np.expand_dims(image, axis=0)
#         return image, key_pts


# 将图像旋转
class Rotate(object):

    def __call__(self, data):
        image = data[0]
        key_pts = data[1]

        # 定义一些旋转角度
        angles = [-20, -10, -5, 5, 10, 20]
        # 随机挑选角度
        angle = random.choice(angles)

        # 旋转体积
        image = ndimage.rotate(image, angle, reshape=False)

        # 保证数值在 [0, 1] 范围内
        image[image < 0] = 0
        image[image > 1] = 1

        return image, key_pts


# 将图像的格式由 HWD 改为 CDHW
class ToCDHW(object):

    def __call__(self, data):
        image = data[0]
        key_pts = data[1]

        # 检查图像的维度
        if image.ndim == 3:
            # 如果是 3D 图像 (H, W, D)，使用 permute 转换为 (C, H, W, D)
            image = torch.tensor(image).permute(2, 0, 1)
            # 增加批量维度 (C, H, W) -> (1, C, H, W)
            image = image.unsqueeze(0)
        elif image.ndim == 2:
            # 如果是 2D 图像 (H, W)，则添加通道维度 (1, H, W)
            image = torch.tensor(image).unsqueeze(0)

        return image, key_pts