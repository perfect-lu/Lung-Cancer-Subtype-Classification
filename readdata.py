# import os
# import numpy as np
# import nibabel as nib
# from data import *
# import random
# import pandas as pd
# from sklearn.model_selection import train_test_split
# # 让我们从类目录中读取PET扫描的路径。

# # 文件夹"CT-0"由具有正常肺组织的CT扫描组成，没有病毒性肺炎的CT征象。
# normal_scan_paths = [
#     os.path.join(os.getcwd(), "MosMedData_1/AI_PET_SUV", x)
#     for x in os.listdir("MosMedData_1/AI_PET_SUV")
# ]
# # 文件夹 "CT-23 "包括有几个磨玻璃不透明的CT扫描件，肺实质受累。
# abnormal_scan_paths = [
#     os.path.join(os.getcwd(), "MosMedData_1/HE_PET_SUV", x)
#     for x in os.listdir("MosMedData_1/HE_PET_SUV")
# ]

# print("PET scans with normal lung tissue: " + str(len(normal_scan_paths)))
# print("PET scans with abnormal lung tissue: " + str(len(abnormal_scan_paths)))

# def read_nifti_file(filepath):
#     """读取和加载数据"""
#     scan = nib.load(filepath)
#     scan = scan.get_fdata()
#     return scan

# # 从目录中读取扫描并分配标签。扫描图像以具有128x128x128的形状。最后将数据集分成训练和验证子集。
# # abnormal_scans = np.array([process_scan(path) for path in abnormal_scan_paths])
# # normal_scans = np.array([process_scan(path) for path in normal_scan_paths])
# # normal_scans = load_scans(normal_scan_paths)
# # abnormal_scans = load_scans(abnormal_scan_paths)
# normal_scans = np.array([read_nifti_file(path) for path in normal_scan_paths])
# abnormal_scans = np.array([read_nifti_file(path) for path in abnormal_scan_paths])

# # 对于具有病毒性肺炎的CT扫描分配1，对于正常分配0。
# abnormal_labels = np.array([1 for _ in range(len(abnormal_scan_paths))])
# normal_labels = np.array([0 for _ in range(len(normal_scan_paths))])

# # 将数据随机分割为训练集和验证集
# # 训练集为70%，验证集为30%
# all_scans = np.concatenate((abnormal_scans, normal_scans), axis=0)
# all_labels = np.concatenate((abnormal_labels, normal_labels), axis=0)

# # 随机打乱数据
# # indices = list(range(len(all_scans)))
# indices = np.arange(all_scans.shape[0])
# np.random.shuffle(indices)
# all_scans = all_scans[indices]
# all_labels = all_labels[indices]

# # 计算训练集和验证集的大小
# train_size = int(0.7 * len(all_scans))
# # x_train, x_val = all_scans[:train_size],all_scans[train_size:]
# # y_train, y_val = all_labels[:train_size],all_labels[train_size:]
# val_size = all_scans.shape[0] - train_size

# x_train, x_val = all_scans[:train_size],all_scans[train_size:]
# y_train, y_val = all_labels[:train_size],all_labels[train_size:]
# # 将训练集和验证集保存到CSV文件

# train_data = pd.DataFrame({'scan_path': list(x_train),
#                             'Label': y_train
#                            })
# val_data = pd.DataFrame({'scan_path': list(x_val),
#                          'Label': y_val
#                          })

# train_data.to_csv("./dataset/train_data3.csv", index=False)
# val_data.to_csv("./dataset/test_data3.csv", index=False)

# print(
#     "数据已保存，训练集样本数：%d,测试集样本数：%d "%(len(x_train),len(x_val))
# )

import os
import numpy as np
import nibabel as nib
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取PET扫描的路径
normal_scan_paths = [
    os.path.join(os.getcwd(), "MosMedData_1/AI_PET_SUV", x)
    for x in os.listdir("MosMedData_1/AI_PET_SUV")
]
abnormal_scan_paths = [
    os.path.join(os.getcwd(), "MosMedData_1/HE_PET_SUV", x)
    for x in os.listdir("MosMedData_1/HE_PET_SUV")
]

print("PET scans with normal lung tissue: " + str(len(normal_scan_paths)))
print("PET scans with abnormal lung tissue: " + str(len(abnormal_scan_paths)))

def read_nifti_file(filepath):
    """读取和加载数据"""
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    return scan

# 读取扫描并分配标签
normal_scans = np.array([read_nifti_file(path) for path in normal_scan_paths])
abnormal_scans = np.array([read_nifti_file(path) for path in abnormal_scan_paths])

abnormal_labels = np.array([1 for _ in range(len(abnormal_scan_paths))])
normal_labels = np.array([0 for _ in range(len(normal_scan_paths))])

# 合并数据和标签
all_scans = np.concatenate((abnormal_scans, normal_scans), axis=0)
all_labels = np.concatenate((abnormal_labels, normal_labels), axis=0)
all_paths = np.concatenate((abnormal_scan_paths, normal_scan_paths), axis=0)

# 随机打乱数据
indices = np.arange(all_scans.shape[0])
np.random.shuffle(indices)
all_scans = all_scans[indices]
all_labels = all_labels[indices]
all_paths = all_paths[indices]

# 分割数据集
train_size = int(0.7 * len(all_scans))
x_train, x_val = all_scans[:train_size], all_scans[train_size:]
y_train, y_val = all_labels[:train_size], all_labels[train_size:]
paths_train, paths_val = all_paths[:train_size], all_paths[train_size:]

# 将训练集和验证集保存到CSV文件
train_data = pd.DataFrame({'scan_path': paths_train,
                           'Label': y_train
                          })
val_data = pd.DataFrame({'scan_path': paths_val,
                         'Label': y_val
                        })

train_data.to_csv("./dataset/train_data2.csv", index=False)
val_data.to_csv("./dataset/test_data2.csv", index=False)

print("数据已保存，训练集样本数：%d,测试集样本数：%d" % (len(x_train), len(x_val)))