import torch
import numpy as np
from torch.utils.data import DataLoader
from datas_define import GetDataset  # 导入您的数据加载函数
from torchvision.transforms import Compose
from transforms import *  # 导入预处理转换函数
from model.model import *
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from read_data import x_val, y_val

# 模型和数据路径
CKPT_PATH_G = './model_save/resnet101/3DPET_CNN_Global2_epochs_100.pkl'
N_CLASSES = 1
CLASS_NAMES = ['1', '0']

# 检查是否有GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集和数据加载器
valid_dataset = GetDataset(x_val, y_val, transform=Compose([ToCDHW()]))
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

# 加载模型
model = resnet101(1)
# model = torch.load(CKPT_PATH_G)
model.load_state_dict(torch.load(CKPT_PATH_G))
model.to(device)
model.eval()

# 存储预测和实际标签的变量
all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in valid_loader:
        inputs = inputs.float().to(device)
        labels = labels.float().to(device)
        
        # 预测
        outputs = model(inputs)
        predictions = (outputs > 0.5).float()
        
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 将列表转换为数组，以计算指标
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

# 计算混淆矩阵
conf_matrix = confusion_matrix(all_labels, all_predictions)

# 定义指标计算类
class mymetrics():
    def __init__(self, y_true='', y_predict='', conf=False):
        self.y_true = y_true
        self.y_predict = y_predict
        if conf:
            self.confusion = metrics.confusion_matrix(y_true, y_predict)
        else:
            self.confusion = None

    def addConf(self, confusion):
        self.confusion = confusion

    def acc(self):
        return metrics.accuracy_score(self.y_true, self.y_predict)

    def recall(self):
        TP, TN, FN, FP = self.TP_TN_FN_FP()
        return TP / (TP + FN) if (TP + FN) != 0 else 0

    def precision(self):
        TP, TN, FN, FP = self.TP_TN_FN_FP()
        return TP / (TP + FP) if (TP + FP) != 0 else 0

    def f1(self):
        precision = self.precision()
        recall = self.recall()
        return 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    def TP_TN_FN_FP(self):
        TP = self.confusion[1, 1]
        TN = self.confusion[0, 0]
        FN = self.confusion[1, 0]
        FP = self.confusion[0, 1]
        return TP, TN, FN, FP

    def IoU(self):
        TP, TN, FN, FP = self.TP_TN_FN_FP()
        return TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0

    def Dice(self):
        TP, TN, FN, FP = self.TP_TN_FN_FP()
        return 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0

# 创建 mymetrics 实例并传递正确的参数名
me = mymetrics(y_true=all_labels, y_predict=all_predictions, conf=True)
me.addConf(confusion=conf_matrix)
recall = me.recall()
pre = me.precision()
f1 = me.f1()
acc = me.acc()

# 输出指标
print("acc:", acc)
print("pre:", pre)
print("recall:", recall)
print("f1:", f1)
print(f'混淆矩阵:\n{conf_matrix}')




