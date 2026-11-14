import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as transforms
import os
import torch
from PIL import Image as PILImage
class GetDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform
    def __getitem__(self, idx):
        img = self.x[idx]
        label = self.y[idx]
        if self.transform:
            img,label = self.transform([img,label])
        return img, label

    def __len__(self):

        return len(self.y)
