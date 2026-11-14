import numpy as np
import torch
import random
from scipy import ndimage
import torchvision.transforms.functional as F
class Rotate(object):

    def __call__(self, data):
        image = data[0]
        key_pts = data[1]
        angles = [-20, -10, -5, 5, 10, 20]
        angle = random.choice(angles)
        image = ndimage.rotate(image, angle, reshape=False)
        image[image < 0] = 0
        image[image > 1] = 1

        return image, key_pts

class ToCDHW(object):

    def __call__(self, data):
        image = data[0]
        key_pts = data[1]

        if image.ndim == 3:
            image = torch.tensor(image).permute(2, 0, 1)
            image = image.unsqueeze(0)
        elif image.ndim == 2:
            image = torch.tensor(image).unsqueeze(0)

        return image, key_pts
