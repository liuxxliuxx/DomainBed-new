from torchvision import transforms as T

import numpy as np
from PIL import Image
class WhitenBright:
    """R、G、B 三个通道都大于 thr 的像素，整体置为 (255, 255, 255)"""

    def __init__(self, thr=200):
        self.thr = thr

    def __call__(self, img):
        a = np.array(img)               # H x W x 3, uint8, np.array 会复制一份
        a[(a > self.thr).all(axis=2)] = 255
        return Image.fromarray(a)

_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def basic(size=224):
    return T.Compose([
        T.Resize((size, size)),
        WhitenBright(200),
        T.ToTensor(),
        _NORM,
    ])


def aug(size=224):
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.3),
        T.RandomGrayscale(p=0.1),
        WhitenBright(200),
        T.ToTensor(),
        _NORM,
    ])