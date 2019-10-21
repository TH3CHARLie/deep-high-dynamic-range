import os
import numpy as np


def read_dir(path, folder_only=True):
    if folder_only:
        return [f.path for f in os.scandir(path) if f.is_dir()]
    else:
        return [f.path for f in os.scandir(path)]


def im2single(img):
    info = np.iinfo(im.dtype)
    return img.astype(float) / info.max
