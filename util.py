import os
import numpy as np


def read_dir(path: str, folder_only: bool = True):
    if folder_only:
        return [f.path for f in os.scandir(path) if f.is_dir()]
    else:
        return [f.path for f in os.scandir(path)]


def im2single(img: np.ndarray):
    info = np.iinfo(img.dtype)
    return img.astype(np.float32) / info.max


def im2double(img: np.ndarray):
    info = np.iinfo(img.dtype)
    return img.astype(np.float64) / info.max


def compute_PSNR(input: np.ndarray, reference: np.ndarray):
    input = im2single(input)
    reference = im2single(reference)

    num_pixels = input.size
    squared_error = np.sum(np.square(input - reference)) / num_pixels
    error = 10 * np.log10(1 / squared_error)
    return error
