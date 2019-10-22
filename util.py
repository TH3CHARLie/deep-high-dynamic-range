import os
import numpy as np
from typing import List


def read_dir(path: str, folder_only: bool = True) -> List[str]:
    """Read a directory

    Args:
        path: A str path
        folder_only: Boolean to indicate whether includes folder results only
    
    Returns:
        A list of str of paths
    """
    if folder_only:
        return [f.path for f in os.scandir(path) if f.is_dir()]
    else:
        return [f.path for f in os.scandir(path)]


def im2single(img: np.ndarray) -> np.ndarray:
    """Map a integer image to single-precision float

    Args:
        img: A integer image
    
    Returns:
        A float image
    """
    info = np.iinfo(img.dtype)
    return img.astype(np.float32) / info.max


def im2double(img: np.ndarray) -> np.ndarray:
    """Map a integer image to double-precision float

    Args:
        img: A integer image
    
    Returns:
        A double image
    """
    info = np.iinfo(img.dtype)
    return img.astype(np.float64) / info.max


def compute_PSNR(input: np.ndarray, reference: np.ndarray) -> float:
    """Compute Peak signal-to-noise ratio(PSNR)

    Args:
        input: A produced image
        reference: A reference image 
    
    Returns:
        Error in float
    """
    input = im2single(input)
    reference = im2single(reference)

    num_pixels = input.size
    squared_error = np.sum(np.square(input - reference)) / num_pixels
    error = 10 * np.log10(1 / squared_error)
    return error
