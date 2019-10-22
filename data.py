import cv2
import numpy as np
import os
import util
from config import Config
from typing import List, Set, Dict, Tuple, Optional

GAMMA = 2.2


def preprocess_training_data(config: Config):
    """
    preprocess training data
    """
    scene_paths = util.read_dir(config.TRAINING_RAW_DATA_PATH)
    # make sure we read scene sequentially
    scene_paths = sorted(scene_paths)
    for scene_path in scene_paths[0:1]:
        exposures = read_exposure(scene_path)
        ldr_imgs, hdr_img = read_ldr_hdr_images(scene_path)
        compute_training_examples(ldr_imgs, exposures, hdr_img, config)
        # write_training_examples(inputs, label, config.TRAINING_DATA_PATH, "TrainingSequence.h5")


def read_exposure(path: str) -> List[float]:
    """Read exposure data from exposures.txt, 

    Args:
        path: A str folder path

    Returns:
        A list of exposure times, empty if error
    """
    paths = [f.path for f in os.scandir(path) if f.name.endswith('.txt')]
    if len(paths) < 1:
        print("[read_exposure]: cannot find exposure file")
        return []
    exposure_file_path = paths[0]
    exposures = []
    with open(exposure_file_path) as f:
        for line in f:
            # exposures are specified in exponent representation
            # thus, return exposure times in 2 ** x
            exposures.append(2 ** float(line))
    return exposures


def read_ldr_hdr_images(path: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    read 3 LDR images and 1 HDR image

    Args:
        path: a str folder path

    Returns:
        A tuple of 
            1: a list of LDR images in np.float32(0-1)
            2: a HDR image in np.float32(0-1)
    """
    paths = [f for f in os.scandir(path)]
    ldr_paths = [x.path for x in paths if x.name.endswith(".tif")]
    # make true we read LDR images based on their exposures
    ldr_paths = sorted(ldr_paths)
    hdr_path = [x.path for x in paths if x.name.endswith(".hdr")]
    if len(ldr_paths) < 3 or len(hdr_path) < 1:
        print("[read_ldr_hdr_images]: cannot find enough ldr/hdr images")
    ldr_imgs = []
    for i in range(3):
        img = util.im2single(cv2.imread(ldr_paths[i], -1))
        # img = util.clamp() TODO: no we really need clamp here
        ldr_imgs.append(img)
    hdr_img = cv2.imread(hdr_path[0], -1)
    return ldr_imgs, hdr_img


def compute_training_examples(ldr_imgs: List[np.ndarray], exposures: List[float], hdr_img: np.ndarray, config: Config):
    prepare_input_features(ldr_imgs, exposures, hdr_img)
    return None


def prepare_input_features(ldr_imgs: List[np.ndarray], exposures: List[float], hdr_img: np.ndarray, is_test: bool = False):
    warpped_ldr_imgs = compute_optical_flow(ldr_imgs, exposures)
    return None


def compute_optical_flow(ldr_imgs: List[np.ndarray], exposures: List[float]) -> List[np.ndarray]:
    """compute optical flow and warp images

    Args:
        ldr_imgs: A list of 3 LDR images
        exposures: A list of 3 corresponding exposure values

    Returns:
        A list of 3 images warpped using optical flow

    Notice: 
        The middle level exposure image is used
        as reference and not warpped
    """
    exposure_adjusted = []
    exposure_adjusted.append(adjust_exposure(ldr_imgs[0:2], exposures[0:2]))
    exposure_adjusted.append(adjust_exposure(ldr_imgs[1:3], exposures[1:3]))

    flow = []
    flow.append(compute_flow(exposure_adjusted[0][1], exposure_adjusted[0][0]))
    flow.append(compute_flow(exposure_adjusted[1][0], exposure_adjusted[1][1]))

    warpped = []
    warpped.append(warp_using_flow(ldr_imgs[0], flow[0]))
    warpped.append(ldr_imgs[1].copy())
    warpped.append(warp_using_flow(ldr_imgs[2], flow[1]))
    return warpped


def compute_flow(prev: np.ndarray, next: np.ndarray) -> np.ndarray:
    """Compute dense optical flow

    Args:
        prev: Reference image
        next: To be warpped image

    Returns:
        A numpy array for estimated flow

    Notice:
        The algorithm can be replaced as long as
        the interface stays unchanged 
    """
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
    prev_gray = util.float2int(prev_gray, np.uint16)
    next_gray = util.float2int(next_gray, np.uint16)

    # TODO: investigate channel layout
    return cv2.calcOpticalFlowFarneback(prev_gray, next_gray, flow=None,
                                        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                                        poly_n=7, poly_sigma=1.2, flags=0)


def warp_using_flow(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp a image using dense optical flow

    Args:
        img: Input image
        flow: Optical flow of the same size

    Returns:
        Warpped image
    """
    h, w, _ = flow.shape
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_CUBIC, borderValue=np.nan)
    return res


def writing_training_examples(inputs, label, path, filename):
    pass


def get_patch_nums(width: int, height: int, config: Config):
    pass


def adjust_exposure(imgs: List[np.ndarray], exposures: List[float]) -> List[np.ndarray]:
    """Adjust image exposure

    Args:
        imgs: A list of images
        exposures: A list of corresponding exposure values

    Returns:
        A list of adjusted images

    Notice:
        The function raise the image with lower exposure to the
        higher one to achieve brightness constancy
    """
    adjusted = []
    max_exposure = max(exposures)
    for i in range(len(imgs)):
        adjusted.append(ldr_to_ldr(imgs[i], exposures[i], max_exposure))
    return adjusted


def ldr_to_ldr(ldr_img: np.ndarray, exposure_src: float, exposure_dst: float) -> np.ndarray:
    """Map a LDR image to a LDR image with different exposure

    Args:
        ldr_img: A LDR image
        exposure_src: Exposure value of the input image
        exposure_dst: Exposure value to raised to

    Returns:
        A image with exposure raised/unchanged(exposure_src == exposure_dst)
    """

    return hdr_to_ldr(ldr_to_hdr(ldr_img, exposure_src), exposure_dst)


def ldr_to_hdr(ldr_img: np.ndarray, exposure: float) -> np.ndarray:
    """Map a LDR image to a HDR image

    Args:
        ldr_img: A LDR image
        exposure: Exposure value of the input image

    Returns:
        A HDR image
    """
    return np.power(ldr_img, GAMMA) / exposure


def hdr_to_ldr(hdr_img: np.ndarray, exposure: float) -> np.ndarray:
    """Map a HDR image to a LDR image

    Args:
        ldr_img: A HDR image
        exposure: Target exposure value

    Returns:
        A LDR image
    """
    hdr_img = hdr_img.astype(np.float32) * exposure
    hdr_img = np.clip(hdr_img, 0, 1)
    return np.power(hdr_img, (1 / GAMMA))
