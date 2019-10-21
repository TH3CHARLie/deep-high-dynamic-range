import cv2
import numpy as np
import os
import util
from config import Config
from typing import List, Set, Dict, Tuple, Optional

GAMMA = 2.2

def preprocess_training_data(config: Config):
    """
    preprocess training data, pack it into h5 file
    """
    scene_paths = util.read_dir(config.TRAINING_RAW_DATA_PATH)
    print(scene_paths)
    for scene_path in scene_paths:
        exposures = read_exposure(scene_path)
        print(f"exposures: {exposures}")
        ldr_imgs, hdr_img = read_ldr_hdr_images(scene_path)
        inputs, label = compute_training_examples(ldr_imgs, exposures, hdr_img, config)
        # write_training_examples(inputs, label, config.TRAINING_DATA_PATH, "TrainingSequence.h5")


def read_exposure(path: str) -> List[float]:
    """
    read exposure data from exposures.txt

    exposures are specified in exponent representation
    thus, return 2 ** x
    """
    paths = [f.path for f in os.scandir(path) if f.name.endswith('.txt')]
    if len(paths) < 1:
        print("[read_exposure]: cannot find exposure file")
        return []
    exposure_file_path = paths[0]
    exposures = []
    with open(exposure_file_path) as f:
        for line in f:
            exposures.append(2 ** float(line))
    return exposures


def read_ldr_hdr_images(path: str) -> Tuple[List[np.ndarray], np.ndarray]:
    paths = [f for f in os.scandir(path)]
    ldr_paths = [x.path for x in paths if x.name.endswith(".tif")]
    hdr_path = [x.path for x in paths if x.name.endswith(".hdr")]
    if len(ldr_paths) < 3 or len(hdr_path) < 1:
        print("[read_ldr_hdr_images]: cannot find enough ldr/hdr images")
    ldr_imgs = []
    for i in range(3):
        img = util.im2single(cv2.imread(ldr_paths[i], -1))
        # img = util.clamp() :TODO
        ldr_imgs.append(img)
    hdr_img = cv2.imread(hdr_path[0], -1)
    return ldr_imgs, hdr_img


def compute_training_examples(ldr_imgs: List[np.ndarray], exposures: List[float], hdr_img: np.ndarray, config: Config):
    inputs, labels = prepare_input_features(ldr_imgs, exposures, hdr_img)


def prepare_input_features(ldr_imgs: List[np.ndarray], exposures: List[float], hdr_img: np.ndarray, is_test: bool = False):
    ldr_imgs = compute_optical_flow(ldr_imgs, exposures)


def compute_optical_flow(ldr_imgs: List[np.ndarray], exposures: List[float]):
    pass


def writing_training_examples(inputs, label, path, filename):
    pass


def get_patch_nums(width: int, height: int, config: Config):
    pass


def ldr_to_ldr(ldr_img: np.ndarray, exposure_A: float, exposure_B: float) -> np.ndarray:
    return hdr_to_ldr(ldr_to_hdr(ldr_img, exposure_A), exposure_B)


def ldr_to_hdr(ldr_img: np.ndarray, exposure: float) -> np.ndarray:
    return np.power(ldr_img, GAMMA) / exposure


def hdr_to_ldr(ldr_img: np.ndarray, exposure: float): -> np.ndarray:
    ldr_img = ldr_img.astype(float) * exposure
    ldr_img = np.clip(ldr_img, 0, 1)
    return np.power(ldr_img, (1/GAMMA))


