from config import *
import util
from data import read_exposure, read_ldr_hdr_images, compute_training_examples, write_training_examples, compute_test_examples, write_test_examples
from itertools import chain
import os
import sys


def preprocess_training_data():
    """
    Preprocess training data
    """
    scene_paths = util.read_dir(TRAINING_RAW_DATA_PATH)
    # make sure we read scene sequentially
    scene_paths = sorted(scene_paths)
    for scene_path in scene_paths:
        exposures = read_exposure(scene_path)
        ldr_imgs, hdr_img = read_ldr_hdr_images(scene_path)
        inputs, label = compute_training_examples(
            ldr_imgs, exposures, hdr_img)
        print(f"processed scene: {scene_path}")

        write_training_examples(
            inputs, label, TRAINING_DATA_PATH, scene_path)


def preprocess_test_data():
    """
    Preprocess test data
    """
    scene_paths = util.read_dir(TEST_RAW_DATA_PATH)
    scene_paths = [util.read_dir(p) for p in scene_paths]
    scene_paths = [p for p in chain.from_iterable(scene_paths)]
    scene_paths = sorted(scene_paths)

    for scene_path in scene_paths:
        exposures = read_exposure(scene_path)
        ldr_imgs, hdr_img = read_ldr_hdr_images(scene_path)
        inputs, label = compute_test_examples(
            ldr_imgs, exposures, hdr_img)
        print(f"processed scene: {scene_path}")

        write_test_examples(
            inputs, label, TEST_DATA_PATH, scene_path)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    if len(sys.argv) < 2:
        preprocess_training_data()
        preprocess_test_data()
    else:
        mode = sys.argv[1]
        if mode == 'train':
            preprocess_training_data()
        elif mode == 'test':
            preprocess_test_data()
        else:
            pass