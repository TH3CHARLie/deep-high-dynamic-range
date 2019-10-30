import tensorflow as tf
import cv2
import numpy as np
from model import create_model_and_loss, tf_compute_PSNR
import sys
import util
from data import read_test_examples
from config import TEST_DATA_PATH


def test_model():
    model_type = sys.argv[1]
    ckpt_path = sys.argv[2]
    model, loss_function_generator = create_model_and_loss(model_type)
    checkpoint = tf.train.Checkpoint(myModel=model)
    checkpoint.restore(tf.train.latest_checkpoint(
        ckpt_path))
    test_paths = util.read_dir(TEST_DATA_PATH, folder_only=False)
    test_dataset = read_test_examples(test_paths)
    test_dataset = test_dataset.batch(1)
    scene_cnt = 0
    sum_psnr = 0.0
    for (inputs, label) in test_dataset:
        outputs = model(inputs)
        psnr = tf_compute_PSNR(outputs, label)
        sum_psnr += psnr.numpy()
        scene_cnt += 1
    print(f"avg PSNR: {sum_psnr / scene_cnt}")


if __name__ == "__main__":
    test_model()
