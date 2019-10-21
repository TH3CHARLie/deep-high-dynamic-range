import numpy as np
import os
import tensorflow as tf
import cv2
from model import DHDRCNN
from config import Config


def train_main(config: Config):

    model = DHDRCNN()

    # optimizer = tf.keras.optimizers.Adam(learning_rate=config.adam_alhpa, beta_1=adam_beta1, beta_2=adam_beta2)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    cfg = Config()
    train_main(cfg)
