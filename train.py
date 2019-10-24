import numpy as np
import os
import tensorflow as tf
import cv2
from model import DHDRCNN
from config import Config
from data import read_training_examples
import util
def train_main(config: Config):

    model = DHDRCNN()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.ADAM_LEARNING_RATE,
        beta_1=config.ADAM_BETA1,
        beta_2=config.ADAM_BETA2)
    paths = util.read_dir(config.TRAINING_DATA_PATH, folder_only=False)
    model.compile(
        
    )
    # dataset = read_training_examples()
    # dataset = dataset.batch(config.BATCH_SIZE)
    # dataset = dataset.prefetch(config.PREFETCH)

    # for i in range(config.ITERATION):



if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    cfg = Config()
    train_main(cfg)
