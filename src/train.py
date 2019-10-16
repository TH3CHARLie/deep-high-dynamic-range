import numpy as np
import os
import tensorflow as tf
import cv2

from model import DHDRCNN

def train_main():
    model = DHDRCNN()

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    train_main()