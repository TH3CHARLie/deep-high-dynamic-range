"""
DHDRCNN

Configuration file
"""

MU = 5000
GAMMA = 2.2
WEIGHT_EPS = 1e-6

PATCH_SIZE: int = 40
STRIDE: int = 20
CROP_SIZE: int = 50
BATCH_SIZE: int = 20
NUM_AUGMENT: int = 10
NUM_TOTAL_AUGMENT: int = 48
BORDER: int = 6
EPS = 1e-6
ITERATION = 2000000
PREFETCH = 1

# Adam optimizer parameters
ADAM_LEARNING_RATE = 0.0001
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

# data path
TRAINING_RAW_DATA_PATH = "./data/Training/"
TRAINING_DATA_PATH = "./tf-data/train-deepflow/"
TEST_RAW_DATA_PATH = "./data/Test/"
TEST_DATA_PATH = "./tf-data/test-deepflow/"
SAVE_PATH = "./save/"
TENSORBOARD_PATH = "./tensorboard/"