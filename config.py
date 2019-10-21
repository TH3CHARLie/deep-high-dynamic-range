"""
DHDRCNN

Configuration class
"""
class Config:
    
    PATCH_SIZE = 40
    STRIDE = 20
    CROP_SIZE = 50
    BATCH_SIZE = 20
    NUM_AUGMENT = 20
    NUM_TOTAL_AUGMENT = 48
    BORDER = 6
    EPS = 1e-6

    # Adam optimizer parameters
    ADAM_LEARNING_RATE = 0.0001
    ADAM_BETA1 = 0.9
    ADAM_BETA2 = 0.999
    ADAM_EPS = 1e-8

    # data path
    TRAINING_RAW_DATA_PATH = "./data/Training/"
    TRAINING_DATA_PATH = "./data/train/"