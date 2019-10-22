from config import Config
from data import preprocess_training_data

if __name__ == "__main__":
    config = Config()
    preprocess_training_data(config)