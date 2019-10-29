from config import Config
import util
from data import read_exposure, read_ldr_hdr_images, compute_training_examples, write_training_examples

if __name__ == "__main__":
    config = Config()
    preprocess_training_data(config)
    preprocess_test_data(config)  # TODO:


def preprocess_training_data(config: Config):
    """
    preprocess training data
    """
    scene_paths = util.read_dir(config.TRAINING_RAW_DATA_PATH)
    # make sure we read scene sequentially
    scene_paths = sorted(scene_paths)
    cnt = 0
    for scene_path in scene_paths:
        exposures = read_exposure(scene_path)
        ldr_imgs, hdr_img = read_ldr_hdr_images(scene_path)
        inputs, label = compute_training_examples(
            ldr_imgs, exposures, hdr_img, config)
        cnt += inputs.shape[0]
        print(f"processed scene: {scene_path}")
        print(f"now image patches cnt: {cnt}")

        write_training_examples(
            inputs, label, config.TRAINING_DATA_PATH, scene_path)
    print(f"total {cnt} patches")


def preprocess_test_data(config: Config):
    pass
