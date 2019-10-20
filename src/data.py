import cv2
import numpy as np
import os
def preprocess_training_data(config):
  """
  preprocess training data, pack it into h5 file
  """
  scene_paths = read_dir(config.TRAINING_RAW_DATA_PATH)
  for scene_path in scene_paths:
    exposures = read_exposure(scene_path)
    ldr_imgs, hdr_img = read_images(scene_path)
    inputs, label = compute_training_examples(ldr_imgs, exposures, hdr_img)
    write_training_examples(inputs, label, config.TRAINING_DATA_PATH, "TrainingSequence.h5")


def read_exposure(path):
  pass

def read_images(path):
  pass

def compute_training_examples(ldr_imgs, exposures, hdr_img):
  pass

def writing_training_examples(inputs, label, path, filename):
  pass