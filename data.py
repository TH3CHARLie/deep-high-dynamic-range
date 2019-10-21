import cv2
import numpy as np
import os
import util
def preprocess_training_data(config):
  """
  preprocess training data, pack it into h5 file
  """
  scene_paths = util.read_dir(config.TRAINING_RAW_DATA_PATH)
  print(scene_paths)
  for scene_path in scene_paths:
    exposures = read_exposure(scene_path)
    print(f"exposures: {exposures}")
    ldr_imgs, hdr_img = read_images(scene_path)
    # inputs, label = compute_training_examples(ldr_imgs, exposures, hdr_img)
    # write_training_examples(inputs, label, config.TRAINING_DATA_PATH, "TrainingSequence.h5")


def read_exposure(path):
  """
  read exposure data from exposures.txt

  exposures are specified in exponent representation
  thus, return 2 ** x
  """
  paths = [x.path for x in os.scandir(path) if x.name.endswith('.txt')]
  if len(paths) < 1:
    print("[read_exposure]: cannot find exposure file")
    return None
  exposure_file_path = paths[0]
  exposures = []
  with open(exposure_file_path) as f:
    for line in f:
      exposures.append(2 ** float(line))
  return exposures



def read_images(path):
  pass

def compute_training_examples(ldr_imgs, exposures, hdr_img):
  pass

def writing_training_examples(inputs, label, path, filename):
  pass