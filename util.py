import os

def read_dir(path, folder_only=True):
  if folder_only:
    return [f.path for f in os.scandir(path) if f.is_dir()]
  else:
    return [f.path for f in os.scandir(path)]