# deep-high-dynamic-range
Tensorflow implementation of SIGGRAPH 17 paper: Deep High Dynamic Range Imaging of Dynamic Scenes


## Installation
This implementation requires `python3`, `tensorflow 2.0` and `opencv`. Please install dependencies via:
```bash
pip install tensorflow==2.0
pip install opencv-python
pip install opencv-contrib-python
```
Tested on MacOS 10.15 and CentOS 7.0

## Training
First download the dataset
```bash
cd data/
./download.sh
```
Then run `preprocess.py`. This file accepts `train` or `test` as optional argument to generate only train/test set. The raw data will be transformed into [`tfrecords`](https://www.tensorflow.org/tutorials/load_data/tfrecord) format and stored in `tf-data` folder.
```bash
python preprocess.py
```
Finally, run `train.py`, this file accepts a argument specifying model type: `direct`, `we` or `wie`
```bash
python train.py [model_type]
```

## Testing
Use pretrained weights for testing, run `test.py`. This file again accepts a model type string and an additional argument specifying checkpoint path.

```bash
python test.py [model_type] [checkpoint_path]
```

Example:
```bash
python test.py direct saved-checkpoints/deepflow-direct/model.ckpt-100
```


## Reference
1. Kalantari, N.K., Ramamoorthi, R.: Deep High Dynamic Range Imaging of Dynamic Scenes. ACM TOG 36(4) (2017)
