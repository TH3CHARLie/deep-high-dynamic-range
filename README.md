# deep-high-dynamic-range (WIP)
Tensorflow implementation of SIGGRAPH 17 paper: Deep High Dynamic Range Imaging of Dynamic Scenes


## Quantitative Results

| name | optical-flow | model | batch_size | epoch | PSNR-L |
| ---- | ------------ | ----- | ---------- | ----- | ------ |
| deepflow-direct | deepflow | direct | 20 | 2 | 39.96 |
| deepflow-wie | deepflow | WIE | 20 | 2 | 40.09 |


## Reference
1. Kalantari, N.K., Ramamoorthi, R.: Deep High Dynamic Range Imaging of Dynamic Scenes. ACM TOG 36(4) (2017)