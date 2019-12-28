# BTS
From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation   
[arXiv](https://arxiv.org/abs/1907.10326)  
[Supplementary material](https://arxiv.org/src/1907.10326v4/anc/bts_sm.pdf) 

## Video Demo 1
[![Screenshot](https://img.youtube.com/vi/2fPdZYzx9Cg/maxresdefault.jpg)](https://www.youtube.com/watch?v=2fPdZYzx9Cg)
## Video Demo 2
[![Screenshot](https://img.youtube.com/vi/1J-GSb0fROw/maxresdefault.jpg)](https://www.youtube.com/watch?v=1J-GSb0fROw)

## Note
This repository contains TensorFlow and PyTorch implementations of BTS.
## Preparation for all implementations
```shell
$ cd ~
$ mkdir workspace
$ cd workspace
### Make a folder for datasets
$ mkdir dataset
### Clone this repo
$ git clone https://github.com/cogaplex-bts/bts
```
## Prepare [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) test set
```shell
$ cd ~/workspace/bts/utils
### Get official NYU Depth V2 split file
$ wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
### Convert mat file to image files
$ python extract_official_train_test_set_from_mat.py nyu_depth_v2_labeled.mat splits.mat ../../dataset/nyu_depth_v2/official_splits/
```
## Prepare [KITTI](http://www.cvlibs.net/download.php?file=data_depth_annotated.zip) official ground truth depth maps
Download the ground truth depthmaps from this link [KITTI](http://www.cvlibs.net/download.php?file=data_depth_annotated.zip).\
Then,
```
$ cd ~/workspace/dataset
$ mkdir kitti_dataset && cd kitti_dataset
$ mv ~/Downloads/data_depth_annotated.zip .
$ unzip data_depth_annotated.zip
```

Follow instructions from one of the below implementations with your choice.

## TensorFlow Implementation
[[./tensorflow/]](./tensorflow/)
## PyTorch Implementation
[[./pytorch/]](./pytorch/)

## Live Demo
Finally, we attach live 3d demo implementations for both of TensorFlow and Pytorch. \
For best performance, get correct intrinsic values for your webcam and put them in bts_live_3d.py. \
Sample Usage for PyTorch:
```
$ cd ~/workspace/bts/pytorch
$ python bts_live_3d.py --model_name bts_nyu_v2_pytorch \
--encoder densenet161_bts \
--checkpoint_path ./models/bts_nyu_v2_pytorch/model \
--max_depth 10 \
--input_height 480 \
--input_width 640
```

## Citation
If you find this work useful for your research, please consider citing our paper:
```
@article{lee2019big,
  title={From big to small: Multi-scale local planar guidance for monocular depth estimation},
  author={Lee, Jin Han and Han, Myung-Kyu and Ko, Dong Wook and Suh, Il Hong},
  journal={arXiv preprint arXiv:1907.10326},
  year={2019}
}
```

## License
Copyright (C) 2019 Jin Han Lee, Myung-Kyu Han, Dong Wook Ko and Il Hong Suh \
This Software is licensed under GPL-3.0-or-later.