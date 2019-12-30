# BTS
From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation   
[arXiv](https://arxiv.org/abs/1907.10326)  
[Supplementary material](https://arxiv.org/src/1907.10326v4/anc/bts_sm.pdf) 

## Note
This folder contains a PyTorch implementation of BTS.\
We tested this code under python 3.6, PyTorch 1.2.0, CUDA 10.0 on Ubuntu 18.04.

## Testing with [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
First make sure that you have prepared the test set using instructions in README.md at root of this repo.
```shell
$ cd ~/workspace/bts/pytorch
$ mkdir models
### Get BTS model trained with NYU Depth V2
$ python ../utils/download_from_gdrive.py 1w8d7Jq6fRSr8g8j-xy55gDEHjxMNP-1x models/bts_nyu_v2_pytorch.zip
$ cd models
$ unzip bts_nyu_v2_pytorch.zip
```
Once the preparation steps completed, you can test BTS using following commands.
```
$ cd ~/workspace/bts/pytorch
$ python bts_test.py arguments_test_nyu.txt
```
This will save results to ./result_bts_nyu_v2_pytorch. With a single RTX 2080 Ti it takes about 41 seconds for processing 654 testing images. 

## Evaluation
Following command will evaluate the prediction results for NYU Depvh V2.
```
$ cd ~/workspace/bts
$ python utils/eval_with_pngs.py --pred_path pytorch/result_bts_nyu_v2_pytorch/raw/ --gt_path ../dataset/nyu_depth_v2/official_splits/test/ --dataset nyu --min_depth_eval 1e-3 --max_depth_eval 10 --eigen_crop
```

You should see outputs like this:
```
Raw png files reading done
Evaluating 654 files
GT files reading done
0 GT files missing
Computing errors
     d1,      d2,      d3,  AbsRel,   SqRel,    RMSE, RMSElog,   SILog,   log10
  0.883,   0.980,   0.996,   0.110,   0.064,   0.394,   0.142,  11.559,   0.047
Done.
```

## Preparation for Training
### NYU Depvh V2
Download the dataset we used in this work.
```
$ cd ~/workspace/bts
$ python utils/download_from_gdrive.py 1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP ../dataset/nyu_depth_v2/sync.zip
$ unzip sync.zip
```

Once the dataset is ready, you can train the network using following command.
```
$ cd ~/workspace/bts/pytorch
$ python bts_main.py arguments_train_nyu.txt
```
You can check the training using tensorboard:
```
$ tensorboard --logdir ./models/bts_nyu_test/ --port 6006
```
Open localhost:6006 with your favorite browser to see the progress of training.

### KITTI
You can also train BTS with KITTI dataset by following procedures.
First, make sure that you have prepared the ground truth depthmaps from [KITTI](http://www.cvlibs.net/download.php?file=data_depth_annotated.zip).
If you have not, please follow instructions on README.md at root of this repo.
Then, download and unzip the raw dataset using following commands.
```
$ cd ~/workspace/dataset/kitti_dataset
$ aria2c -x 16 -i ../../bts/utils/kitti_archives_to_download.txt
$ parallel unzip ::: *.zip
```
Finally, we can train our network with
```
$ cd ~/workspace/bts/pytorch
$ python bts_main.py arguments_train_eigen.txt
```

## Testing and Evaluation with [KITTI](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)
Once you have KITTI dataset and official ground truth depthmaps, you can test and evaluate our model with following commands.
```
# Get KITTI model trained with KITTI Eigen split
$ cd ~/workspace/bts
$ python utils/download_from_gdrive.py 1iod9ohaJ9C2pmzsXqVrfwLLMlDSCvwBW pytorch/models/bts_eigen_v2_pytorch.zip
$ cd pytorch/models && unzip bts_eigen_v2_pytorch.zip
```
Test and save results.
```
$ cd ~/workspace/bts/pytorch
$ python bts_test.py arguments_test_eigen.txt
```
This will save results to ./result_bts_eigen_v2_pytorch.
Finally, we can evaluate the prediction results with
```
$ cd ~/workspace/bts
$ python utils/eval_with_pngs.py --pred_path pytorch/result_bts_eigen_v2_pytorch/raw/ --gt_path ../dataset/kitti_dataset/data_depth_annotated/ --dataset kitti --min_depth_eval 1e-3 --max_depth_eval 80 --do_kb_crop --garg_crop
```
You should see outputs like this:
```
GT files reading done
45 GT files missing
Computing errors
     d1,      d2,      d3,  AbsRel,   SqRel,    RMSE, RMSElog,   SILog,   log10
  0.952,   0.992,   0.998,   0.063,   0.264,   2.892,   0.100,   9.186,   0.028
Done.
```

Also, in this pytorch implementation, you can use various base networks with pretrained weights as the encoder for bts.\
Available options are: resnet50_bts, resnet101_bts, densenet121_bts and densenet161_bts\
Simply change the argument '--encoder' in arguments_train_*.txt with your choice.

## License
Copyright (C) 2019 Jin Han Lee, Myung-Kyu Han, Dong Wook Ko and Il Hong Suh \
This Software is licensed under GPL-3.0-or-later.