# Lending Orientation to Neural Networks for Cross-view Geo-localization

This contains the ACT dataset and codes for training cross-view geo-localization method described in: Lending Orientation to Neural Networks for Cross-view Geo-localization, CVPR2019. 

![alt text](./ourCNN_twoStream.PNG)

# Abstract
This paper studies image-based geo-localization (IBL) problem using ground-to-aerial cross-view matching. The goal is to predict the spatial location of a ground-level query image by matching it to a large geotagged aerial image database (e.g., satellite imagery). This is a challenging task due to the drastic differences in their viewpoints and visual appearances. Existing deep learning methods for this problem have been focused on maximizing feature similarity between spatially closeby image pairs, while minimizing other images pairs which are far apart. They do so by deep feature embedding based on visual appearance in those ground-and-aerial images. However, in everyday life, humans commonly use orientation information as an important cue for the task of spatial localization. Inspired by this insight, this paper proposes a novel method which endows deep neural networks with the commonsense of orientation. Given a ground-level spherical panoramic image as query input (and a large geo-referenced satellite image database), we design a Siamese network which explicitly encodes the orientation (i.e., spherical directions) of each pixel of the images. Our method significantly boosts the discriminative power of the learned deep features, leading to a much higher recall and precision outperforming all previous methods. Our network is also more compact using only 1/5th number of parameters than a previously best-performing network. To evaluate the generalization of our method, we also created a large-scale cross-view localization benchmark containing 100K geotagged ground-aerial pairs covering a geographic area of 300 square miles.

# ACT dataset
Our ACT dataset is targgeted for fine-grain and city-scale cross-view localization. The ground-view images are panoramas, and satellite images are downloaded from Google map. 
ACT dataset densely cover the Canberra city, and a sample cross-view pair is depicted as below.

![alt text](./pano_img.png)

![alt text](./sat_img.png)

Our ACT dataset has two subsets:

1. [ACT_small](https://pages.github.com/). Small-scale dataset for training and validation.
Note the number of training and validation cross-view image pairs are extractly the same as the [CVUSA dataset](https://github.com/viibridges/crossnet) 

2. [ACT_test](https://pages.github.com/). Large-scale dataset for testing.
Note the number of testing cross-view image pairs are 10x bigger than [CVUSA dataset](https://github.com/viibridges/crossnet)

# Codes and Models

## Overview
Our model is implemented in Tensorflow 1.4.0. Other tensorflow versions should be OK.
All our models are trained from scratch, so please run the training codes to obtain models.

For pre-trained model on CVUSA dataset, please download [CVUSA_model](https://drive.google.com/file/d/1_kkBw1oIHbmsikTL9VJJTL_WLMD_B_R5/view?usp=sharing)

For pre-trained model on CVACT dataset, please download [CVACT_model](https://drive.google.com/file/d/14Yd0-ICAaABQQlWaGjSkDsH8L3YUuML0/view?usp=sharing)

In the above [CVUSA_model](https://drive.google.com/file/d/1_kkBw1oIHbmsikTL9VJJTL_WLMD_B_R5/view?usp=sharing) and [CVACT_model](https://drive.google.com/file/d/14Yd0-ICAaABQQlWaGjSkDsH8L3YUuML0/view?usp=sharing),
we also include the pre-extracted feature embeddings, in case you want to directly use them.

Some may want to know how the training preformance improves along with epoches, please refer to 
[recalls_epoches_CVUSA](https://drive.google.com/file/d/15KN_N8Dc1FzRthDyW_1eeiAXWrGchzNv/view?usp=sharing) and [recalls_epoches_CVACT](https://drive.google.com/file/d/1l1pfw9PSkswk1phVWyHed_31FoG0ZsQM/view?usp=sharing).  


## Codes for [CVUSA dataset](https://github.com/viibridges/crossnet)

If you want to use [CVUSA dataset](https://github.com/viibridges/crossnet), first download it, and then modify the img_root variable in [input_data_rgb_ori_m1_1_augument.py](./CVPR2019_codes/OriNet_CVUSA/input_data_rgb_ori_m1_1_augument.py) (line 12)

For example:

```diff
img_root = '..../..../CVUSA/'
```

For training, run:
```diff
python train_deep6_scratch_m1_1_concat3conv_rgb_ori_gem_augment.py
```
For testing, run:
```diff
python eval_deep6_scratch_m1_1_concat3conv_rgb_ori_gem_augment.py
```

Recall@1% is automatically calculated after running the evaluation script, and is saved to [PreTrainModel folder](./CVPR2019_codes/OriNet_CVUSA/PreTrainModel).

To calculate the recall@N figures, you need to use the extracted feature embeddings, and run the matlab script [Recall@N.m](./CVPR2019_codes/Matlab/Recall@N.m). You also need to change the path (variable desc_path) to your descriptor file.

Some may want to know how the cross-view orientations are defined, please refer to [ground_view_orientations](./CVPR2019_codes/OriNet_CVUSA/CVUSA_orientations/yaw_pitch_grd_CVUSA.mat) and [satellite_view_orientations](./CVPR2019_codes/OriNet_CVUSA/CVUSA_orientations/yaw_radius_sat_CVUSA.mat)


## Codes for [CVACT dataset]

Most of the steps for ACT dataset are the same as CVUSA dataset. The differences are:

1. [input_data_ACT.py](./CVPR2019_codes/OriNet_CVACT/input_data_ACT.py) is used in [train_deep6_scratch_m1_1_concat3conv_rgb_ori_gem_ACT.py](./CVPR2019_codes/OriNet_CVACT/train_deep6_scratch_m1_1_concat3conv_rgb_ori_gem_ACT.py) to train CNNs.
It uses the [ACT_small]() dataset for fast training.

2. To test Geo-localization performances on [ACT_test]() dataset, you need to use [input_data_ACT_test.py](./CVPR2019_codes/OriNet_CVACT/input_data_ACT_test.py) in the evaluation script [eval_deep6_scratch_m1_1_concat3conv_rgb_ori_gem_ACT.py](./CVPR2019_codes/OriNet_CVACT/eval_deep6_scratch_m1_1_concat3conv_rgb_ori_gem_ACT.py).

That is to say: change the first line to 
```diff
from input_data_ACT_test import InputData
```

3. To test Geo-localization performances on [ACT_test]() dataset, run the matlab script [Recall@Geo.m](./CVPR2019_codes/Matlab/Recall@Geo.m). You also need to change the path (variable desc_path) to your descriptor file.


# Publication

If you like, you can cite the following publication:
*Liu Liu; Hongdong Li. Lending Orientation to Neural Networks for Cross-view Geo-localization. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.*

@InProceedings{Liu_2019_CVPR,
author = {Liu, Liu and Li, Hongdong},
title = {Lending Orientation to Neural Networks for Cross-view Geo-localization},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}













