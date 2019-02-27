# Lending Orientation to Neural Networks for Cross-view Geo-localization

This contains the ACT dataset and codes for training cross-view geo-localization method described in: Lending Orientation to Neural Networks for Cross-view Geo-localization, CVPR2019. 

![alt text](./ourCNN_twoStream.PNG)

# Abstract
This paper studies image-based geo-localization (IBL) problem using ground-to-aerial cross-view matching. The goal is to predict the spatial location of a ground-level query image by matching it to a large geotagged aerial image database (e.g., satellite imagery). This is a challenging task due to the drastic differences in their viewpoints and visual appearances. Existing deep learning methods for this problem have been focused on maximizing feature similarity between spatially closeby image pairs, while minimizing other images pairs which are far apart. They do so by deep feature embedding based on visual appearance in those ground-and-aerial images. However, in everyday life, humans commonly use orientation information as an important cue for the task of spatial localization. Inspired by this insight, this paper proposes a novel method which endows deep neural networks with the commonsense of orientation. Given a ground-level spherical panoramic image as query input (and a large geo-referenced satellite image database), we design a Siamese network which explicitly encodes the orientation (i.e., spherical directions) of each pixel of the images. Our method significantly boosts the discriminative power of the learned deep features, leading to a much higher recall and precision outperforming all previous methods. Our network is also more compact using only 1/5th number of parameters than a previously best-performing network. To evaluate the generalization of our method, we also created a large-scale cross-view localization benchmark containing 100K geotagged ground-aerial pairs covering a geographic area of 300 square miles.

# Codes and Models
Our model is implemented in Tensorflow 1.4.0. Other tensorflow versions should be OK.
All our models are trained from scratch, so please run the training codes to obtain models. Specifically:

train_CVUSA.py is used to train model for CVUSA dataset.
train_CVACT.py is used to train model for CVACT dataset.

The model will be saved after each epoch in directory /src/Models/. The accuracy on test set is computed after each epoch and is saved in directory /src/Result/.

# ACT dataset
Our ACT dataset is targgeted for fine-grain and city-scale cross-view localization. The ground-view images are panoramas, and satellite images are downloaded from Google map. 
ACT dataset densely cover the Canberra city, and a sample cross-view pair is depicted as below.

![alt text](./pano_img.png)

![alt text](./sat_img.png)

Our ACT dataset has three subsets:

1. [ACT-small](https://pages.github.com/). Small-scale dataset for training and validation.



