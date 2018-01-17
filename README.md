# Burst Denoising with Kernel Prediction Networks
Ben Mildenhall | Jonathan T. Barron | Jiawen Chen | Dillon Sharlet | Ren Ng | Robert Carroll

This is not an official Google product. This repository contains code for training models from the paper [Burst Denoising with Kernel Prediction Networks](https://arxiv.org/abs/1712.02327).

## Dependencies

This code uses the following external packages:

* TensorFlow
* NumPy
* SciPy
* Matplotlib

## Dataset

Synthetic training data is generated using the OpenImages dataset, which can be manually downloaded following [these instructions](https://github.com/cvdfoundation/open-images-dataset).  

## Training

Run the following command to train the kernel prediction network (KPN) burst denoising model:

```
python kpn_train.py --dataset_dir $OPEN_IMAGES_DATASET_DIR --data_dir $REAL_BURST_DATA_DIR
