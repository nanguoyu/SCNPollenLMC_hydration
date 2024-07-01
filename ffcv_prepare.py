"""
@File : prepare_ffcv_dataset.py
@Author: Dong Wang
@Date : 2024/06/25
@Description : prepare FFCV dataset files. You need first install FFCV in your environment: https://github.com/libffcv/ffcv
"""
import os
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import transforms, datasets
import numpy as np
import torch
import torchvision

dataset_infor = {
    'FashionMNIST':{'num_classes':10, 'num_channels':1},
    'MNIST':{'num_classes':10, 'num_channels':1},
    'ImageNet':{'num_classes':1000, 'num_channels':3},
    'CIFAR10':{'num_classes':10, 'num_channels':3},
    'pollen':{'num_classes':7, 'num_channels':3},
    'CIFAR100':{'num_classes':100, 'num_channels':3},


}

def prepare_data(split, dataset_name, datadir):

    import ffcv
    from ffcv.fields import IntField, RGBImageField
    from ffcv.writer import DatasetWriter


    import os
    output_dir = './ffcv_datasets'
    os.makedirs(output_dir, exist_ok=True)
    
    if split=="train":
        sub="train"
    else:
        sub="test"

    output_file = f'{output_dir}/{dataset_name}_{sub}_ffcv.beton'

    image_field = RGBImageField(write_mode='smart', max_resolution=96, jpeg_quality=90)
    dataset_root_folder = f'./data/images_7_types_7030'
    train_directory = f'{dataset_root_folder}_train'
    valid_directory = f'{dataset_root_folder}_val'
    test_directory = f'{dataset_root_folder}_test'
    if split == 'train':
        dataset = datasets.ImageFolder(root=train_directory)
    else:
        dataset = datasets.ImageFolder(root=train_directory)



    write_config = {
        'image': image_field,
        'label': IntField()
    }

    writer = DatasetWriter(output_file, write_config)
    writer.from_indexed_dataset(dataset)

# Now you can generate FFCV dataset before use it for training.

# CIFAT10
# prepare_data(split="train", dataset_name="CIFAR10", datadir="datasets")
# prepare_data(split="test", dataset_name="CIFAR10", datadir="datasets")

# CIFAR100
# prepare_data(split="train", dataset_name="CIFAR100", datadir="datasets")
# prepare_data(split="test", dataset_name="CIFAR100", datadir="datasets")

# For ImageNet, `~/data/ImageNet` should be a folder containing files ILSVRC2012_devkit_t12.tar.gz, ILSVRC2012_img_train.tar, ILSVRC2012_img_val.tar 
# prepare_data(split="train", dataset_name="ImageNet", datadir="~/data/ImageNet")
# prepare_data(split="test", dataset_name="ImageNet", datadir="~/data/ImageNet")

# Pollen dataset
prepare_data(split="train", dataset_name="pollen", datadir="datasets")
prepare_data(split="test", dataset_name="pollen", datadir="datasets")