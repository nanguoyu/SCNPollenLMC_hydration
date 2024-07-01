import ffcv
from ffcv.transforms import RandomHorizontalFlip, NormalizeImage, Squeeze,  RandomHorizontalFlip, ToTorchImage, ToDevice, Convert, ToTensor, Convert
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder,SimpleRGBImageDecoder
from ffcv.fields.basics import IntDecoder
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import transforms, datasets
import numpy as np
import torch
import torchvision

def ffcv_data(dataset_name, split, num_workers, batch_size, image_pipeline, label_pipeline):
    from ffcv.loader import Loader, OrderOption
    output_dir = './ffcv_datasets'
    if split=="train":
        sub="train"
    else:
        sub="test"
    #todo: check file exists.
    output_file = f'{output_dir}/{dataset_name}_{sub}_ffcv.beton'

    data_loader = Loader(output_file, batch_size=batch_size, num_workers=num_workers, order=OrderOption.RANDOM if split=="train" else OrderOption.SEQUENTIAL, distributed=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        })
    return data_loader

def load_ffcv_data(split, dataset_name, batch_size,device,num_workers=4):
    mean =  [0.5732364, 0.5861441, 0.4746769]
    std = [0.1434643,  0.16687445, 0.15344492]
    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(device, non_blocking=True),

    ]
    if split =='train':
        image_pipeline= [
                        RandomResizedCropRGBImageDecoder((96, 96)),
                        ToTensor(),
                        ToDevice(device, non_blocking=True),
                        ToTorchImage(),
                        Convert(torch.float32),
                        transforms.RandomCrop(96, padding=4),
                        # transforms.RandomHorizontalFlip(), 
                        # transforms.RandomRotation(15),
                        torchvision.transforms.Normalize(np.array(mean)*255, np.array(std)*255),
                        ]
    elif split == 'test':
        image_pipeline =[
                        RandomResizedCropRGBImageDecoder((96, 96)),
                        ToTensor(),
                        ToDevice(device, non_blocking=True),
                        ToTorchImage(),
                        Convert(torch.float32),
                        transforms.RandomCrop(96, padding=4),
                        # transforms.RandomHorizontalFlip(), 
                        # transforms.RandomRotation(15),
                        torchvision.transforms.Normalize(np.array(mean)*255, np.array(std)*255),
        ]
    data_loader = ffcv_data(dataset_name, split, num_workers, batch_size, image_pipeline, label_pipeline)
    print("Using FFCV dataset.")
    return data_loader