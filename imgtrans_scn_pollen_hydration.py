import argparse
import os
import pickle
import random
import timeit

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
from torch import nn, Tensor
from torchvision import datasets, transforms
from tqdm import tqdm

import utils
# import wandb

from torch.utils.data import Dataset, DataLoader
from PIL import Image

os.environ["WANDB_API_KEY"] = "fcff66c814af270f0fa4d6ef837609b0da2cccc4"

parser = argparse.ArgumentParser(description='SCN Pollen')
parser.add_argument('--datadir', default='data', type=str)
parser.add_argument('--dataset', default='CIFAR10', type=str, help='MNIST | FashionMNIST | CIFAR10 | CIFAR100 | SVHN')
parser.add_argument('--batchsize', default=128, type=int) # 
parser.add_argument('--save-dir', dest='save_dir', default='save_temp', type=str)
parser.add_argument('--arch', '-a', metavar='ARCH', default='hhnmlpb_hydration') # hhnmlpb_hydration hhnsconv_hydration
parser.add_argument('--nlayers', default=1, type=int)
parser.add_argument('--width', default=64, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--dimensions', default=8, type=int)
parser.add_argument('--transform', default='hydration', type=str) # brightness contrast saturation sharpness
parser.add_argument('--output', default='output', type=str)
args = parser.parse_args()

dataset_root_folder = './data/images_3_types_hydrated_7030'

train_directory = f'{dataset_root_folder}_train'
train_dry_directory = './data/images_3_types_dry_7030_train'
train_half_hydrated_directory =  './data/images_3_types_half_hydrated_7030_train'


valid_directory = f'{dataset_root_folder}_val'
test_directory = f'{dataset_root_folder}_test'

valid_dry_directory = './data/images_3_types_dry_7030_val'
valid_half_hydrated_directory = './data/images_3_types_half_hydrated_7030_val'

    

# Applying transforms to the data
from datasettings import image_transforms



dataset = {
    'train_hydrated': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'train_dry': datasets.ImageFolder(root=train_dry_directory, transform=image_transforms['train']),
    'train_half_hydrated': datasets.ImageFolder(root=train_half_hydrated_directory, transform=image_transforms['train']),
    
    'valid_half_hydrated': datasets.ImageFolder(root=valid_half_hydrated_directory, transform=image_transforms['valid']),    
    'valid_dry': datasets.ImageFolder(root=valid_dry_directory, transform=image_transforms['valid']),    
    
    'test': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
}

# Create iterators for data loading
dataloaders = {
    'train_hydrated': data.DataLoader(dataset['train_hydrated'], batch_size=args.batchsize, shuffle=True),
    'train_dry': data.DataLoader(dataset['train_dry'], batch_size=args.batchsize, shuffle=True),
    'train_half_hydrated': data.DataLoader(dataset['train_half_hydrated'], batch_size=args.batchsize, shuffle=True),
    
    'valid_half_hydrated': data.DataLoader(dataset['valid_half_hydrated'], batch_size=args.batchsize, shuffle=False),
    'valid_dry': data.DataLoader(dataset['valid_dry'], batch_size=args.batchsize, shuffle=False),
    
    'test': data.DataLoader(dataset['test'], batch_size=args.batchsize, shuffle=False,
                            num_workers=4, pin_memory=True, drop_last=False),
}


dimensions = [1, 2, 3, 5, 8, 16]
width_list = [256]
# Dataset name
args.dataset = os.path.split(dataset_root_folder)[1]

def main():
    for ww in width_list:
        for dimension in dimensions:
            args.dimensions = dimension
            args.width = ww
            
            start = timeit.default_timer()
        
            ######## shape parameters
            nchannels, nclasses = 3, len(dataset['train_hydrated'].classes)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
            ######## prepare model structure
            model, save_dir = utils.prepare_model(args, nchannels, nclasses, hin=1)
            # wandb.init(project="SCNPollen", entity="caonam", name=f"SCN_{args.transform}_{save_dir}")
            model.to(device)
            print(model)
            print(utils.count_model_parameters(model))
        
            ######## train model
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-5)
        
            cos = nn.CosineSimilarity(dim=0, eps=1e-6)
            
            def reset_iterator(data_loader):
                return iter(data_loader)
            
        
            def train(model, loss_fn, optimizer):
                # Create iterators for the data loaders
                iterator1 = iter(dataloaders['train_hydrated'])
                iterator2 = iter(dataloaders['train_half_hydrated'])
                iterator3 = iter(dataloaders['train_dry'])
                
                for _ in range(12):  # loop for 10 batches
                    # Randomly choose a dataset
                    t = random.choice([0, 0.5, 1.0])
                    
                    if t == 0:
                        try:
                            inputs, labels = next(iterator1)
                        except StopIteration:
                            iterator1 = reset_iterator(dataloaders['train_hydrated'])
                            inputs, labels = next(iterator1)
                    elif t == 0.5:
                        try:
                            inputs, labels = next(iterator2)
                        except StopIteration:
                            iterator2 = reset_iterator(dataloaders['train_half_hydrated'])
                            inputs, labels = next(iterator2)
                    elif t == 1:
                        try:
                            inputs, labels = next(iterator3)
                        except StopIteration:
                            iterator3 = reset_iterator(dataloaders['train_dry'])
                            inputs, labels = next(iterator3)
                    
                    X, y = inputs.to(device), labels.to(device)
        
                    Hyper_X = Tensor([t]).to(device)
        
                    pred = model(X, Hyper_X)
                    loss = loss_fn(pred, y)
        
                    beta1 = model.hyper_stack(Hyper_X)
                    param2 = random.choice([0, 0.5, 1.0])
                    beta2 = model.hyper_stack(Tensor([param2]).to(device))
                    loss += pow(cos(beta1, beta2), 2)
        
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                scheduler.step()
        
            def validate(dataloader, model, loss_fn, param):

                model.eval()
                test_loss, correct = 0, 0
                with torch.no_grad():
                    for (X, y) in dataloader:
                        X, y = X.to(device), y.to(device)
                 
                        Hyper_X = Tensor([param]).to(device)
        
                        pred = model(X, Hyper_X)
                        test_loss += loss_fn(pred, y).item()
                        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                test_loss /= len(dataloader)
                correct /= len(dataloader.dataset)
                print(f"Test with param={param}: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")
                return correct, test_loss
        
        
            for t in range(args.epochs):
                print(f"=================\n Epoch: {t + 1} \n=================")
                train(model, loss_fn, optimizer)

                test_acc, test_loss = validate(dataloaders['test'], model, loss_fn, 0)
                test_acc, test_loss = validate(dataloaders['valid_half_hydrated'], model, loss_fn, 0.5)
                test_acc, test_loss = validate(dataloaders['valid_dry'], model, loss_fn, 1.0)
                
                # wandb.log({"test/loss": test_loss, "test/acc": test_acc})
            print("Done!")
        
            # wandb.finish()
        
            ######## test model
            def test(dataloader, model, loss_fn, param):
                model.eval()
                test_loss, correct = 0, 0
                with torch.no_grad():
                    for (X, y) in dataloader:
                        X, y = X.to(device), y.to(device)
                       
                        Hyper_X = Tensor([param]).to(device)
        
                        pred = model(X, Hyper_X)
                        test_loss += loss_fn(pred, y).item()
                        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                test_loss /= len(dataloader)
                correct /= len(dataloader.dataset)
                print(f"Test with param={param}: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")
                return correct
        
            acc = []

            acc.append(test(dataloaders['test'], model, loss_fn, 0))
            acc.append(test(dataloaders['valid_half_hydrated'], model, loss_fn, 0.5))
            acc.append(test(dataloaders['valid_dry'], model, loss_fn, 1.0))
         
        
            # ######## test model fixed degree
            # def test_fixed(dataloader, model, loss_fn, param):
            #     model.eval()
            #     test_loss, correct = 0, 0
            #     with torch.no_grad():
            #         for X, y in dataloader:
            #             X, y = X.to(device), y.to(device)
            #             if args.transform == "brightness":
            #                 X = TF.adjust_brightness(X, brightness_factor=param)
            #             elif args.transform == "contrast":
            #                 X = TF.adjust_contrast(X, contrast_factor=param)
            #             elif args.transform == "saturation":
            #                 X = TF.adjust_saturation(X, saturation_factor=param)
            #             elif args.transform == "sharpness":
            #                 X = TF.adjust_sharpness(X, sharpness_factor=param)
            #             Hyper_X = Tensor([1.0]).to(device) # fixed
            #
            #             pred = model(X, Hyper_X)
            #             test_loss += loss_fn(pred, y).item()
            #             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            #     test_loss /=  len(dataloader)
            #     correct /= len(dataloader.dataset)
            #     print(f"Test with param={param}: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
            #     return correct
            #
            # acc_fixed = []
            # for param in tqdm(np.arange(0.2, 2, 0.05), desc='Testing'):
            #     acc_fixed.append(test_fixed(test_loader, model, loss_fn, param))
        
            # ######## compute beta space
            beta_space = []
            for param in [0,0.5,1.0]:
                Hyper_X = Tensor([param]).to(device)
                beta_space.append(model.hyper_stack(Hyper_X).cpu().detach().numpy())
            
            beta_space = np.stack(beta_space)
            print(beta_space.shape)
            #
            # hhn_dict = {'acc': acc, 'acc_fixed': acc_fixed, 'beta_space': np.array(beta_space)}
            hhn_dict = {'acc': acc, 'beta_space': np.array(beta_space)}
        
            ######## write to the bucket

            destination_name = f'{args.output}/{args.transform}/SCN/{save_dir}'
            os.makedirs(destination_name, exist_ok=True)
            np.save(f'{destination_name}/acc.npy', pickle.dumps(hhn_dict))
        
            torch.save(model.state_dict(), f'{destination_name}/model.pt')
        
            stop = timeit.default_timer()
            print('Time: ', stop - start)


if __name__ == '__main__':
    main()
