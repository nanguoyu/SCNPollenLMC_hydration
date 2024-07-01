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
import wandb

# os.environ["WANDB_API_KEY"] = "fcff66c814af270f0fa4d6ef837609b0da2cccc4"

parser = argparse.ArgumentParser(description='SCN Pollen')
parser.add_argument('--datadir', default='data', type=str)
parser.add_argument('--batchsize', default=64, type=int)
parser.add_argument('--save-dir', dest='save_dir', default='save_temp', type=str)
parser.add_argument('--arch', '-a', metavar='ARCH', default='hhnsconvb')
parser.add_argument('--nlayers', default=1, type=int)
parser.add_argument('--width', default=32, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--learning_rate', default=0.0001, type=float)
parser.add_argument('--dimensions', default=8, type=int)
parser.add_argument('--transform', default='brightness', type=str)
parser.add_argument('--output', default='.', type=str)
args = parser.parse_args()

dataset_root_folder = f'./data/images_7_types_7030'
train_directory = f'{dataset_root_folder}_train'
valid_directory = f'{dataset_root_folder}_val'
test_directory = f'{dataset_root_folder}_test'

from datasettings import image_transforms

dataset = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'test': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
}

# Create iterators for data loading
# dataloaders = {
#     'train': data.DataLoader(dataset['train'], batch_size=args.batchsize, shuffle=True),
#     'test': data.DataLoader(dataset['test'], batch_size=args.batchsize, shuffle=False,
#                             num_workers=4, pin_memory=True, drop_last=False),
# }


def main():
    utils.set_seed(15)
    start = timeit.default_timer()

    ######## shape parameters
    nchannels, nclasses = 3, len(dataset['train'].classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from ffcv_dataset import load_ffcv_data
    dataloaders = {
        'train' : load_ffcv_data('train', 'pollen', batch_size=64, device=device,num_workers=4),
        'test' : load_ffcv_data('test', 'pollen', batch_size=64, device=device,num_workers=4)

    }
    ######## download datasets
    train_loader = dataloaders['train']
    test_loader = dataloaders['test']
    # for images, labels in train_loader:
    #     print(images.shape, labels.shape)  # prints [batch_size, 1, 28, 28] and [batch_size]
    #     break  # remove break to go through all batches

    ######## prepare model structure
    model, save_dir = utils.prepare_model(args, nchannels, nclasses, hin=1)
    # wandb.init(project=f"SCNPollen", entity="ahinea", name=f"SCN_{args.transform}_{save_dir}")
    wandb_proiject_name = "scn_pollen" 
    experiment_name = f"train_{args.transform}/SCN/{save_dir}"
    wandb.init(project=wandb_proiject_name, name=experiment_name, entity="naguoyu", 
               config={"arch":args.arch, "transform":args.transform},
               )
    
    model.to(device)
    print(model)
    print(utils.count_model_parameters(model))

    ######## train model
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    def train(dataloader, model, loss_fn, optimizer):
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch, (X, y) in enumerate(tqdm(dataloader, desc='Training')):
            param = random.uniform(0.05, 3.0)
            X, y = X.to(device), y.to(device)
            if args.transform == "brightness":
                X = TF.adjust_brightness(X, brightness_factor=param)
            elif args.transform == "contrast":
                X = TF.adjust_contrast(X, contrast_factor=param)
            elif args.transform == "saturation":
                X = TF.adjust_saturation(X, saturation_factor=param)
            elif args.transform == "sharpness":
                X = TF.adjust_sharpness(X, sharpness_factor=param)
            Hyper_X = Tensor([param]).to(device)

            pred = model(X, Hyper_X)
            loss = loss_fn(pred, y)
            train_loss += loss.item()

            beta1 = model.hyper_stack(Hyper_X)
            param2 = random.uniform(0.05, 3.0)
            beta2 = model.hyper_stack(Tensor([param2]).to(device))
            loss += pow(cos(beta1, beta2), 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predicted = pred.max(1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()

        train_accuracy = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss:>10f}, Training Accuracy: {100 * train_accuracy:.2f}%")
        scheduler.step()
        return train_accuracy, avg_train_loss

    def validate(dataloader, model, loss_fn):
        param = random.uniform(0.05, 3.0)
        model.eval()
        test_loss, correct = 0, 0
        test_total = 0 
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                if args.transform == "brightness":
                    X = TF.adjust_brightness(X, brightness_factor=param)
                elif args.transform == "contrast":
                    X = TF.adjust_contrast(X, contrast_factor=param)
                elif args.transform == "saturation":
                    X = TF.adjust_saturation(X, saturation_factor=param)
                elif args.transform == "sharpness":
                    X = TF.adjust_sharpness(X, sharpness_factor=param)
                Hyper_X = Tensor([param]).to(device)

                pred = model(X, Hyper_X)
                test_total += pred.size(0)

                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= len(dataloader)
        correct /= test_total
        print(f"Test with param={param}: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")
        return correct, test_loss

    for t in range(args.epochs):
        print(f"=================\n Epoch: {t + 1} \n=================")
        train_acc, train_loss = train(train_loader, model, loss_fn, optimizer)
        test_acc, test_loss = validate(test_loader, model, loss_fn)
        # wandb.log({"test/loss": test_loss, "test/acc": test_acc})
        wandb.log({"train/acc": train_acc, "train/loss": train_loss, "test/acc": test_acc, "test/loss": test_loss})

    print("Done!")

    # wandb.finish()

    ######## test model
    def test(dataloader, model, loss_fn, param):
        model.eval()
        test_loss, correct = 0, 0
        test_total = 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                if args.transform == "brightness":
                    X = TF.adjust_brightness(X, brightness_factor=param)
                elif args.transform == "contrast":
                    X = TF.adjust_contrast(X, contrast_factor=param)
                elif args.transform == "saturation":
                    X = TF.adjust_saturation(X, saturation_factor=param)
                elif args.transform == "sharpness":
                    X = TF.adjust_sharpness(X, sharpness_factor=param)
                Hyper_X = Tensor([param]).to(device)

                pred = model(X, Hyper_X)
                test_total += pred.size(0)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= len(dataloader)
        correct /= test_total
        print(f"Test with param={param}: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")
        return correct

    acc = []
    for param in tqdm(np.arange(0.05, 3.0, 0.05), desc='Testing'):
        acc.append(test(test_loader, model, loss_fn, param))

    ######## test model fixed degree
    def test_fixed(dataloader, model, loss_fn, param):
        test_loss, correct = 0, 0
        test_total = 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                if args.transform == "brightness":
                    X = TF.adjust_brightness(X, brightness_factor=param)
                elif args.transform == "contrast":
                    X = TF.adjust_contrast(X, contrast_factor=param)
                elif args.transform == "saturation":
                    X = TF.adjust_saturation(X, saturation_factor=param)
                elif args.transform == "sharpness":
                    X = TF.adjust_sharpness(X, sharpness_factor=param)
                Hyper_X = Tensor([1.0]).to(device) # fixed

                pred = model(X, Hyper_X)
                test_total += pred.size(0)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /=  len(dataloader)
        correct /= test_total
        print(f"Test with param={param}: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
        return correct

    acc_fixed = []
    model.eval()
    for param in tqdm(np.arange(0.05, 3.0, 0.05), desc='Testing'):
        acc_fixed.append(test_fixed(test_loader, model, loss_fn, param))

    # ######## compute beta space
    beta_space = []
    for param in np.arange(0.05, 3.0, 0.05):
        Hyper_X = Tensor([param]).to(device)
        beta_space.append(model.hyper_stack(Hyper_X).cpu().detach().numpy())

    beta_space = np.stack(beta_space)
    print(beta_space.shape)

    hhn_dict = {'acc': acc, 'acc_fixed': acc_fixed, 'beta_space': np.array(beta_space)}

    ######## write to the bucket
    destination_name = f'{args.output}/{args.transform}/SCN/{save_dir}'
    os.makedirs(destination_name, exist_ok=True)
    np.save(f'{destination_name}/acc.npy', pickle.dumps(hhn_dict))

    torch.save(model.state_dict(), f'{destination_name}/model.pt')

    stop = timeit.default_timer()
    print('Time: ', stop - start)


if __name__ == '__main__':
    main()
