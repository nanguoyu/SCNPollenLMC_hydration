import os
import math
import timeit
import argparse
import numpy as np
import utils
import pickle
import random
from tqdm import tqdm
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import wandb
from torchvision import datasets, models, transforms
import torch.utils.data as data

# os.environ["WANDB_API_KEY"] = "0585383768fd4b78bb36a7ed79cf1e7f1c29957f"

parser = argparse.ArgumentParser(description='SCN Pollen')
parser.add_argument('--datadir', default='data', type=str)
parser.add_argument('--batchsize', default=64, type=int)
parser.add_argument('--save-dir', dest='save_dir', default='save_temp', type=str)
parser.add_argument('--arch', '-a', metavar='ARCH', default='sconvb')
parser.add_argument('--nlayers', default=1, type=int)
parser.add_argument('--width', default=32, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--learning_rate', default=0.0001, type=float)
parser.add_argument('--dimensions', default=1, type=int)
parser.add_argument('--transform', default='brightness', type=str)
parser.add_argument('--output', default='output', type=str)
args = parser.parse_args()

dataset_root_folder = f'./data/images_7_types_7030'
train_directory = f'{dataset_root_folder}_train'
valid_directory = f'{dataset_root_folder}_val'
test_directory = f'{dataset_root_folder}_test'

# Applying transforms to the data
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
    utils.set_seed(13)
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
    img_shape = (1,3,96,96)
    for images, labels in train_loader:
        print(images.shape, labels.shape)
        c, h, w = images[0].shape
        img_shape = (1,c,h,w)
        print(img_shape)
        break  # remove break to go through all batches

    ######## prepare model structure
    model, save_dir = utils.prepare_model(args, nchannels, nclasses, hin=1)
    # wandb.init(project=f"SCNPollen", entity="ahinea", name=f"One4All_{args.transform}_{save_dir}")
    model.to(device)
    print(model)
    print(utils.count_model_parameters(model))
    
    ######## test model
    loss_fn = nn.CrossEntropyLoss()

    ######## load weight from the bucket
    org_name = f'{args.output}/{args.transform}/One4All/{save_dir}'

    model.load_state_dict(torch.load(f'{org_name}/model.pt', map_location='cpu'))

    ######## test model
    def test(dataloader, model, loss_fn, param):
        model.eval()
        test_loss, correct = 0, 0
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

                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= len(dataloader)
        correct /= len(dataloader.dataset)
        print(f"Test with param={param}: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")
        return correct

    acc = []
    # for param in tqdm(np.arange(0.2, 2, 0.05), desc='Testing'):
    #     acc.append(test(test_loader, model, loss_fn, param))

    # Conversion
    import onnx
    from onnx import helper
    dummy_input = torch.randn(img_shape)
    # dummy_input = torch.randn(1, 3, 96, 96)
    torch.onnx.export(
            model=model.cpu(),
            args=dummy_input,
            f=f'{org_name}/model.onnx',
            verbose=False,
            export_params=True,
            do_constant_folding=False,
            input_names=['input'],
            opset_version=10,
            output_names=['output'])
    print('ONNX model successfully converted')
    onnx_model = onnx.load(f'{org_name}/model.onnx')

    # adding softmax
    last_node_output = onnx_model.graph.node[-1].output[0]
    softmax_input = [last_node_output]

    softmax_output = last_node_output + "_softmax"
    softmax_node = helper.make_node(
        "Softmax",
        inputs=softmax_input,
        outputs=[softmax_output],
        axis=1
    )
    onnx_model.graph.node.append(softmax_node)
    onnx_model.graph.output[0].name = softmax_output
    onnx.save(onnx_model, f'{org_name}/model.onnx')
    onnx.checker.check_model(onnx_model)
    print(f'Pytorch-->ONNX: saved in {org_name}/model.onnx')
    print('ONNX model successfully updated. Added a softmax.')


    stop = timeit.default_timer()
    print('Time: ', stop - start)



if __name__ == '__main__':
    main()
