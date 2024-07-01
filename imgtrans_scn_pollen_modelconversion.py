import argparse
import os
import pickle
import random
import timeit
import copy

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
# parser.add_argument('--epochs', default=200, type=int)
# parser.add_argument('--learning_rate', default=0.0001, type=float)
parser.add_argument('--dimensions', default=8, type=int)
parser.add_argument('--transform', default='brightness', type=str)
parser.add_argument('--output', default='.', type=str)
parser.add_argument('--weight_path', default='.', type=str)

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


def extract_all_base_models(scn_model):
  D = scn_model.dimensions
  with torch.no_grad():
      models = []
      for d in range(D):
        bsmodel = copy.deepcopy(scn_model.base_model)
        bsmodel_weight = scn_model.base_model.state_dict()

        for k,v in scn_model.base_model.named_parameters():
            if not v.requires_grad:
                continue
            r_k = k.replace('.', '-')

            if r_k.find('bn')>=0:
                print(f'{r_k} will not be processed')
                continue
            bsmodel_weight[k] = scn_model._modules[f'base_model-{r_k}-list'][d].clone().detach()
        bsmodel.load_state_dict(bsmodel_weight)
        models.append(bsmodel)
      return models


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
    img_shape = (1,3,96,96)
    for images, labels in train_loader:
        print(images.shape, labels.shape)
        c, h, w = images[0].shape
        img_shape = (1,c,h,w)
        print(img_shape)
        break  # remove break to go through all batches

    ######## prepare model structure
    model, save_dir = utils.prepare_model(args, nchannels, nclasses, hin=1)
    model.to(device)
    # wandb.init(project=f"SCNPollen", entity="ahinea", name=f"SCN_{args.transform}_{save_dir}")
    print(model)
    print(utils.count_model_parameters(model))

    loss_fn = nn.CrossEntropyLoss()

    ######## load weight from the bucket
    org_name = f'{args.output}/{args.transform}/SCN/{save_dir}'

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
                Hyper_X = Tensor([param]).to(device)

                pred = model(X, Hyper_X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= len(dataloader)
        correct /= len(dataloader.dataset)
        print(f"Test with param={param}: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")
        return correct

    acc = []
    # for param in tqdm(np.arange(0.2, 2, 0.2), desc='Testing'):
        # acc.append(test(test_loader, model, loss_fn, param))
    # print(f"average test acc : {np.mean(acc)}")

    # Conversion
    import onnx
    from onnx import helper

    dummy_input = torch.randn(1, 1)
    torch.onnx.export(
            model=model.hyper_stack.cpu(),
            args=dummy_input,
            f=f'{org_name}/model_hypernet.onnx',
            verbose=False,
            export_params=True,
            do_constant_folding=False,
            input_names=['input'],
            opset_version=10,
            output_names=['output'])
    print('ONNX model_hypernet successfully converted')

    onnx_model = onnx.load(f'{org_name}/model_hypernet.onnx')
    onnx.checker.check_model(onnx_model)

    # add softmax to the model
    nodes = onnx_model.graph.node
    node_t1 = next((node for node in nodes if node.name == "/3/Transpose"), None)
    node_softmax = next((node for node in nodes if node.name == "/3/Softmax"), None)
    node_t2 = next((node for node in nodes if node.name == "/3/Transpose_1"), None)
    node_softmax.input[0] = node_t1.input[0]

    for idx, output in enumerate(node_softmax.output):
        node_softmax.output[idx] = node_t2.output[0]
    nodes = [node for node in nodes if node.name not in ["/3/Transpose", "/3/Transpose_1"]]
    onnx_model.graph.ClearField("node")
    onnx_model.graph.node.extend(nodes)
    onnx.save(onnx_model, f"{org_name}/model_hypernet.onnx")
    onnx.checker.check_model(onnx_model)
    print(f"Hypernet of SCN Pytorch --> ONNX: saved in {org_name}/model_hypernet.onnx")
    print('ONNX model_hypernet successfully modified')

    # save base models
    print(model.parameter_name_list)

    extracted_models = extract_all_base_models(model)

    for d in range(model.dimensions):
        dummy_input = torch.randn(img_shape)
        torch.onnx.export(
                model=extracted_models[d].cpu(),
                args=dummy_input,
                f=f"{org_name}/bsmodel{d}.onnx",
                verbose=False,
                export_params=True,
                do_constant_folding=False,
                input_names=['input'],
                opset_version=10,
                output_names=['output'])
        onnx_model = onnx.load(f"{org_name}/bsmodel{d}.onnx")

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
        onnx.save(onnx_model, f"{org_name}/bsmodel{d}.onnx")
        onnx.checker.check_model(onnx_model)
        print(f'ONNX base-model{d} successfully converted Pytorch --> ONNX: {org_name}/bsmodel{d}.onnx')


    stop = timeit.default_timer()
    print('Time: ', stop - start)


if __name__ == '__main__':
    main()
