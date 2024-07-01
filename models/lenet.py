from torch import nn
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from models.scn import SCN
import torch.nn.functional as F
from typing import Any

class LeNet(nn.Module):
    def __init__(self, n_layers, n_units, n_channels, n_classes=7, wider_factor:int =1):
        super(LeNet, self).__init__()
        self.linear_stack = nn.Sequential(
            # 3x32x32
            nn.Conv2d(in_channels=n_channels, out_channels=4*wider_factor, kernel_size=5, stride=2),
            # 4x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 4x14x14
            nn.Conv2d(in_channels=4*wider_factor, out_channels=12*wider_factor, kernel_size=5, stride=2),
            # 12x10x10
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 12x5x5
            nn.Flatten(1),
            nn.Linear(in_features=12 * 5 * 5*wider_factor, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=n_classes),
        )


    def forward(self, x):
        x = self.linear_stack(x)
        return x

    def linearcurve(self, e0, e1, t, inputs):
        for count, layer in enumerate(self.linear_stack):
            if isinstance(layer, nn.Linear):
                inputs = (1-t) * e0.linear_stack[count](inputs) + \
                         t * e1.linear_stack[count](inputs)
            elif isinstance(layer, nn.Conv2d):
                inputs = (1-t) * e0.linear_stack[count](inputs) + \
                         t * e1.linear_stack[count](inputs)
            else:
                inputs = self.linear_stack[count](inputs)
        return inputs

def lenet(**kwargs: Any) -> LeNet:
    model = LeNet(**kwargs)
    return model

def lenet2xwider(**kwargs: Any) -> LeNet:
    kwargs['wider_factor']=2
    model = LeNet(**kwargs)
    return model

def lenet4xwider(**kwargs: Any) -> LeNet:
    kwargs['wider_factor']=4
    model = LeNet(**kwargs)
    return model

def lenet8xwider(**kwargs: Any) -> LeNet:
    kwargs['wider_factor']=8
    model = LeNet(**kwargs)
    return model


class SCN_LeNet(SCN):

    def __init__(self, hin: int, dimensions: int, n_layers: int, n_units:int, n_channels:int, device, n_classes=7) -> None:
        num_alpha = hin
        base_model = LeNet(n_layers=n_layers, n_units=n_units, n_channels=n_channels, n_classes=n_classes)
        super(SCN_LeNet, self).__init__(num_alpha, dimensions, base_model, device)
        self.num_classes = n_classes

    def forward(self, x, hyper_x):
        hyper_output = self.hyper_stack(hyper_x)

        para_name = iter(self.parameter_name_list)

        # 3x32x32
        logits = F.conv2d(x, weight=self.calculate_weighted_sum(hyper_output, next(para_name)),
                          bias=self.calculate_weighted_sum(hyper_output, next(para_name)),
                          stride=2)
        # 4x32x32
        logits = torch.relu(logits)
        logits = F.max_pool2d(logits, 2)
        # 4x14x14
        logits = F.conv2d(logits, weight=self.calculate_weighted_sum(hyper_output, next(para_name)),
                          bias=self.calculate_weighted_sum(hyper_output, next(para_name)),
                          stride=2)
        # 16x10x10
        logits = torch.relu(logits)
        logits = F.max_pool2d(logits, 2)
        # 16x5x5

        logits  = torch.flatten(logits, 1)
        # fc
        logits = F.linear(logits, weight=self.calculate_weighted_sum(hyper_output, next(para_name)),
                            bias=self.calculate_weighted_sum(hyper_output, next(para_name)),)
        logits = torch.relu(logits)


        logits = F.linear(logits, weight=self.calculate_weighted_sum(hyper_output, next(para_name)),
                            bias=self.calculate_weighted_sum(hyper_output, next(para_name)),)
        return logits
