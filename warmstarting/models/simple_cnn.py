import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = nn.Sequential()
        in_channels = 1
        out_dim = 28
        num_filters_per_layer = [3, 3]
        conv_kernel_size = 3
        pool_kernel_size = 2

        for i, num_filters in enumerate(num_filters_per_layer):
            self.model.add_module(f"conv{i}", nn.Conv2d(in_channels, num_filters, conv_kernel_size, stride=1, padding=0))
            out_dim = (out_dim - conv_kernel_size) + 1
            self.model.add_module(f"act{i}", nn.ReLU())
            self.model.add_module(f"pool{i}", nn.MaxPool2d(pool_kernel_size))
            out_dim = int((out_dim - pool_kernel_size) / pool_kernel_size + 1)
            in_channels = num_filters

        self.model.add_module("flat", Flatten())
        # out_features is equal to channels of last layer * height of output * width of output
        out_features = \
            num_filters_per_layer[-1] * (out_dim * out_dim)

        self.model.add_module("fc1", nn.Linear(out_features, 30))
        self.model.add_module("relu", nn.ReLU())
        self.model.add_module("fc2", nn.Linear(30, num_classes))

    def forward(self, x):
        return self.model(x)
