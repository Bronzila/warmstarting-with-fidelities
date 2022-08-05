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

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x