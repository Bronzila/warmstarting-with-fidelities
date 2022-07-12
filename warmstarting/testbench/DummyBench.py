from __future__ import annotations

from typing import Dict, Union, Tuple
import ConfigSpace as CS
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from warmstarting.data_loader import DataHandler
from warmstarting.testbench.WarmstartingBenchTemplate import WarmstartingBenchTemplate


class DummyBench(WarmstartingBenchTemplate):
    def __init__(self,
                 data_handler: DataHandler,
                 configuration_space: CS.ConfigurationSpace,
                 fidelity_space: CS.ConfigurationSpace,
                 model_type: str,
                 criterion: torch.nn.Module,
                 device: torch.device,
                 only_new: bool = False,
                 shuffle: bool = False,
                 use_checkpoints: bool = True,
                 rng: Union[np.random.RandomState, int, None] = None,):
        super(DummyBench, self).__init__(data_handler, configuration_space, fidelity_space, device, only_new, shuffle, use_checkpoints, rng)
        self.model_type = model_type
        self.criterion = criterion

    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[Dict, CS.Configuration, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        pass

    def get_meta_information(self) -> Dict:
        pass

    def init_model(self, config: Union[CS.Configuration, Dict], fidelity: Union[CS.Configuration, Dict, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None):
        if self.model_type == "FF":
            model = nn.Sequential(
                nn.Linear(4, 50),
                nn.ReLU(),
                nn.Linear(50, 50),
                nn.ReLU(),
                nn.Linear(50, 3),
                nn.Softmax(dim=1)
            )
        elif self.model_type == "CNN_MNIST":
            model = nn.Sequential()
            in_channels = 1
            out_dim = 28
            num_filters_per_layer = [3, 5]
            conv_kernel_size = 5
            pool_kernel_size = 2

            for i, num_filters in enumerate(num_filters_per_layer):
                model.add_module(f"conv{i}", nn.Conv2d(in_channels, num_filters, conv_kernel_size, stride=1, padding=0))
                out_dim = (out_dim - conv_kernel_size) + 1
                model.add_module(f"act{i}", nn.ReLU())
                model.add_module(f"pool{i}", nn.MaxPool2d(pool_kernel_size))
                out_dim = int((out_dim - pool_kernel_size) / pool_kernel_size + 1)
                in_channels = num_filters

            model.add_module("flat", Flatten())
            # out_features is equal to channels of last layer * height of output * width of output
            out_features = \
                num_filters_per_layer[-1] * (out_dim * out_dim)

            model.add_module("fc1", nn.Linear(out_features, 10))
            model.add_module("fc2", nn.LogSoftmax(1))

        else:
            raise ValueError("Type of model false or unspecified")
        return model.to(self.device)

    def init_optim(self, param: nn.Parameter, config: Union[CS.Configuration, Dict],
                   fidelity: Union[CS.Configuration, Dict, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None) -> optim.Optimizer:
        return config["optimizer"](param, lr=config["lr"])

    def init_lr_sched(self, optimizer: optim.Optimizer, config: CS.Configuration,
                      fidelity: Union[CS.Configuration, None] = None,
                      rng: Union[int, np.random.RandomState, None] = None):
        return None
        # return optim.lr_scheduler.ConstantLR(optimizer, 0.5, 5, -1)

    def init_criterion(self, config: CS.Configuration, fidelity: Union[CS.Configuration, None] = None,
                       rng: Union[int, np.random.RandomState, None] = None):
        return self.criterion()


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)
