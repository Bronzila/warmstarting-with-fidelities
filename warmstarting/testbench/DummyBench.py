from __future__ import annotations

from typing import Dict, Union, Tuple
import ConfigSpace as CS
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from warmstarting.data_loader import DataHandler
from warmstarting.testbench.WarmstartingBenchTemplate import WarmstartingBenchTemplate
from warmstarting.models.resnet import *
from warmstarting.models.simple_cnn import *
from warmstarting.models.simple_mlp import *

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
                 rng: Union[np.random.RandomState, int, None] = None,
                 num_classes: int = 10,):
        super(DummyBench, self).__init__(data_handler, configuration_space, fidelity_space, device, only_new, shuffle, use_checkpoints, rng)
        self.model_type = model_type
        self.criterion = criterion
        self.num_classes = num_classes

    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[Dict, CS.Configuration, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        pass

    def get_meta_information(self) -> Dict:
        pass

    def init_model(self, config: Union[CS.Configuration, Dict], fidelity: Union[CS.Configuration, Dict, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None):
        if self.model_type == "FF":
            model = SimpleMLP()
        elif self.model_type == "SIMPLE-CNN":
            model = SimpleCNN()
        elif self.model_type == "CNN":
            model = CNN()
        elif self.model_type == "RESNET-18":
            model = ResNet18(num_classes=self.num_classes)
        elif self.model_type == "RESNET-34":
            model = ResNet34(num_classes=self.num_classes)
        elif self.model_type == "RESNET-50":
            model = ResNet50(num_classes=self.num_classes)
        elif self.model_type == "RESNET-101":
            model = ResNet101(num_classes=self.num_classes)
        elif self.model_type == "RESNET-152":
            model = ResNet152(num_classes=self.num_classes)
        
        else:
            raise ValueError("Type of model false or unspecified")
        return model.to(self.device)

    def init_optim(self, param: nn.Parameter, config: Union[CS.Configuration, Dict],
                   fidelity: Union[CS.Configuration, Dict, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None) -> optim.Optimizer:
        if type(config["optimizer"]) is optim.SGD:
            return config["optimizer"](param, lr=config["lr"], momentum=config["momentum"])
        else:
            return config["optimizer"](param, lr=config["lr"])

    def init_lr_sched(self, optimizer: optim.Optimizer, config: CS.Configuration,
                      fidelity: Union[CS.Configuration, None] = None,
                      rng: Union[int, np.random.RandomState, None] = None):
        return None
        # return optim.lr_scheduler.ConstantLR(optimizer, 0.5, 5, -1)

    def init_criterion(self, config: CS.Configuration, fidelity: Union[CS.Configuration, None] = None,
                       rng: Union[int, np.random.RandomState, None] = None):
        return self.criterion()
