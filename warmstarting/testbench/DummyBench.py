from __future__ import annotations

from itertools import accumulate
from typing import Dict, Union, Tuple

import ConfigSpace as CS
import numpy as np

from ConfigSpace import Configuration, ConfigurationSpace, OrdinalHyperparameter, CategoricalHyperparameter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from warmstarting.testbench.WarmstartingBenchTemplate import WarmstartingBenchTemplate


class DummyBench(WarmstartingBenchTemplate):
    def __init__(self,
                 train_dataloader: DataLoader,
                 valid_dataloader: DataLoader,
                 configuration_space: CS.ConfigurationSpace,
                 fidelity_space: CS.ConfigurationSpace,
                 model_type: str,
                 criterion: torch.nn.Module,
                 device: torch.device,
                 writer: SummaryWriter,
                 rng: Union[np.random.RandomState, int, None] = None,):
        super(DummyBench, self).__init__(train_dataloader, valid_dataloader, configuration_space, fidelity_space, device, writer, rng)
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