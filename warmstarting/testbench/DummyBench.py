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
from warmstarting.data_loader import DataHandler

from warmstarting.testbench.WarmstartingBenchTemplate import WarmstartingBenchTemplate


class DummyBench(WarmstartingBenchTemplate):
    def __init__(self,
                 data_handler: DataHandler,
                 configuration_space: CS.ConfigurationSpace,
                 fidelity_space: CS.ConfigurationSpace,
                 device: torch.device,
                 writer: SummaryWriter,
                 only_new: bool = False,
                 shuffle: bool = False,
                 rng: Union[np.random.RandomState, int, None] = None,):
        super(DummyBench, self).__init__(data_handler, configuration_space, fidelity_space, device, writer, only_new, shuffle, rng)

    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[Dict, CS.Configuration, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        pass

    def get_meta_information(self) -> Dict:
        pass

    def init_model(self, config: Union[CS.Configuration, Dict], fidelity: Union[CS.Configuration, Dict, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None):
        model = nn.Sequential(
            nn.Linear(4, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 3),
            nn.Softmax(dim=1)
        )
        return model.to(self.device)

    def init_optim(self, param: nn.Parameter, config: Union[CS.Configuration, Dict],
                   fidelity: Union[CS.Configuration, Dict, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None) -> optim.Optimizer:
        return optim.Adam(param, lr=config["lr"])

    def init_lr_sched(self, optimizer: optim.Optimizer, config: CS.Configuration,
                      fidelity: Union[CS.Configuration, None] = None,
                      rng: Union[int, np.random.RandomState, None] = None):
        return None
        # return optim.lr_scheduler.ConstantLR(optimizer, 0.5, 5, -1)

    def init_criterion(self, config: CS.Configuration, fidelity: Union[CS.Configuration, None] = None,
                       rng: Union[int, np.random.RandomState, None] = None):
        return nn.CrossEntropyLoss()







