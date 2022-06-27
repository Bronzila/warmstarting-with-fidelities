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
    def __init__(self, train_dataloader: DataLoader, valid_dataloader: DataLoader,
                 writer: SummaryWriter, rng: Union[np.random.RandomState, int, None] = None,):
        super(DummyBench, self).__init__(train_dataloader, valid_dataloader, writer, rng)

    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[Dict, CS.Configuration, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        pass

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        cs = ConfigurationSpace()

        lr = CS.hyperparameters.CategoricalHyperparameter('lr', choices=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6])

        cs.add_hyperparameter(lr)

        return cs

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
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
        return model

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







