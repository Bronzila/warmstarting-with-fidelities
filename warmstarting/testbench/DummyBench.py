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

from warmstarting.testbench.WarmstartingBenchTemplate import WarmstartingBenchTemplate


class DummyBench(WarmstartingBenchTemplate):
    def __init__(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, rng: Union[np.random.RandomState, int, None] = None,):
        super(DummyBench, self).__init__(train_dataloader, valid_dataloader, rng)

    def objective_function_test(self, configuration: Union[CS.Configuration, Dict],
                                fidelity: Union[Dict, CS.Configuration, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        pass

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        cs = ConfigurationSpace()

        p1 = OrdinalHyperparameter("hyp_1", [16, 32, 64, 128, 256, 512])
        p2 = OrdinalHyperparameter("hyp_2", [16, 32, 64, 128, 256, 512])

        cs.add_hyperparameter(p1)
        cs.add_hyperparameter(p2)

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
            nn.Linear(50, 10)
        )
        return model

    def init_optim(self, param: nn.Parameter, config: Union[CS.Configuration, Dict],
                   fidelity: Union[CS.Configuration, Dict, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None) -> optim.Optimizer:
        return optim.SGD(param, lr=1e-3)

    def init_lr_sched(self, optimizer: optim.Optimizer, config, fidelity, rng):
        return optim.lr_scheduler.ConstantLR(optimizer, 0.5, 5, -1)

    def init_criterion(self, config, fidelity, rng):
        return nn.CrossEntropyLoss()




