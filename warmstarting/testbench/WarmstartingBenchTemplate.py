import time
from typing import Union, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score, Precision, MetricCollection
import ConfigSpace as CS

from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.util.rng_helper import get_rng

from torch.utils.tensorboard import SummaryWriter
from warmstarting.checkpoint_gatekeeper import CheckpointGatekeeper


class WarmstartingBenchTemplate(AbstractBenchmark):
    def __init__(self,
                 train_dataloader: DataLoader,
                 valid_dataloader: DataLoader,
                 writer: SummaryWriter,
                 rng: Union[np.random.RandomState, int, None] = None):
        """
        This class is a base class for implementing warm starting for HPO
        """
        if isinstance(rng, int):
            self.seed = rng
        else:
            self.seed = self.rng.randint(1, 10**6)

        super(AbstractBenchmark).__init__()
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        # Observation and fidelity spaces
        self.fidelity_space = self.get_fidelity_space(self.seed)
        self.configuration_space = self.get_configuration_space(self.seed)

        # Metrics
        metrics = MetricCollection([Accuracy(), F1Score(), Precision()])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')

        self.writer = writer
        self.gk = CheckpointGatekeeper()
        self.valid_epochs = 0

    def objective_function(self, configuration: CS.Configuration,
                           fidelity: Union[CS.Configuration, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:
        """ This function evaluates a config and fidelity on the validation dataset

        Parameters
        ----------
        configuration
        fidelity
        rng
        kwargs

        Returns
        -------

        """
        criterion = self.init_criterion(configuration, fidelity, rng)

        config_id, model, model_fit_time, train_loss, train_score_cost = self._train_objective(configuration, fidelity, criterion, rng)

        valid_score_cost = 0
        valid_loss_list = []
        for i, (X_valid, y_valid) in enumerate(self.valid_dataloader):
            _start = time.time()
            pred = model(X_valid)
            # Valid Acc for one data point
            loss = criterion(pred, y_valid)
            self.valid_metrics(pred, y_valid)
            valid_score_cost += time.time() - _start
            # self.writer.add_scalar("Valid_accuracy_{}".format(config_id), metrics["val_Accuracy"], self.valid_epochs)
            valid_loss_list.append(loss.detach().numpy())
            self.valid_epochs += 1
        self.writer.add_scalar("Valid_accuracy_{}_{}".format(config_id, configuration["lr"]), np.mean(valid_loss_list), self.valid_epochs)

        # Valid Acc for the valid dataset
        total_valid_acc = self.valid_metrics["Accuracy"].compute()
        val_loss = 1 - total_valid_acc

        return {
            'train_loss': train_loss,
            'train_cost': train_score_cost,
            'val_loss': val_loss,
            'val_cost': model_fit_time + valid_score_cost
        }

    def objective_function_test(self, configuration: CS.Configuration,
                                fidelity: Union[CS.Configuration, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None,
                                **kwargs) -> Dict:
        raise NotImplementedError()

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        raise NotImplementedError()
        # # TODO: Call own config space creator
        # cs = CS.ConfigurationSpace()
        # return cs

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Fidelity space available --- specifies the fidelity dimensions
        """
        raise NotImplementedError()
        # TODO: Call own fidelity space creator
        # fidelity_space = CS.ConfigurationSpace(seed=seed)
        # return fidelity_space

    def get_meta_information(self) -> Dict:
        raise NotImplementedError()

    def _train_objective(self,
                         config: CS.Configuration,
                         fidelity: Union[CS.Configuration, None],
                         criterion: nn.Module,
                         rng: Union[np.random.RandomState, int, None] = None):
        """ Function that uses gate keeping for the training loop

        Parameters
        ----------
        config
        fidelity
        criterion
        rng

        Returns
        -------

        """
        if rng is not None:
            rng = get_rng(rng, self.rng)

        # initializing model
        model = self.init_model(config, fidelity, rng)
        optim = self.init_optim(model.parameters(), config, fidelity, rng)

        lr_sched = self.init_lr_sched(optim, config, fidelity, rng)

        # call the loader

        checkpoint = self.gk.load_model_state(model, optim, config)
        if not checkpoint:
            config_id = self.gk.add_config_to_store(config)

        # fitting the model with subsampled data
        start = time.time()
        train_score_cost, train_loss = self.train(model, criterion, optim, lr_sched)
        model_fit_time = time.time() - start - train_score_cost

        config_id = self.gk.save_model_state(model, optim, config, lr_sched)

        return config_id, model, model_fit_time, train_loss, train_score_cost

    def init_model(self, config: Union[CS.Configuration, Dict],
                   fidelity: Union[CS.Configuration, Dict, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None):
        """ Function that returns the model initialized model based on the configuration and fidelity
        """
        raise NotImplementedError()

    def init_optim(self, param: nn.Parameter, config: CS.Configuration,
                   fidelity: Union[CS.Configuration, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None) -> optim.Optimizer:
        """ Function that returns the initialized optimizer based on the models parameters, config and fidelity
        """
        raise NotImplementedError()

    def init_lr_sched(self, optimizer: optim.Optimizer, config: CS.Configuration,
                      fidelity: Union[CS.Configuration, None] = None,
                      rng: Union[int, np.random.RandomState, None] = None):
        """
        Function that returns a learning rate scheduler or None based on config and fidelity

        Parameters
        ----------
        optimizer
        config
        fidelity
        rng
        Returns
        -------
        torch.optim.lr_scheduler or None
        """
        raise NotImplementedError()

    def init_criterion(self, config: CS.Configuration,
                       fidelity: Union[CS.Configuration, None] = None,
                       rng: Union[int, np.random.RandomState, None] = None):
        """ Function that returns the loss criterion for the testbench
        """
        raise NotImplementedError()

    def train(self, model: nn.Module, criterion: nn.Module, optim: torch.optim.Optimizer,
              lr_scheduler: torch.optim.lr_scheduler = None):
        """
        The training function as a pytorch implementation
        Parameters
        ----------
        model
            Model to be optimized
        criterion
            Loss function
        optim
            Optimizer
        config_id
            Internal ID of config, used for plotting
        lr_scheduler
            If Scheduler exists
        """
        train_score_cost = 0
        loss_list = []
        for i, (X_train, y_train) in enumerate(self.train_dataloader):
            optim.zero_grad()
            pred = model(X_train)
            loss = criterion(pred, y_train)
            loss.backward()
            optim.step()

            if i % 5 == 0:
                loss_list.append(loss)
                print("Batch {}; Loss: {}".format(i, loss))

            # Metric
            _start = time.time()
            self.train_metrics(pred, y_train)
            train_score_cost += time.time() - _start

        if lr_scheduler is not None:
            lr_scheduler.step()

        return train_score_cost, loss_list
