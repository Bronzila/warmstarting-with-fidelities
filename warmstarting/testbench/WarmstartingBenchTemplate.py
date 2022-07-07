import time
from typing import Union, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score, Precision, MetricCollection
import ConfigSpace as CS
from warmstarting.data_loader import DataHandler

from warmstarting.testbench.AbstractBenchmark import AbstractBenchmark

from torch.utils.tensorboard import SummaryWriter
from warmstarting.checkpoint_gatekeeper import CheckpointGatekeeper


class WarmstartingBenchTemplate(AbstractBenchmark):
    def __init__(self,
                 data_handler: DataHandler,
                 configuration_space: CS.ConfigurationSpace,
                 fidelity_space: CS.ConfigurationSpace,
                 device: torch.device,
                 writer: SummaryWriter,
                 only_new: bool = False,
                 shuffle: bool = False,
                 use_checkpoints: bool = True,
                 rng: Union[np.random.RandomState, int, None] = None):
        """
        This class is a base class for implementing warm starting for HPO
        """
        if isinstance(rng, int):
            self.seed = rng
        else:
            self.seed = self.rng.randint(1, 10**6)

        self.device = device
        self.data_handler = data_handler
        super(AbstractBenchmark).__init__()

        # Observation and fidelity spaces
        self._configuration_space = configuration_space
        self._fidelity_space = fidelity_space

        # Metrics
        metrics = MetricCollection([
            Accuracy().to(self.device), 
            F1Score().to(self.device), 
            Precision().to(self.device)])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')

        self.writer = writer
        self.gk = CheckpointGatekeeper()
        self.valid_epochs = 0

        self.only_train_on_new_data = only_new
        self.shuffle_subset = shuffle
        self.use_checkpoints = use_checkpoints

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
            valid_loss_list.append(loss.cpu().detach().numpy())
            self.valid_epochs += 1
        self.writer.add_scalar("Valid_accuracy_{}_{}".format(config_id, configuration["lr"]), np.mean(valid_loss_list), self.valid_epochs)

        # Valid Acc for the valid dataset
        total_valid_acc = self.valid_metrics["Accuracy"].compute()
        val_loss = 1 - total_valid_acc

        return {
            'train_loss': train_loss,
            'train_cost': train_score_cost,
            'val_loss': val_loss,
            'val_cost': valid_score_cost
        }

    def objective_function_test(self, configuration: CS.Configuration,
                                fidelity: Union[CS.Configuration, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None,
                                **kwargs) -> Dict:
        raise NotImplementedError()

    def get_configuration_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        return self._configuration_space

    def get_fidelity_space(self, seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Fidelity space available --- specifies the fidelity dimensions
        """
        return self._fidelity_space

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
        
        # initializing model
        model = self.init_model(config, fidelity, rng)
        optimizer = self.init_optim(model.parameters(), config, fidelity, rng)
        lr_sched = self.init_lr_sched(optimizer, config, fidelity, rng)

        # call the loader
        saved_fidelitiy = None
        if self.use_checkpoints:
            model, optimizer, lr_scheduler, saved_fidelitiy = self.gk.load_model_state(model, optimizer, config)

        # It it doesnt yet exist
        if not model:
            model = self.init_model(config, fidelity, rng)
            optimizer = self.init_optim(model.parameters(), config, fidelity, rng)
            lr_sched = self.init_lr_sched(optimizer, config, fidelity, rng)

        if "data_subset_ratio" in fidelity:
            old_ratio = 0
            if saved_fidelitiy is not None:
                old_ratio = saved_fidelitiy["data_subset_ratio"]
            new_ratio = fidelity["data_subset_ratio"]
            self.train_dataloader, self.valid_dataloader = \
                self.data_handler.get_train_and_val_set(batch_size=10, device=self.device,
                                                        shuffle_subset=self.shuffle_subset,
                                                        only_new_data=self.only_train_on_new_data,
                                                        old_ratio=old_ratio, new_ratio=new_ratio)
        else:
            self.train_dataloader, self.valid_dataloader = \
                self.data_handler.get_train_and_val_set(batch_size=10, device=self.device,
                                                        shuffle_subset=self.shuffle_subset,
                                                        only_new_data=self.only_train_on_new_data)

        # fitting the model with subsampled data
        fit_times, train_losses, train_scores = [], [], []
        for _ in range(fidelity['epoch']):
            start = time.time()
            train_score_cost, train_loss = self.train(model, criterion, optimizer, lr_sched)
            fit_times.append(time.time() - start - train_score_cost)
            train_losses.append(train_loss)
            train_scores.append(train_score_cost)

        fidelity = fidelity.get_dictionary()
        if saved_fidelitiy is not None:
            # add difference to fidelity dict for proper updating
            fidelity["data_subset_ratio"] = fidelity["data_subset_ratio"] - saved_fidelitiy["data_subset_ratio"]
            fidelity = self.add_total_fidelity(fidelity, saved_fidelitiy)

        config_id = self.gk.save_model_state(model, optimizer, config, lr_sched, fidelity)

        return config_id, model, fit_times, train_losses, train_scores

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

    @staticmethod
    def add_total_fidelity(current, saved):
        """ This method adds every fidelity value from our saved fidelity space to our current one
        """
        for c, s in zip(sorted(current), sorted(saved)):
            current[c] += saved[s]
        return current
