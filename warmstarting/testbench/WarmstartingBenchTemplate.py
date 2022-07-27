import time
from typing import Union, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import ConfigSpace as CS
from warmstarting.data_loader import DataHandler
from warmstarting.testbench.AbstractBenchmark import AbstractBenchmark
from warmstarting.checkpoint_gatekeeper import CheckpointGatekeeper


class WarmstartingBenchTemplate(AbstractBenchmark):
    def __init__(self,
                 data_handler: DataHandler,
                 configuration_space: CS.ConfigurationSpace,
                 fidelity_space: CS.ConfigurationSpace,
                 device: torch.device,
                 only_new: bool = False,
                 shuffle: bool = False,
                 use_checkpoints: bool = True,
                 rng: Union[np.random.RandomState, int, None] = None,
                 batch_size:int = 10,):
        """
        This class is a base class for implementing warm starting for HPO
        """
        if isinstance(rng, int):
            self.seed = rng
        else:
            self.seed = self.rng.randint(1, 10**6)

        self.device = device
        self.data_handler = data_handler
        self.batch_size = batch_size
        super(AbstractBenchmark).__init__()

        # Observation and fidelity spaces
        self._configuration_space = configuration_space
        self._fidelity_space = fidelity_space

        self.gk = CheckpointGatekeeper()
        self.valid_steps = 0

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

        train_loss, train_cost, valid_loss, valid_cost, time_step = self._train_objective(configuration, fidelity, criterion, rng)

        return {
            'train_loss': train_loss,
            'train_cost': train_cost,
            'val_loss': valid_loss,
            'val_cost': valid_cost,
            'time_step': time_step
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
        train_start_timestamp = time.perf_counter()
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
                self.data_handler.get_train_and_val_set(batch_size=self.batch_size, device=self.device,
                                                        shuffle_subset=self.shuffle_subset,
                                                        only_new_data=self.only_train_on_new_data,
                                                        old_ratio=old_ratio, new_ratio=new_ratio,
                                                        seed=self.seed)
        else:
            self.train_dataloader, self.valid_dataloader = \
                self.data_handler.get_train_and_val_set(batch_size=self.batch_size, device=self.device,
                                                        shuffle_subset=self.shuffle_subset,
                                                        only_new_data=self.only_train_on_new_data,
                                                        seed=self.seed)

        # fitting the model with subsampled data
        train_cost_list, train_loss_list = [], []
        valid_cost_list, valid_loss_list = [], []
        time_step_list = []

        # validate once before training to have a value corresponding to train_start timestamp
        valid_loss, valid_cost = self.evaluate(model, criterion)
        valid_loss_list.append(float(valid_loss))
        valid_cost_list.append(valid_cost)
        time_step_list.append(train_start_timestamp)

        for _ in range(fidelity['epoch']):
            train_loss, train_cost = self.train(model, criterion, optimizer, lr_sched)
            train_loss_list.append(train_loss.item())
            train_cost_list.append(train_cost)

            valid_loss, valid_cost = self.evaluate(model, criterion)
            valid_loss_list.append(float(valid_loss))
            valid_cost_list.append(valid_cost)
            time_step_list.append(time.perf_counter())

        fidelity = fidelity.get_dictionary()
        if saved_fidelitiy is not None:
            # add difference to fidelity dict for proper updating
            fidelity["data_subset_ratio"] = fidelity["data_subset_ratio"] - saved_fidelitiy["data_subset_ratio"]
            fidelity = self.add_total_fidelity(fidelity, saved_fidelitiy)

        if self.use_checkpoints:
            config_id = self.gk.save_model_state(model, optimizer, config, lr_sched, fidelity)
        else:
            is_saved, config_id = self.gk.check_config_saved(config)
            if not is_saved:
                config_id = self.gk.add_config_to_store(config)

        # Adjust last time step, to take model saving into account
        time_step_list[-1] = time.perf_counter()

        return train_loss_list, train_cost_list, valid_loss_list, valid_cost_list, time_step_list

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
        train_cost = 0
        train_loss_list = []
        for i, (X_train, y_train) in enumerate(self.train_dataloader):
            X_train = X_train.to(self.device)
            y_train = y_train.to(self.device)
            _start = time.perf_counter()

            optim.zero_grad()
            pred = model(X_train)
            loss = criterion(pred, y_train.long())
            loss.backward()
            optim.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            train_cost += time.perf_counter() - _start

            if i % 5 == 0:
                train_loss_list.append(loss.cpu().detach().numpy())
                print("Batch {}; Loss: {}".format(i, loss))

        return np.mean(train_loss_list), train_cost

    def evaluate(self, model: nn.Module, criterion: nn.Module):
        valid_cost = 0
        valid_loss_list = []
        for i, (X_valid, y_valid) in enumerate(self.valid_dataloader):
            X_valid = X_valid.to(self.device)
            y_valid = y_valid.to(self.device)
            self.valid_steps += 1

            _start = time.perf_counter()
            pred = model(X_valid)
            loss = criterion(pred, y_valid.long())
            valid_cost += time.perf_counter() - _start

            valid_loss_list.append(loss.cpu().detach().numpy())

        return np.mean(valid_loss_list), valid_cost

    @staticmethod
    def add_total_fidelity(current, saved):
        """ This method adds every fidelity value from our saved fidelity space to our current one
        """
        for c, s in zip(sorted(current), sorted(saved)):
            current[c] += saved[s]
        return current
