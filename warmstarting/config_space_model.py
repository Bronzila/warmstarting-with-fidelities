from tokenize import String
from typing import List, Tuple
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from torch import optim


class ConfigSpaceModel:
    def __init__(self, seed) -> None:
        """
        Defines two configuration space: hp_config_space, fidelitiy_space
        Parameters
        ----------
        seed: int
            Fixed seed to make results reproducible
        """
        self._config_space: CS.ConfigurationSpace = None
        self.seed = seed

    def setup_config_space(
            self,
            lr_list: List[float],
            momentum_list: List[float],
            optim_list: List[optim.Optimizer],
            epoch_list: List[int],
            data_subset_list: List[float]) -> CS.ConfigurationSpace:

        """ Defines a conditional hyperparameter search-space.
        Parameters
        ----------
        lr_list: List[float]
            Learning rate list that has the learning rates to sample from

        momentum_list: List[float]
            momentum rate for optimizers that has the minimum and maximum learning rates
            Must contain maximum 2 elements.
            momentum_list[0] must be > momentum_list[1]

        optim_list: List[torch.optim.Optimizer]
            List of possible optimizers

        epoch_list: List
            epoch_list rate list that has the minimum and maximum of epochs
            Must contain maximum 2 elements.
            epoch_list[0] must be > epoch_list[1]

        data_subset_list: List
            list that has the minimum and maximum of data subset percentage

        ----------
        :return: ConfigurationSpace ("lr", "momentum", "optimizer", "epoch")
        """
        if not isinstance(lr_list, list):
            raise ValueError("setup_config_space: [lr_list] must be a list")

        if not isinstance(momentum_list, list):
            raise ValueError("setup_config_space: [momentum_list] must be a list")

        if not isinstance(optim_list, list):
            raise ValueError("setup_config_space: [optim_list] must be a list")

        if not isinstance(epoch_list, list):
            raise ValueError("setup_config_space: [epoch_list] must be a list")

        if not isinstance(data_subset_list, list):
            raise ValueError("setup_config_space: [epoch_list] must be a list")

        self._config_space = CS.ConfigurationSpace(seed=self.seed)
        lr = CS.hyperparameters.CategoricalHyperparameter('lr', choices=lr_list)
        momentum = CS.hyperparameters.CategoricalHyperparameter('momentum', choices=momentum_list)
        optimizer = CSH.CategoricalHyperparameter('optimizer', choices=optim_list)
        epoch = CSH.UniformIntegerHyperparameter('epoch', lower=epoch_list[0], upper=epoch_list[1])
        data_subset_size = CSH.UniformFloatHyperparameter('data_subset_size', lower=data_subset_list[0], upper=data_subset_list[1])
        self._config_space.add_hyperparameters([lr, momentum, optimizer, epoch, data_subset_size])

        return self._config_space

    def get_config_spaces(self, config_names: List[str], fidelities: List[str]) -> Tuple[CS.ConfigurationSpace, CS.ConfigurationSpace]:
        """ Splits the _config_space into two config spaces and returns hp_space and fidelity_space
        Parameters
        ----------
        fidelities : List
            fidelity names to split from the _config_space

        Returns:
            Configurationspace
            Configurationspace

        Raises
        ------
        ValueError
            self._config_space must be initalized before calling this function
        """

        if self._config_space is None:
            raise ValueError("split_spaces: _config_space is None, setup_config_space must be called first.")

        hp_space = CS.ConfigurationSpace(seed=self.seed)
        fidelity_space = CS.ConfigurationSpace(seed=self.seed)
        for param in self._config_space.get_hyperparameter_names():
            if param in fidelities:
                fidelity_space.add_hyperparameter(self._config_space.get_hyperparameter(param))
            elif param in config_names:
                hp_space.add_hyperparameter(self._config_space.get_hyperparameter(param))

        return hp_space, fidelity_space
