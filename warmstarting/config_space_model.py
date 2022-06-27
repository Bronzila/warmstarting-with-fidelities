from tokenize import String
from typing import List, Tuple
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from torch import optim

class ConfigSpaceModel():
    """Defines two configuration space: hp_config_space, fidelitiy_space
    """
    def __init__(self) -> None:
        self._config_space: CS.ConfigurationSpace = None

    def setup_config_space(
        self,
        seed: int,
        lr_list: List[float],
        momentum_list: List[float],
        optim_list: List[optim.Optimizer],
        epoch_list:List[int]) -> CS.ConfigurationSpace:

        """ Defines a conditional hyperparameter search-space.
        Parameters
        ----------
        lr_list : List[float]
            Learning rate list that has the minimum and maximum learning rates
            Must contain maximum 2 elements.
            lr_list[0] must be > lr_list[1]

        momentum : List[float]
            momentum rate for optimizers that has the minimum and maximum learning rates
            Must contain maximum 2 elements.
            momentum_list[0] must be > momentum_list[1]

        optim_list: List[torch.optim.Optimizer]
            List of possible optimizers

        epoch_list : List
            epoch_list rate list that has the minimum and maximum learning rates
            Must contain maximum 2 elements.
            epoch_list[0] must be > epoch_list[1]

        Returns:
            Configurationspace ("lr", "momentum", "optimizer", "epoch")
        """
        if not isinstance(lr_list, list):
            raise ValueError("setup_config_space: [lr_list] must be a list")

        if len(lr_list) != 2:
            raise ValueError("setup_config_space: [lr_list] must have 2 elements")

        if lr_list[0] >= lr_list[1]:
            raise ValueError("setup_config_space: lr_list[0] must be > lr_list[1]")

        if not isinstance(momentum_list, list):
            raise ValueError("setup_config_space: [momentum_list] must be a list")

        if len(momentum_list) != 2:
            raise ValueError("setup_config_space: [momentum_list] must have 2 elements")

        if momentum_list[0] >= momentum_list[1]:
            raise ValueError("setup_config_space: [momentum_list] momentum_listlr[0] must be > momentum_list[1]")

        if not isinstance(optim_list, list):
            raise ValueError("setup_config_space: [optim_list] must be a list")

        if not isinstance(epoch_list, list):
            raise ValueError("setup_config_space: [epoch_list] must be a list")

        if len(epoch_list) != 2:
            raise ValueError("setup_config_space: [epoch_list] must have 2 elements")

        if epoch_list[0] >= epoch_list[1]:
            raise ValueError("setup_config_space: epoch_list[0] must be > epoch_list[1]")

        self.seed = seed
        self.lr_list = lr_list
        self.momentum_list = momentum_list
        self.optim_list = momentum_list
        self.epoch_list = epoch_list

        self._config_space = CS.ConfigurationSpace(seed=seed)
        lr = CSH.UniformFloatHyperparameter('lr', lower=lr_list[0], upper=lr_list[1],
            log=True)
        momentum = CSH.UniformFloatHyperparameter('momentum', lower=momentum_list[0], upper=momentum_list[1], 
            log=True)
        optimizer = CSH.CategoricalHyperparameter('optimizer', choices=optim_list)
        epoch = CSH.UniformIntegerHyperparameter('epoch', lower=epoch_list[0], upper=epoch_list[1], log=True)
        self._config_space.add_hyperparameters([lr, momentum, optimizer, epoch])
        
        return self._config_space

    def get_config_spaces(self, fidelities: List[String] = []) -> Tuple[CS.ConfigurationSpace, CS.ConfigurationSpace]:
        """ Splits the _config_space into two config spaces and returns hp_space and fidelity_space
        Parameters
        ----------
        fidelities : List
            fidelity names to split from the _config_space

        Returns:
            Configurationspace
            Configurationspace
        """

        if self._config_space is None:
            raise ValueError("split_spaces: _config_space is None, setup_config_space must be called first.")

        hp_space = CS.ConfigurationSpace(seed=self.seed)
        fidelity_space = CS.ConfigurationSpace(seed=self.seed)
        for param in self._config_space.keys():
            if (param in fidelities):
                fidelity_space.add_hyperparameter(self._config_space[param])
            else:
                hp_space.add_hyperparameter(self._config_space[param])

        return hp_space, fidelity_space
