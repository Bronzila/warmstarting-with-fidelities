from typing import List
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from torch import optim

class ConfigSpaceModel():
    """Defines two configuration space: hp_config_space, fidelitiy_space
    """
    def __init__(self) -> None:
        self.hp_config_space:CS.ConfigurationSpace = None
        self.fidelity_space:CS.ConfigurationSpace = None

    def setup_config_space(
        self, 
        lr_list: List[float],
        momentum_list: List[float], 
        optim_list: List[optim.Optimizer]) -> CS.ConfigurationSpace:

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
        Returns:
            Configurationspace ("lr", "momentum", "optimizer")
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

        self.hp_config_space = CS.ConfigurationSpace(seed=0)
        lr = CSH.UniformFloatHyperparameter('lr', lower=lr_list[0], upper=lr_list[1], log=True)
        momentum = CSH.UniformFloatHyperparameter('momentum', lower=momentum_list[0], upper=momentum_list[1], log=True)
        optimizer = CSH.CategoricalHyperparameter('optimizer', choices=optim_list)
        self.hp_config_space.add_hyperparameters([lr,momentum, optimizer])
        
        return self.hp_config_space

    def setup_fidelity_space(self, epoch_list:List[int]):
        """ Defines a conditional hyperparameter search-space.
        Parameters
        ----------
        epoch_list : List
            epoch_list rate list that has the minimum and maximum learning rates
            Must contain maximum 2 elements.
            epoch_list[0] must be > epoch_list[1]

        Returns:
            Configurationspace
        """
        if not isinstance(epoch_list, list):
            raise ValueError("setup_config_space: [epoch_list] must be a list")

        if len(epoch_list) != 2:
            raise ValueError("setup_config_space: [epoch_list] must have 2 elements")

        if epoch_list[0] >= epoch_list[1]:
            raise ValueError("setup_config_space: epoch_list[0] must be > epoch_list[1]")

        self.fidelity_space = CS.ConfigurationSpace(seed=0)
        epoch = CSH.UniformIntegerHyperparameter('epoch', lower=epoch_list[0], upper=epoch_list[1], log=True)
        self.fidelity_space.add_hyperparameters([epoch])
        
        return self.fidelity_space

    def sample_hp_config_space_configuration(self, size: int = 1) -> List[CS.configuration_space.Configuration]:
        """ Sample (size) configurations from the hp configuration space object.
        Parameters
        ----------
        size (int, optional) – Number of configurations to sample. Default to 1

        Returns:
            List[Configuration]
        """
        if self.hp_config_space == None:
            raise ValueError("sample_hp_config_space_configuration: hp_config_space is None")
        return self.hp_config_space.sample_configuration(size=size)

    def sample_fidelity_space_configuration(self, size: int = 1) -> List[CS.configuration_space.Configuration]:
        """ Sample (size) configurations from the fidelity configuration space object.
        Parameters
        ----------
        size (int, optional) – Number of configurations to sample. Default to 1

        Returns:
            List[Configuration]
        """
        if self.fidelity_space == None:
            raise ValueError("sample_hp_config_space_configuration: fidelity_space is None")
        return self.fidelity_space.sample_configuration(size=size)