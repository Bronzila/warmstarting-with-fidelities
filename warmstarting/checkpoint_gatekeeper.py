import torch
import os
import ConfigSpace as CS

from datetime import datetime


class CheckpointGatekeeper:
    """This class manages the checkpointing of models

    A model can be saved to the specified location on the filesystem. When a model is trained, the gatekeeper checks, if
    it has been trained before and if so returns the pretrained weights of the model.

    Parameters
    ----------
    path : str = "./checkpoints"
        Defines base path, where checkpoints are saved
    """

    def __init__(
            self,
            path: str = "./checkpoints"
    ):
        path = os.path.join(path, "run_%s" % datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.base_path = path

        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

        self.last_id = -1
        self.config_store = dict()

    def check_config_saved(self, config: CS.Configuration):
        """ Checks whether the specified config has been saved before

        In this context, saved means, that this config has already been trained and the checkpoint is available on
        the filesystem.

        Parameters
        ----------
        config : CS.Configuration
            Config to check

        Returns
        -------
        Tuple(bool, int)
            Returns True and the id of the config, if it is present in the config_store otherwise False and the
            (invalid) id -1 is returned.
        """
        for id in self.config_store:
            config_equal = True
            for config_key in config.get_dictionary():
                if config.get(config_key) != self.config_store[id].get(config_key):
                    config_equal = False
            if config_equal:
                return True, id
        return False, -1

    def add_config_to_store(self, config: CS.Configuration):
        """ Adds the HP-config to the config_store

        Parameters
        ----------
        config : CS.Configuration
            Config to add to the store

        Returns
        -------
        int
            id of the newly stored config
        """
        self.last_id += 1
        self.config_store[self.last_id] = config
        return self.last_id

    def save_model_state(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                         config: CS.Configuration, lr_scheduler: torch.optim.lr_scheduler = None):
        """ Saves the model state to disk

        Saves the model, optimizer and scheduler state to the filesystem.

        Parameters
        ----------
        model : torch.nn.Module
            Model to save

        optimizer : torch.optim.Optimizer
            Optimizer state to save

        config : CS.Configuration
            Config used for model, optimizer etc.

        lr_scheduler : torch.optim.lr_scheduler = None
            Scheduler of learning rate, if used

        config : CS.Configuration
            The Configuration of the specific model
        """
        # check if config has already been saved
        exists, id = self.check_config_saved(config)

        if not exists:
            id = self.add_config_to_store(config)

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler
        }

        checkpoint_name = "model_" + str(id) + ".pth"
        path = os.path.join(self.base_path, checkpoint_name)

        torch.save(checkpoint, path)
        return id

    def load_model_state(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, config: CS.Configuration):
        """ Checks if configuration config has been trained before and if so returns the weights, otherwise None

        Loads the model, optimizer and scheduler state from the filesystem, if it has been trained before, otherwise
        None is returned.

        Parameters
        ----------
        model : torch.nn.Module
            Model to load state into

        optimizer : torch.optim.Optimizer
            Optimizer to load state into

        config : CS.Configuration
            Config used for model, optimizer etc.

        Returns
        -------
        Tuple(torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler) or None
            None is returned, if no checkpoint exists yet
        """
        exists, id = self.check_config_saved(config)
        if not exists:
            return id, None, None, None

        checkpoint_name = "model_" + str(id) + ".pth"

        checkpoint = torch.load(os.path.join(self.base_path, checkpoint_name))

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler = checkpoint['lr_scheduler']

        return id, model, optimizer, lr_scheduler


if __name__ == "__main__":
    test_keeper = CheckpointGatekeeper()

    lr = CS.hyperparameters.UniformFloatHyperparameter('lr', lower=0.1, upper=1, log=True)
    cs = CS.ConfigurationSpace()
    cs.add_hyperparameter(lr)

    config = cs.sample_configuration()
    model = torch.nn.Linear(10, 10)
    optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    # lr_sched = torch.optim.lr_scheduler.LinearLR(optim)

    test_keeper.save_model_state(model, optim, config)

    optim.param_groups[0]['lr'] = 1

    test_keeper.load_model_state(model, optim, config)

    assert optim.param_groups[0]['lr'] == 0.001

    config2 = cs.sample_configuration()
    test_keeper.save_model_state(model, optim, config2)
