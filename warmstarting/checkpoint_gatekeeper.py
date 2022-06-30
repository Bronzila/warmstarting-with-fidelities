import torch
import os
import re
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

    use_existing_folder : bool = False
        If this option is active, parameter path will be used as model folder and no new folder will be created
    """

    def __init__(
            self,
            path: str = "./checkpoints",
            use_existing_folder: bool = False
    ):
        self.model_dir_prefix = "models_"
        self.model_prefix = "model_"
        self.subset_precision = 1e2
        self.last_id = -1
        self.config_store = dict()
        self.base_path = path

        if use_existing_folder:
            self.populate_store_from_disk()
        else:
            self.base_path = os.path.join(self.base_path, "run_%s" % datetime.now().strftime("%Y%m%d-%H%M%S"))

        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

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
            if self.config_store[id] == config:
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
                         config: CS.Configuration, lr_scheduler: torch.optim.lr_scheduler,
                         fidelities: dict):
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

        lr_scheduler : torch.optim.lr_scheduler
            Scheduler of learning rate

        config : CS.Configuration
            The Configuration of the specific model

        fidelities : dict(str, float/int)
            The fidelities used for the model run
        """
        # check if config has already been saved
        exists, id = self.check_config_saved(config)

        if not exists:
            id = self.add_config_to_store(config)
            os.makedirs(os.path.join(self.base_path, self.model_dir_prefix + str(id)))

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler,
            'fidelity': fidelities,
            'config': self.config_store[id]
        }

        fidelity_substring = ""
        for k in sorted(fidelities):
            val = fidelities[k] * self.subset_precision if k == "subset" else fidelities[k]
            fidelity_substring += "_" + str(val)

        checkpoint_name = self.model_prefix + str(id) + fidelity_substring + ".pth"
        path = os.path.join(os.path.join(self.base_path, self.model_dir_prefix + str(id)), checkpoint_name)

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
            return None

        checkpoint_dir = self.model_dir_prefix + str(id)

        name_map = dict()
        for filename in os.scandir(os.path.join(self.base_path, checkpoint_dir)):
            if filename.is_file():
                # ignore first number, since this is the model id
                fidelities = list(map(int, re.findall("[0-9]+", filename.name)[1:]))
                name_map[filename.name] = sum(fidelities)

        # latest checkpoint has the highest sum of fidelities, since we are training iteratively
        latest_checkpoint_name = max(name_map, key=name_map.get)

        relative_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint_name)
        checkpoint = torch.load(os.path.join(self.base_path, relative_checkpoint_path))

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler = checkpoint['lr_scheduler']
        fidelities = checkpoint["fidelity"]

        return model, optimizer, lr_scheduler, fidelities

    def populate_store_from_disk(self):
        """ Populates the config_store from the specified base_path

        """
        for item in os.scandir(self.base_path):
            if item.is_dir():
                id = int(re.findall("[0-9]+", item.name)[0])
                folder_path = os.path.join(self.base_path, item.name)
                config_filename = next(os.scandir(folder_path)).name
                config = torch.load(os.path.join(folder_path, config_filename))['config']
                self.config_store[id] = config



if __name__ == "__main__":
    test_keeper = CheckpointGatekeeper()

    lr = CS.hyperparameters.UniformFloatHyperparameter('lr', lower=0.1, upper=1, log=True)
    cs = CS.ConfigurationSpace()
    cs.add_hyperparameter(lr)

    config = cs.sample_configuration()
    model = torch.nn.Linear(10, 10)
    optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    # lr_sched = torch.optim.lr_scheduler.LinearLR(optim)

    test_keeper.save_model_state(model, optim, config, None, {'epoch': 10})

    optim.param_groups[0]['lr'] = 1

    test_keeper.load_model_state(model, optim, config)

    assert optim.param_groups[0]['lr'] == 0.001

    config2 = cs.sample_configuration()
    test_keeper.save_model_state(model, optim, config2, None, {'epoch': 10})
