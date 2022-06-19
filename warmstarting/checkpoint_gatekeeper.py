import torch
import os

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
            os.mkdir(self.base_path)

    def save_model_state(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int,
                         lr_scheduler: torch.optim.lr_scheduler = None, overwrite: bool = True):
        """ Saves the model state to disk

        Saves the model, optimizer and scheduler state to the filesystem. If the parameter overwrite is True, the latest
        state is overwritten, otherwise every checkpoint is kept.

        Parameters
        ----------
        model : torch.nn.Module
            Model to save

        optimizer : torch.optim.Optimizer
            Optimizer state to save

        epoch : int
            Current epoch

        lr_scheduler : torch.optim.lr_scheduler = None
            Scheduler of learning rate, if used

        overwrite : bool = True
            If True, existing checkpoints will be overwritten.
        """
        # TODO: Properly name the model to be sure that it is unique
        opt_name = type(optimizer).__name__
        checkpoint_name_prefix = opt_name + "_" + str(optimizer.defaults['lr']) + "_epoch_"

        if overwrite:
            for file in os.listdir(self.base_path):
                if file.startswith(checkpoint_name_prefix):
                    os.remove(os.path.join(self.base_path, file))

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler
        }

        checkpoint_name = checkpoint_name_prefix + str(epoch) + ".pth"
        path = os.path.join(self.base_path, checkpoint_name)

        torch.save(checkpoint, path)

    def load_model_state(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        """ Checks if configuration config has been trained before and if so returns the weights, otherwise None

        Saves the model, optimizer and scheduler state to the filesystem. If the parameter overwrite is True, the latest
        state is overwritten, otherwise every checkpoint is kept.

        Parameters
        ----------
        model : torch.nn.Module
            Model to load state into

        optimizer : torch.optim.Optimizer
            Optimizer to load state into

        Returns
        -------
        Tuple(torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler)
        """
        opt_name = type(optimizer).__name__
        checkpoint_name_prefix = opt_name + "_" + str(optimizer.defaults['lr']) + "_epoch_"
        checkpoint_epochs = [int(file.replace(checkpoint_name_prefix, "").replace(".pth", ""))
                             for file in os.listdir(self.base_path)
                             if file.startswith(checkpoint_name_prefix)]

        if len(checkpoint_epochs) == 0:
            return

        if len(checkpoint_epochs) == 1:
            latest_checkpoint = checkpoint_epochs[0]
        else:
            latest_checkpoint = sorted(checkpoint_epochs)[-1]

        checkpoint = torch.load(os.path.join(self.base_path, checkpoint_name_prefix + str(latest_checkpoint) + ".pth"))

        model.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optimizer'])
        lr_scheduler = checkpoint['lr_scheduler']

        return model, optim, lr_scheduler


if __name__ == "__main__":
    test_keeper = CheckpointGatekeeper()
    model = torch.nn.Linear(10, 10)
    optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    # lr_sched = torch.optim.lr_scheduler.LinearLR(optim)

    test_keeper.save_model_state(model, optim, 1)

    optim.param_groups[0]['lr'] = 1

    test_keeper.load_model_state(model, optim)
