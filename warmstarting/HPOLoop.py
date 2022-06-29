import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from warmstarting.testbench.DummyBench import DummyBench
from warmstarting.data_loader import DataHandler
from warmstarting.optimizers.random_search import random_search
from warmstarting.config_space_model import ConfigSpaceModel

def HPOLoop():
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(_device)

    seed = 10
    writer = SummaryWriter()

    lr_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    momentum_list = [0.1]
    optim_list = [optim.SGD, optim.Adam]
    epoch_list = [3, 10]
    data_subset_list = [0.2, 1]

    config_space_model = ConfigSpaceModel(seed)
    config_space_model.setup_config_space(lr_list, momentum_list, optim_list, epoch_list, data_subset_list)
    config, fidelity = config_space_model.get_config_spaces(["lr", "optimizer"], ["epoch", "data_subset_size"])


    handler = DataHandler()
    handler.set_dataset(61) # iris

    problem = DummyBench(handler, config, fidelity, device, writer, seed)

    random_search(problem, n_models=20)


if __name__ == "__main__":
    HPOLoop()
