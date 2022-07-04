import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from warmstarting.testbench.DummyBench import DummyBench
from warmstarting.data_loader import DataHandler
from warmstarting.optimizers.random_search import random_search
from warmstarting.config_space_model import ConfigSpaceModel
from typing import List


def HPOLoop(
        model_type: str,
        lr: List[float],
        momentum: List[float],
        optimizer: List[torch.optim.Optimizer],
        lr_sched: List[torch.nn.Module],
        criterion: torch.nn.Module,
        epoch: List[int],
        config_space: List[str],
        fidelity_space: List[str],
        batch_size: int = 32,
        n_models: int = 20,
        dataset_id: int = 61,
):
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(_device)

    seed = 10
    writer = SummaryWriter()

    config_space_model = ConfigSpaceModel(seed)
    config_space_model.setup_config_space(lr, momentum, optimizer, epoch)
    config, fidelity = config_space_model.get_config_spaces(config_space, fidelity_space)

    handler = DataHandler()
    handler.set_dataset(dataset_id)
    train_dataloader, valid_dataloader = handler.get_train_and_val_set(batch_size=batch_size, device=device)

    problem = DummyBench(train_dataloader, valid_dataloader, config, fidelity, model_type, criterion, device, writer, seed)

    random_search(problem, n_models=n_models)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main loop of HPO with warmstarting over fidelities')

    parser.add_argument('--model_type', type=str, help='type of model to be trained', default="FF")
    parser.add_argument('--lr', help='list of initial learning rates to sample from', default=[1e-2, 1e-3, 1e-4])
    parser.add_argument('--momentum', help='list of momentum values to sample from', default=[0.1])
    parser.add_argument('--optimizer', help='list of optimizers to sample from', default=[optim.Adam, optim.SGD])
    parser.add_argument('--lr_sched', help='list of lr scheduler or None', required=False)
    parser.add_argument('--criterion', help='criterion to evaluate the loss', default=torch.nn.CrossEntropyLoss)
    parser.add_argument('--epoch', help='list of min and max epochs to sample from', default=[3, 10])
    parser.add_argument('--fidelity_space', type=List[str], help='list of names belonging to the fidelity space',
                        default=['epoch'])
    parser.add_argument('--config_space', type=List[str], help='list of names belonging to the configuration space',
                        default=['lr', 'momentum', 'optimizer'])
    parser.add_argument('--n_models', help='number of models to train', default=20)
    parser.add_argument('--batch_size', help='size of batches to run', default=10)
    parser.add_argument('--dataset_id', help='ID of the OpenML dataset', required=False, default=61)

    args = parser.parse_args()

    HPOLoop(args.model_type, args.lr, args.momentum,
            args.optimizer, args.lr_sched, args.criterion, args.epoch,
            args.config_space, args.fidelity_space,
            args.batch_size, args.dataset_id)
