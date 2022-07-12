import argparse
import torch
import torch.optim as optim
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
        epoch_bounds: List[int],
        subset_bounds: List[float],
        config_space: List[str],
        fidelity_space: List[str],
        epoch_steps: List[int],
        subset_ratios: List[float],
        use_checkpoints: bool,
        shuffle: bool,
        only_train_on_new: bool,
        dataset_id: int = 61,
        results_file_name: str = "results"
):
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(_device)

    seed = 10

    config_space_model = ConfigSpaceModel(seed)
    config_space_model.setup_config_space(lr, momentum, optimizer, epoch_bounds, subset_bounds)
    config, fidelity = config_space_model.get_config_spaces(config_space, fidelity_space)

    handler = DataHandler()
    handler.set_dataset(dataset_id) # iris

    problem = DummyBench(handler, config, fidelity, model_type, criterion, device,
                         rng=seed, use_checkpoints=use_checkpoints, shuffle=shuffle, only_new=only_train_on_new)

    random_search(problem, subset_ratios=subset_ratios, epochs=epoch_steps, results_file_name=results_file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main loop of HPO with warmstarting over fidelities')

    parser.add_argument('--model_type', type=str, help='type of model to be trained', default="FF")
    parser.add_argument('--lr', help='list of initial learning rates to sample from', default=[1e-2, 1e-3, 1e-4])
    parser.add_argument('--momentum', help='list of momentum values to sample from', default=[0.1])
    parser.add_argument('--optimizer', help='list of optimizers to sample from', default=[optim.Adam, optim.SGD])
    parser.add_argument('--lr_sched', help='list of lr scheduler or None', required=False)
    parser.add_argument('--criterion', help='criterion to evaluate the loss', default=torch.nn.CrossEntropyLoss)
    parser.add_argument('--epoch_bounds', help='list of min and max epochs to sample from', default=[3, 10])
    parser.add_argument('--subset_bounds', help='list of min and max subset ratios to sample from', default=[0.2, 1])
    parser.add_argument('--epoch_steps', help='list of epoch steps to use', default=[10])
    parser.add_argument('--subset_steps', help='list of subset ratio steps to use', default=[0.2, 0.4, 0.6, 0.8, 1])
    parser.add_argument('--use_checkpoints', help='use checkpointing or not', default=True)
    parser.add_argument('--shuffle', help='shuffle data subsets', default=False)
    parser.add_argument('--only_train_on_new', help='only train on new data', default=False)
    parser.add_argument('--fidelity_space', type=List[str], help='list of names belonging to the fidelity space',
                        default=['epoch'])
    parser.add_argument('--config_space', type=List[str], help='list of names belonging to the configuration space',
                        default=['lr', 'momentum', 'optimizer'])
    parser.add_argument('--batch_size', help='size of batches to run', default=10)
    parser.add_argument('--dataset_id', help='ID of the OpenML dataset', required=False, default=61)

    args = parser.parse_args()

    HPOLoop(args.model_type, args.lr, args.momentum,
            args.optimizer, args.lr_sched, args.criterion, args.epoch_bounds, args.subset_bounds,
            args.config_space, args.fidelity_space, args.epoch_steps, args.subset_ratios, args.use_checkpoints,
            args.shuffle, args.only_train_on_new, args.dataset_id)
