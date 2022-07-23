import torch.optim as optim
from torch import nn

from warmstarting.HPOLoop import HPOLoop


if __name__ == "__main__":
    model_type = "FF"
    lr = [0.00025, 0.0005, 0.00075, 0.001, 0.00125]
    momentum = [0.1, 0.3, 0.6]
    optimizer = [optim.SGD, optim.Adam]
    lr_sched = None
    criterion = nn.CrossEntropyLoss
    epoch_bound = [3, 10]
    subset_bounds = [0.1, 1]
    config_space = ['lr', 'momentum', 'optimizer']
    fidelity_space = ['epoch', 'data_subset_ratio']
    epoch_steps = [10]
    subset_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    use_checkpoints = True
    shuffle = True
    only_train_on_new = False
    dataset_id = 61
    results_file_name = "random_sanity_check"

    seed = 10

    test = HPOLoop(model_type, lr, momentum, optimizer, lr_sched, criterion,
                   epoch_bound, subset_bounds, config_space, fidelity_space,
                   epoch_steps, subset_ratios, use_checkpoints, shuffle,
                   only_train_on_new, seed, dataset_id, results_file_name)

    seed = 100

    test1 = HPOLoop(model_type, lr, momentum, optimizer, lr_sched, criterion,
                   epoch_bound, subset_bounds, config_space, fidelity_space,
                   epoch_steps, subset_ratios, use_checkpoints, shuffle,
                   only_train_on_new, seed, dataset_id, results_file_name)

    assert test["performance"] != test1["performance"]

    test2 = HPOLoop(model_type, lr, momentum, optimizer, lr_sched, criterion,
                    epoch_bound, subset_bounds, config_space, fidelity_space,
                    epoch_steps, subset_ratios, use_checkpoints, shuffle,
                    only_train_on_new, seed, dataset_id, results_file_name)
    assert test1["performance"] == test2["performance"]
