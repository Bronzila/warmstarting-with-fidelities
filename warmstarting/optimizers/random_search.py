import ConfigSpace
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from warmstarting.testbench import WarmstartingBenchTemplate
from warmstarting.fixed_configuration import load_configurations


def random_search(
        problem: WarmstartingBenchTemplate,
        subset_ratios,
        epochs
        ) -> None:
    """ Implementation of a random search algorithm

    Parameters
    ----------
    problem: WarmstartingBenchTemplate
        Testbench containing an objective function and
    subset_ratios: list[float]
        Subset ratio steps to train configs on
    epochs: list[int]
        Epoch steps to train configs on
    """

    problem.get_configuration_space()
    cs = problem.get_configuration_space()
    fs = problem.get_fidelity_space()

    configs = load_configurations(cs)

    for config in configs:
        for epoch_step in epochs:
            for subset_step in subset_ratios:
                fidelity = ConfigSpace.Configuration(fs, values={
                    "epoch": epoch_step,
                    "data_subset_ratio": subset_step
                })
                results = problem.objective_function(config, fidelity)
        # results.append((model, sampled_config, val_errors))
    # save_result('random_result', results)
