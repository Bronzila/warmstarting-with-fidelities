import time
import ConfigSpace
import numpy as np
from warmstarting.testbench import WarmstartingBenchTemplate
from warmstarting.fixed_configuration import load_configurations
from warmstarting.utils.serialization import serialize_results

def random_search(
        problem: WarmstartingBenchTemplate,
        subset_ratios,
        epochs,
        results_file_name: str
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
    results_file_name
        name of the results file
    """

    problem.get_configuration_space()
    cs = problem.get_configuration_space()
    fs = problem.get_fidelity_space()

    configs = load_configurations(cs)

    performance_data = np.zeros((len(configs), len(subset_ratios)), dtype=list)
    fit_time_data = np.zeros((len(configs), len(subset_ratios)), dtype=list)
    epoch_list = np.zeros((len(configs), len(subset_ratios)), dtype=list)
    subset_list = np.zeros((len(configs), len(subset_ratios)), dtype=list)
    time_step_list = np.zeros((len(configs), len(subset_ratios)), dtype=list)

    for i, config in enumerate(configs):
        for x, (subset_step, epoch_step) in enumerate(zip(subset_ratios, epochs)):
            fidelity = ConfigSpace.Configuration(fs, values={
                "epoch": epoch_step,
                "data_subset_ratio": subset_step
            })
            results = problem.objective_function(config, fidelity)
            performance_data[i, x] = results['val_loss']
            epoch_list[i, x] = epoch_step
            subset_list[i, x] = subset_step
            fit_time_data[i, x] = results['train_cost']
            time_step_list[i, x] = results['time_step']

            score = {
                "configs": [],
                "performance": performance_data.tolist(),
                "fit_time": fit_time_data.tolist(),
                "epochs": epoch_list.tolist(),
                "subsets": subset_list.tolist(),
                "time_step": time_step_list.tolist()
            }

    serialize_results(score, configs, file_name=results_file_name)
    return score

