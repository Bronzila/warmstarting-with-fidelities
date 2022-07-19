from typing import List
import numpy as np
from warmstarting.testbench import WarmstartingBenchTemplate
from warmstarting.optimizers.successive_halving import successive_halving


def hyperband(
        problem: WarmstartingBenchTemplate,
        min_budget_per_model: float,
        max_budget_per_model: float,
        epoch_bounds: List[int],
        eta: float,
        fixed: bool,
        results_file_name: str,
        random_seed: int = None
) -> List[dict]:
    """The hyperband algorithm

    Parameters
    ----------
    problem : Problem
        A problem instance to run on
    min_budget_per_model : int
        The minimum budget per model
    max_budget_per_model : int
        The maximum budget per model
    epoch_bounds : List[int]

    eta : float
        The eta float parameter
    random_seed : int | None = None
        The random seed to use
    Returns
    -------
    dict
        The dictionary with the config information
    """
    min_budget_per_model *= 100
    max_budget_per_model *= 100

    s_max = int(np.log(max_budget_per_model / min_budget_per_model) / np.log(eta))
    iterations = reversed(range(s_max + 1))

    configs_dicts = []
    for s in iterations:
        n_models = int((s_max + 1) / (s + 1) * np.power(eta, s))
        r = int(max_budget_per_model / np.power(eta, s))

        configs_dict = successive_halving(
            problem=problem,
            n_models=n_models,
            min_budget_per_model=r,
            max_budget_per_model=max_budget_per_model,
            epoch_bounds=epoch_bounds,
            eta=eta,
            fixed=fixed,
            random_seed=random_seed,
        )
        configs_dicts.append(configs_dict)

    return configs_dicts
