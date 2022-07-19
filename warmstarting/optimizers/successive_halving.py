from typing import List
import ConfigSpace
from warmstarting.testbench import WarmstartingBenchTemplate
from warmstarting.fixed_configuration import load_configurations


def successive_halving(
    problem: WarmstartingBenchTemplate,
    n_models: int,
    min_budget_per_model: float,
    max_budget_per_model: float,
    epoch_bounds : List[int],
    eta: float,
    fixed: bool = False,
    random_seed: int = None,
) -> dict:
    """The successive_halving routine as called by hyperband

    Parameters
    ----------
    problem : Problem
        A problem instance to evaluate on
    n_models : int
        How many models to use
    min_budget_per_model : float
        The minimum budget per model
    max_budget_per_model : float
        The maximum budget per model
    epoch_bounds : List[int]
        bounds of our epochs
    eta : float
        The eta float parameter to use
    fixed : bool = False
        If we use fixed configs or sample some
    Returns
    -------
    model
    """

    cs = problem.get_configuration_space()
    fs = problem.get_fidelity_space()

    if fixed:
        predefined_configs = load_configurations(cs)
        configs = {id: (config, {}) for id, config in enumerate(predefined_configs)}
        n_models = len(predefined_configs)
    else:
        configs = {id: (cs.sample_configuration(), {}) for id in range(n_models)}

    configs_to_eval = list(range(n_models))
    budget = round(min_budget_per_model)

    configs_dict = dict()

    while budget <= max_budget_per_model:
        for id in configs_to_eval:
            fidelity = ConfigSpace.Configuration(fs, values={
                # for now max epochs
                "epoch": epoch_bounds[1],
                "data_subset_ratio": budget / 100
            })
            config, evaluations = configs[id]
            evaluations[budget] = problem.objective_function(config, fidelity)

        num_configs_to_proceed = round(len(configs_to_eval) / eta)

        configs_evaluated_with_budget = \
            dict((id, (config, info)) for id, (config, info) in configs.items() if budget in info)
        configs_dict.update(configs_evaluated_with_budget)

        if not fixed:
            evaluation = lambda _id: configs[_id][1][budget].y
            configs_to_eval.sort(key=lambda i: evaluation(i))
            configs_to_eval = configs_to_eval[:num_configs_to_proceed]

        budget = budget * eta

    return configs_dict
